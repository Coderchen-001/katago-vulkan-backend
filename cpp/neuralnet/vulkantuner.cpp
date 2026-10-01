#ifdef USE_VULKAN_BACKEND

#include "../neuralnet/vulkantuner.h"
#include "../neuralnet/tunestats.h"

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/sha2.h"
#include "../dataio/homedata.h"

#include <chrono>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// The build revision, obtained exactly the way main.cpp obtains it: a macro generated into the
// build directory.  The plan's invalidation table requires "a code change invalidates the
// row set" (t1-schema.md section 4, the buildHash row), and this is the sanction source for it;
// asking the backend instead would need a fifth interface function (R1 forbids that).
#include <program/gitinfo.h>

// For the durable flush in writeFileAtomic(): _commit (the MSVC spelling of fsync, which
// needs only <io.h> -- deliberately NOT the Win32 FlushFileBuffers, whose headers drag in
// windows.h and its macro hazards) on Windows, fsync elsewhere.
#if defined(_WIN32)
#include <io.h>
#include <process.h>
#include <stdio.h>
#else
#include <stdio.h>
#include <unistd.h>
#endif

using namespace std;

namespace VulkanTuner {

  namespace {

    const char* const LOG_PREFIX = "KataGoVulkanTuner: [T3] ";

    // ONE insertion per line, TERMINATOR INCLUDED, like the backend's ggml_vk_t3_line().
    // Two things go wrong otherwise, both measured: `<< a << b` lets another NN server
    // thread land inside this line (even inside its own prefix, which makes the line
    // unfindable), and `<< std::endl` is a SECOND insertion, so two messages can end up on
    // one physical line ("lineAlineB\n").  `LOG_PREFIX` is used HERE and nowhere else,
    // which is what makes that checkable (guardrail R10).
    void logLine(const string& msg) {
      cerr << (string(LOG_PREFIX) + msg + "\n");
    }

    // The ONE place a load refusal is worded.  Every "the file is not usable" path goes
    // through it, which (a) guarantees the wording a harness greps is stable, and (b) keeps
    // the shell-side inventory at ONE occurrence no matter how many reasons are added --
    // a new reason is a new CALL, not a new message shape.
    void refuseLoad(const string& kind, const string& why) {
      logLine("WARNING: " + kind + " refused: " + why + " (fail-open: no tuned rows)");
    }

    // Schema version of anything this file reads or writes.  Bump it when the MEANING of
    // a field changes: a cache written by an older tuner must not be read as if it had
    // today's meaning (t1-schema.md section 1, "VERSION").
    //
    // VERSION 2 (2026-09-13): the outer key moved into the FILE NAME (one file = one key) and
    // ROWS/NEGATIVE became mutually exclusive statements about that key.  There is NO
    // migration and none is needed: no production cache file existed before this (EXPORT was
    // the only writer and only tests asked for it).  A VERSION 1 file is refused loudly,
    // which is right for a file whose rows were not keyed to the name they live under.
    const int SCHEMA_VERSION = 5;   // was 4: the NEGATIVE LINE gained a required SCOPE token
                                    // (`NEGATIVE <reason> <kind>[,<kind>...]`), so an older file's
                                    // marker does not say which kind it is about -- and under this
                                    // batch's rule an unscoped marker would have to be read as
                                    // "some kind", which is exactly the fail-open default the closed
                                    // set refuses.  The REQUIRED header-key set is UNCHANGED: this
                                    // bump is the "same keys, different meaning" case section 1
                                    // describes, which is what VERSION tracks.
                                    // was 3: the REQUIRED header-key set changed (MODE added;
                                    // VERSION/TUNER_VERSION promoted to required).  A file that
                                    // does not declare the new set must not be read as if it did.
    // TUNER_VERSION = the ENUMERATOR's version: it changes when the way candidates are
    // enumerated/measured changes.  This cut is the ENUMERATOR ITSELF (its first writer: the
    // stock-preset catalogue and the measured row source), so it moves with SCHEMA_VERSION.
    const int TUNER_VERSION = 3;

    // The policy baseline counter (t1-schema.md section 4, "baselineVersion").
    //
    // SAME SOURCE as the `p<N>` component of the policy fingerprint: one counter, recorded in
    // two places (the POLICY string and the BASELINE field).  NOT split into a second counter
    // on purpose -- see the note in t1-schema.md section 1.1: two manual counters can each be
    // forgotten, leaving a key that LOOKS maintained, while the diagnostic distinction they
    // would buy ("was this a policy change or a baseline change?") is already given by
    // b<rev16> for anything committed.  Split it later only if the stock baseline becomes
    // mechanically reportable (a backend-reported baseline id), which today would cost the
    // fifth interface name that step 5 wants for probe_candidate.
    //
    // 🔴 BUMP THIS whenever a change alters what an already-tuned row MEANS, without changing
    // any env value and without being caught by the build revision (i.e. an in-place edit of a
    // default, before it is committed):
    //   1. the scalar l/m/s routing thresholds            (ggml_vk_guess_matmul_pipeline)
    //   2. the m<=192 / n<=32 style bucket predicates      (same function)
    //   3. the sentinel set / row-matching rule            (ggml_vk_tuned_row_matches, K1)
    //   4. a fork default that changes a resolved variant  (FA tile default, RMS subgroup
    //                                                       variant, M4 fusion default)
    //   5. the geometry rule table                         (vulkan-tile-validity.h)
    // Committed changes are ALSO covered by b<rev16>; this counter is the belt for the braces,
    // so it must not be the only thing anyone relies on.
    // 🔴 1 -> 2 (2026-10-01): a PROTOCOL change, not a knob change.  `allocateReps` used to let the
    // Gaussian formula decide `reps`; it no longer does -- the budget IS the declared `PAIRS_FLOOR`.  That
    // rule is not a `POLICY_CONSTANTS` row, so the fingerprint does not move by itself, and a row set
    // measured under "the formula may vote" is not the same article as one measured under "the floor is
    // the budget" -- which is exactly what this counter exists to say.
    const int TUNER_POLICY_VERSION = 2;

    // Which header keys are required depends on the CHANNEL, and the split follows what each
    // key GUARANTEES.  These two arrays are the single source of truth: the parser builds its
    // "absent" list from REQUIRED_FOR_CACHE, and R12 in t3-guardrails.py verifies both sets,
    // so moving a key between buckets is a deliberate act that has to go through the guardrail.
    //
    //   REQUIRED_ANY_CHANNEL -- the key carries a guarantee ("the policy changed, so this row
    //     set is void").  Because the parser ACCEPTS a file that is missing a merely-advisory
    //     key, putting such a key in the other bucket would make the guarantee vanish
    //     silently: the file loads, the policy changes, nothing invalidates anything.
    //   REQUIRED_FOR_CACHE -- a cache is written by us, so absence has no legitimate
    //     explanation.  R-29 removed the other channel, and with it the tolerance: an absent
    //     key is a REFUSAL that names the key, and ROWSET_HASH is compared against the hash
    //     of the parsed rows instead of being taken as a hint.
    const char* const REQUIRED_ANY_CHANNEL[] = { "POLICY", "BASELINE" };
    // VERSION / TUNER_VERSION are here to CLOSE the gap that NegativeReasonSpec used to
    // self-document: a file omitting them was neither refused nor reported.  MODE is the
    // user-facing precision request (Decision::modeString).  All three are name-routed exactly
    // like MODEL/BOARD -- the file NAME already carries the key, so the header only has to
    // confirm it: absence is refused on the cache channel -- the one channel left.
    const char* const REQUIRED_FOR_CACHE[] = { "MODEL", "BOARD", "MAXBATCH", "ROWSET_HASH", "CREATED",
                                               "VERSION", "TUNER_VERSION", "MODE" };


    // Row-count cap.  ONE constant used by BOTH the writer and the parser (a cap that only
    // one side enforces is a cap that produces files the other side rejects): a file with
    // more rows than this is refused on read and never produced on write.  v1 is one row
    // per (family, acc), so 64 is room for the multi-row version many times over -- the
    // point of the number is to bound what a corrupt or hostile file can make us allocate.
    const size_t MAX_ROWS_PER_FILE = 64;

    // KATAGO_VK_TUNE: unset or anything other than "0" = enabled.  "0" is the reference
    // configuration and must stay bit-identical to stock, which is why every entry point
    // below starts with this check and returns before doing anything observable.
    bool tuneEnabled() {
      const char* env = getenv("KATAGO_VK_TUNE");
      return !(env != nullptr && strcmp(env, "0") == 0);
    }

    string shortHash(const string& s) {
      char hash[65];
      SHA2::get256(reinterpret_cast<const uint8_t*>(s.data()), s.size(), hash);
      return string(hash).substr(0, 16);
    }

    // Make a *value* safe to place in a line-oriented text format.  The REAL guarantee, stated
    // exactly: every control character (including newline, 0x0A) becomes '_', so a value can
    // never forge a field or start a new line, and POLICY is the LAST field on its line, so
    // any spaces inside a value are harmless.  It is deliberately NOT "whitespace-safe" in
    // general -- do not reuse it as a filename or CSV sanitiser.
    //
    // NOT used for hashing: hashing a sanitised value would make two different raw values
    // collide (`a\x01b` and `a\x02b` both become `a_b`), which is the opposite of what a
    // fingerprint is for.  See policyFingerprint().
    string sanitizeValue(const string& in, size_t maxLen) {
      string out;
      for (char c : in) {
        if (out.size() >= maxLen) {
          break;
        }
        const unsigned char u = (unsigned char) c;
        out += (u < 0x20 || u == 0x7f) ? '_' : c;
      }
      return out;
    }

    // The request-type switch table (t3-shell-plan.md section 6, "P1").  Every switch that ASKS
    // the tuner to do something must be reported when the tuner is not allowed to act,
    // otherwise the operator gets "I set the env var and nothing happened" (lesson #46).  This
    // was hand-written three times before -- IMPORT, then EXPORT, and RETUNE was silently
    // dropped until a review noticed that the guardrail's own inventory count (2) was the
    // proof.  So the list is DATA and t3-guardrails.py R13 pins ALL FOUR FIELDS: a switch
    // added without thought, or a `vacuousWhen` quietly changed, turns the guardrail red.
    //
    //   env               the switch
    //   vacuousWhen       value that means "the TOOL set it, so under a not-allowed tuner
    //                     nothing was really requested" (nullptr = any set value is a request).
    //                     KATAGO_VK_TUNE_PRIME is set by benchmark/genconfig themselves, so
    //                     warning about "auto" on every reference-arm run would train people to
    //                     ignore warnings -- while the reference arm's probe face is itself an
    //                     acceptance criterion.  A non-default PRIME really was asked for.
    //   ignoredUnderOff   is it ALSO ignored when cfg vulkanTune=off?  Both rows left in the
    //                     table (RETUNE, PRIME) ask for TUNING, which a read-only tuner cannot
    //                     do, and both are `true` below.
    //   consequence       what the operator needs to hear.
    struct RequestSwitch {
      const char* env;
      const char* vacuousWhen;
      bool ignoredUnderOff;
      const char* consequence;
    };
    static const RequestSwitch REQUESTS[] = {
      { "KATAGO_VK_TUNE_RETUNE", nullptr, true,  "nothing was re-tuned and the cache was not even read" },
      { "KATAGO_VK_TUNE_PRIME",  "auto",  true,  "no prime-time tuning happened (auto is the tool's own setting)" },
    };

    // Report the request switches that cannot be honoured, under whichever reason applies.
    // ONCE PER PROCESS: this is a property of the configuration, not of one context, and a
    // benchmark thread-count sweep creates one context per bucket -- N identical warnings is
    // how a warning becomes wallpaper.
    void reportIgnoredRequests(bool tunerOff, bool readOnly) {
      if (!tunerOff && !readOnly) {
        return;
      }
      static std::atomic<bool> alreadyReported {false};
      bool expected = false;
      if (!alreadyReported.compare_exchange_strong(expected, true)) {
        return;
      }
      const char* why = tunerOff ? "KATAGO_VK_TUNE=0 disables the tuner"
                                 : "vulkanTune=off makes the tuner read-only";
      for (const RequestSwitch & r : REQUESTS) {
        if (readOnly && !tunerOff && !r.ignoredUnderOff) {
          continue;
        }
        const char* v = getenv(r.env);
        if (v == nullptr || v[0] == '\0') {
          continue;
        }
        if (r.vacuousWhen != nullptr && string(v) == r.vacuousWhen) {
          continue;
        }
        logLine(string("WARNING: ") + r.env + "=" + v + " was requested but " + why
                + "; " + r.consequence
                + (tunerOff ? " (reference run stays clean)" : ""));
      }
    }

    // ONE derivation of "this process is in read-only mode".  Today `policy.off` and the write
    // permission are the same fact, but they would otherwise be two SPELLINGS of it, and the
    // moment "read-only" grows a second meaning (e.g. "do not touch an existing cache either")
    // the two would drift apart.
    bool readOnlyMode(const TunePolicy & policy) {
      return policy.off;
    }

    // THE single place that decides whether this process may write a cache file.
    //
    // Why a named predicate rather than `!policy.off` at each site: a guardrail can only check
    // TEXT, so what it must recognise has to be one NAME (the same reason R1/R3 exist).  Three
    // earlier versions of that guardrail were unsound: "is the string policy.off nearby?" is
    // satisfied by an unrelated `if (policy.off) { }`, and "the first `if (` within N lines" is
    // satisfied by an unrelated `if (cacheWriteAllowed(policy)) { log(); }` while the real write
    // sits outside it -- and it went RED for a write whose enclosing condition was simply more
    // than N lines away, which pushes people to write comments that exist only for the guardrail.
    //
    // KATAGO_VK_TUNE=0 is deliberately not consulted here: it never reaches a decision at all.
    bool cacheWriteAllowed(const TunePolicy & policy) {
      return !readOnlyMode(policy);
    }

    // The CLOSED SET of NEGATIVE reasons (t3-shell-plan.md section 6.4 item 2).  A file-level
    // reason is a claim about the DEVICE that outlives this process, so it must be comparable by
    // machine and interpretable by the NEXT release that reads it -- which is exactly why the set
    // is declared data rather than "whatever string the deciding code happens to produce".
    //
    // What may be in here: class (a) only, i.e. the device STRUCTURALLY cannot be tuned (there is
    // no injectable slot, or the row lookup is unreachable by construction).  Class (a) does not
    // depend on any gate / tau / soft-gate parameter, so a policy change can never invalidate it.
    //
    // What must NOT be in here: class (b) (every candidate rejected by the validation gate) and
    // class (c) (timeout / IO failure / DEVICE_LOST / no winner).  (b) is only sound to write once
    // the gate parameters are part of policyFingerprint -- which they now ARE (POLICY_CONSTANTS, read
    // by policyFingerprint; R-28 step (1), 2026-09-26) -- because otherwise changing a gate parameter
    // would leave the key permanently skipped -- and (c) is not a property of the device at all
    // (t1-schema.md section 1 discipline (b): "we ran out of time" must never be remembered as
    // "this device cannot be tuned").  Adding a name here is therefore a deliberate act, and both
    // the re-evaluation trigger (section 6.4 item 5) and R17 point at this array.
    //
    // WHY EACH REASON ALSO DECLARES ITS INVALIDATION RULE (2026-09-13 review): a reason alone is
    // not enough.  "This device cannot be tuned" is a claim that OUTLIVES the process, and a claim
    // is only sound if something is guaranteed to void it -- otherwise the file becomes "this key
    // is permanently skipped", which is the worst outcome the negative cache can produce.  So the
    // rule is DATA here (pinned by R17), not a sentence in a comment.
    //
    //   `full-key-set` = the ONLY rule implemented today, and the strongest one available: the
    //   loader refuses the file unless FINGERPRINT, TUNER_VERSION, POLICY, BASELINE and (on the
    //   cache channel) MODEL, BOARD, MAXBATCH and ROWSET_HASH all match, so a change of device,
    //   driver, capability, tuner code, policy, model, board or batch voids the claim.  Every
    //   reason therefore has the SHORTEST possible lifetime, and over-invalidation is the safe
    //   direction (the cost is one re-decision, not a stale skip).
    //
    // A reason that needs anything else is a FORMAT-LEVEL act, not an edit here: a SHORTER
    // lifetime means a new invalidation condition, which means a new key in the header and the
    // file name (R12 forces that decision too); a LONGER lifetime -- surviving a key change -- is
    // not expressible at all, because the file name carries the whole key.  The practical reason
    // the rule is per-reason: the reasons are not obviously equal (no-injection-slot is a hard
    // fact about the device, no-entry leans on the model/family), so "they happen to share a rule
    // today" has to be written down instead of being assumed.
    //
    // WHERE `full-key-set` IS REDEEMED (a declared rule that nothing redeems is a comment in
    // disguise, and this struct carries no runtime weight at all): the LOADER's key comparisons --
    // makeDecision()'s refusal chain (fingerprint, VERSION/TUNER_VERSION, POLICY, BASELINE, MODE,
    // then the outer key item by item, then ROWSET_HASH on the cache channel) together with
    // parseRowsFile()'s required-key loop, which is fed by REQUIRED_ANY_CHANNEL/REQUIRED_FOR_CACHE.
    // Nothing mechanically ties the value to those lines: R17 pins the value AND pins that each
    // named comparison still exists somewhere in this file, and the behavioural half is the
    // smoke's refusal legs (S6, A5/A6, A14/A15/A16, A18, A31/A32).  Change either side and both
    // have to be revisited.
    //
    // ✅ GAP CLOSED (C2, 2026-09-18): VERSION / TUNER_VERSION are now IN REQUIRED_FOR_CACHE, which
    // is the fix this note recommended -- absence is refused on the cache channel.  The comparisons
    // below still keep their `found*` guard, and that is not redundant: they answer "is it the
    // RIGHT version", while
    // REQUIRED_FOR_CACHE answers "did it say at all".  Dropping the guard would make an absent key
    // read as a mismatching one and would name the wrong cause in the refusal line.
    struct NegativeReasonSpec {
      const char* reason;
      const char* invalidation;
    };
    const NegativeReasonSpec NEGATIVE_REASONS[] = {
      { "no-injection-slot", "full-key-set" },
      { "no-entry",          "full-key-set" },
      // A1 (probe batch): `no-gain` -- "the key was swept and nothing beat stock" -- is registered
      // while NOTHING produces it, which is exactly the status `no-entry` above already has, and for
      // the same reason: a name in the closed set is not a name something PRODUCES.  Both halves are
      // needed.  (i) The PARSER must read it -- a hand-written or reflowed file may carry it, and a
      // format that refuses a name it admits is a format that cannot be interpreted.  (ii) Its
      // producer does not exist yet and that is written down rather than implied: the writer is
      // branch (2) of the three-branch decision ("nothing better AND pairsComplete"), which lands
      // with the measurement primitive's decision path.  R17 pins the writer COUNT per reason, so the
      // day that branch assigns this literal the guardrail reddens until its author returns and says
      // what it means -- "reserved" cannot decay into "accidentally live".
      // (iii) REACHABILITY is a separate question from existence: even with the branch in place this
      // write cannot fire today, because `swept-here` is false while the catalogue is a sampling
      // surface.  The first REACHABLE write is therefore step 5 (the first real negative entry).
      // 🔴 GRANULARITY (pinned here so that `full-key-set` is not read as answering it): the negative
      // entry is ONE whole-key claim -- every per-shape pair was measured and no candidate was
      // non-inferior -- and the per-shape readings stay in the report/log.  A reason whose
      // invalidation differs from `full-key-set` would be a FORMAT-level act (a new header key + a
      // loader comparison + an R12 update), not an edit to this row; a per-shape LIFETIME is not
      // expressible at all, because the file name carries the whole key.
      { "no-gain",           "full-key-set" },
      // R-28 step (2), 2026-09-26: `gate-reject` is the FOURTH name, and it is the first one whose
      // producer lands in the SAME batch that declares it.  The reason and the OUTCOME share the
      // spelling deliberately: they are TWO closed sets (this one is the file's reason token, the other
      // is the decision's state) that name ONE meaning -- "the runtime spike-rate gate refused this
      // run's measurement" -- so neither may be read as "there was nothing to measure" (that is the
      // primitive's own NOT_MEASURED, and R-10 keeps the two names apart).
      // WHY IT IS DURABLE AT ALL (the ruling, tune-gaps.md A1-A): a fail-closed gate plus the deleted
      // ceiling (R-1) means a machine whose TRUE spike rate sits near the bound would pay the whole
      // measurement on every start and never land anything (13-26 s per start on the 5080) -- this
      // entry IS the backoff that ends that loop, and it is what the writable outcome above is FOR.
      // SCOPE vs FACT: the fact is device-level (the spike rate follows the device, not the run), while
      // the entry is key-level, so every key pays its own measurement and records its own entry.  That
      // is accepted and written down rather than read as "one entry should have been enough".
      // Its lifetime is `full-key-set` like every other reason here (R17 pins the whole set and the
      // loader comparisons that redeem that value).
      { "gate-reject",       "full-key-set" },
    };

    // The closed set holds on BOTH sides: the parser (so an uninterpretable file is refused
    // instead of loading as if it had said nothing) and the cache writer's "is there anything to
    // write?" predicate below (so free text can never reach the disk).  A set only the reader
    // checks is a set on paper.
    bool negativeReasonKnown(const string & reason) {
      for (const NegativeReasonSpec & r : NEGATIVE_REASONS) {
        if (reason == r.reason) {
          return true;
        }
      }
      return false;
    }

    // W4b: is this a kind this build knows?  The registry and its membership test live HERE, next to
    // NEGATIVE_REASONS and for the same reason: they are the second closed set the file format depends on
    // (the first is the reason), and BOTH the parser (which reads a NEGATIVE line's scope) and the
    // candidate catalogue (which declares the search domain per kind) need them.  Putting the registry
    // here rather than inside the completeness-gate block below is what keeps those two from growing a
    // SECOND kind list -- F17's "the value domain comes from declared data" applies to the closed set of
    // kind names exactly as it does to the axis values.
    struct TuneKindSpec {
      const char* name;
    };
    const TuneKindSpec TUNE_KINDS[] = {
      { "mm" },   // matmul rows: the only kind this tuner can produce so far (FA adds a row)
    };
    const size_t TUNE_KIND_COUNT = sizeof(TUNE_KINDS) / sizeof(TUNE_KINDS[0]);

    // Fail-closed: an unknown kind is never known, so a hand-edited scope cannot name a kind this build
    // would silently accept as "some kind".
    bool tuneKindKnown(const string & kind) {
      for (const TuneKindSpec & k : TUNE_KINDS) {
        if (kind == k.name) {
          return true;
        }
      }
      return false;
    }

    // ── W3-P1: PROVENANCE -- "this candidate set is THIS process's own measurement" ────────────────
    // "This key has no injection slot" is a KEY-SCOPED and STICKY claim: once it is in the cache the
    // key is skipped until the file's invalidation rule voids it.  It may therefore only be concluded
    // from a candidate set this process produced ITSELF -- a set read from a file (the cache)
    // says what SOME OTHER run saw, and a hand-made or dev-reflowed file says nothing at all.  This is
    // the ONE predicate that answers that question, and it is named and printable because the write
    // path has to cite it (R42 pins the citation).
    //
    // Fail-closed by construction: the tested value is the string "measured", and nothing but the
    // enumerator (step 5) ever writes it.  So today the durable negative entry is UNREACHABLE, which
    // is the intended state until that writer exists -- no writer, no verdict.  The older guard
    // (`anyRowInstalled`) stays in place: it answers "did anything install here", which is a
    // different question from "may a durable verdict be minted from this decision".
    bool negativeProvable(const Decision & d) {
      return d.rowSource == "measured";
    }

    // The registered marker of the provenance refusal, as its OWN literal (W3b).  Why a named constant
    // rather than a literal inside the sentence: the acceptance leg (A36) greps exactly this string, the
    // self-test asserts against the same string the writer prints, and R42(e) pins that the two
    // spellings are EQUAL -- all three only work if the marker is one literal in one place.  The refusal
    // sentence below starts with it, so a reader who greps the marker finds the clause that owns it.
    const char* const MEASURED_MARKER = "not this process's own measurement";

    // The EXTRA conditions a table row may name: a CLOSED SET, each entry owning the WHOLE sentence it
    // refuses with.  Why the sentence lives here and not in the writer: the generic state-gate message
    // ("not in a writable state ... only a completed measurement or a structural all-reject may be
    // remembered") is accurate for an outcome that is simply not writable, but it is FALSE for a named
    // condition -- it lists `structural-all-reject` as allowed while refusing exactly that.  A refusal
    // that names the wrong clause sends the next reader to the wrong code.
    //   🔴 W3b: a PREFIX was not enough.  With a prefix, the rest of the sentence lived in the writer,
    // hanging off the `refusal != nullptr` branch -- so the NEXT condition to gain a sentence would have
    // inherited the provenance narrative.  Each entry now returns its own complete sentence, and
    // `refusal == nullptr` stays as the fail-closed default a new entry gets until somebody decides
    // what it should say (the writer then uses the generic sentence, which is why an entry may only be
    // left null when that sentence is TRUE for it).
    struct ConditionSpec {
      const char* name;
      bool (*holds)(const Decision &);
      string (*refusal)(const Decision &);
    };
    const ConditionSpec CONDITION_SPECS[] = {
      { "device-only", [](const Decision & x) { return x.allRejectReasonsAreDevice; },
        [](const Decision & x) {
          // G-D20's own truth: the refusal is about OUR gap, and the sentence this replaces said the
          // outcome "may be remembered" while refusing it.
          return "at least one rejection reason came from OUR implementation gap (our-gap="
               + to_string(x.install.dropped_unsupported_our_gap) + "), and our own gap is not a "
                 "property of this device -- so the coarse outcome 'structural-all-reject' is recorded "
                 "WITHOUT a durable verdict (G-D20)";
        } },
      { "measured-here", [](const Decision & x) { return negativeProvable(x); },
        [](const Decision & x) {
          return string(MEASURED_MARKER) + ": this decision concluded '" + x.outcome
               + "' from a candidate set it did not produce itself (row source '" + x.rowSource
               + "'), and only a candidate set THIS process measured may be remembered as a durable "
                 "verdict -- a file says what some other run saw, and a hand-made or reflowed file says "
                 "nothing at all";
        } },
      { "swept-here", [](const Decision & x) { return x.candidateCoverage == "swept"; },
        [](const Decision & x) {
          // W4b.  The sentence has to name the COVERAGE, because the generic state-gate text is false
          // here in the same way it was false for provenance and for our-gap: the outcome it refuses
          // is the one that sentence says may be remembered.  And the misleading reading is specific
          // and predictable -- "we tried a handful of candidate geometries and none was usable" is
          // NOT "this device has no injectable slot", so the refusal says which of the two happened.
          return "the candidate set is a SAMPLING SURFACE (coverage '" + x.candidateCoverage
               + "'), not a sweep of the declared domain: a sampled all-reject means only that the "
                 "SAMPLED geometries were rejected, so the coarse outcome 'structural-all-reject' is "
                 "recorded WITHOUT a durable verdict (t3-shell-plan.md F17)";
        } },
    };

    // Fail-closed lookup: an unknown name is never satisfied AND has no sentence to quote, so a typo
    // in the table refuses the write with the generic message instead of opening it.
    const ConditionSpec * findCondition(const string & name) {
      for (const ConditionSpec & c : CONDITION_SPECS) {
        if (name == c.name) {
          return &c;
        }
      }
      return nullptr;
    }

    bool conditionHolds(const Decision & d, const string & name) {
      const ConditionSpec * c = findCondition(name);
      return c != nullptr && c->holds(d);
    }

    // A row's `condition` field is a comma-separated LIST of names from that set ("device-only",
    // "device-only,measured-here", ...): one row can then require several extra conditions without
    // growing a column per condition, while every name still comes from the closed set above.
    vector<string> splitConditions(const char * condition) {
      vector<string> out;
      string cur;
      for (const char * p = condition; p != nullptr && *p != '\0'; p++) {
        if (*p == ',') {
          if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        }
        else {
          cur.push_back(*p);
        }
      }
      if (!cur.empty()) { out.push_back(cur); }
      return out;
    }

    // "Is there anything to write?" -- the SECOND half of the write permission.
    // `cacheWriteAllowed(policy)` answers "may this process touch the cache at all" (the user's
    // mode); this answers "does this decision say anything" (the decision's state), and they are
    // separate because they fail for different reasons and are fixed by different people.  The
    // state it rejects is real: a file carrying neither ROW nor NEGATIVE loads without complaint
    // and reads as "this key is already decided" (see the writer comment in formatRowsFileImpl),
    // so the CACHE write path must refuse to create one -- it is the only writer left, and a
    // decision that says nothing may not become "this key is already decided".
    // W2: THE effective row set this process may write down -- ONE spelling, used by BOTH the write
    // gate below and the formatter.  Why one spelling and not two: the gate answers "is there anything
    // to write?" and the formatter emits it, so if they disagreed about a set whose rows were all
    // dropped, the gate would say yes and the file would carry `ROWS 0` with no NEGATIVE -- which the
    // reader treats as "this key is already decided", i.e. tuning suppressed for ever.
    //
    // 🔴 THREE states, and the third one is why this is not a two-line predicate:
    //   Absent       (no install recorded)         -> the DECLARED rows.  The self-test's synthetic
    //                                                 decisions and a set that never reached install:
    //                                                 an ABSENT mask must never silently turn a
    //                                                 declared set into an empty file.
    //   Present      (mask and sizes match)        -> the rows the mask marks installed (decision C).
    //   Inconsistent (mask, sizes DO NOT match)    -> the EMPTY set + loud.  This is an INTERNAL ERROR,
    //                                                 and returning the declared rows here would write
    //                                                 rows this process never installed -- exactly what
    //                                                 section 5 item 4 / ruling D forbid.  The fold takes
    //                                                 the same direction, and the two must not disagree:
    //                                                 a fail-closed fold cancelled by a fail-open writer
    //                                                 would be a fail-open feature with extra steps.
    // The Inconsistent loud fires on each call (the gate and the formatter both ask).  That is accepted
    // on purpose: it is an error that must be impossible, and "silent after the first line" is the
    // failure mode this project keeps refusing.
    vector<vk_tune_row_spec> rowsForWrite(const Decision& d) {
      if (!d.rowInstalledSeen) {
        return d.rows;
      }
      if (d.rowInstalled.size() != d.rows.size()) {
        logLine("WARNING: the durable per-row install mask has " + to_string(d.rowInstalled.size())
                + " byte(s) for " + to_string(d.rows.size()) + " declared row(s): refusing to guess, so "
                "NOTHING is written from this decision (fail-closed -- the mask and the row set must "
                "describe the same set)");
        return vector<vk_tune_row_spec>();
      }
      vector<vk_tune_row_spec> out;
      for (size_t i = 0; i < d.rows.size(); i++) {
        if (d.rowInstalled[i] != 0) {
          out.push_back(d.rows[i]);
        }
      }
      return out;
    }

    // The effect column of this decision's outcome row, by name (`nullptr` for an outcome this build
    // does not know -- the same fail-closed shape as `outcomeWritable`).  FORWARD DECLARED because this
    // gate is its first reader; the definition sits with the table, so the column has exactly ONE
    // reader and cannot drift between the gate, the writer and the self-test.
    const char * declaredEffectFor(const Decision & d);

    bool cacheEntryWritable(const Decision & d) {
      // W2: the gate now asks about the rows this process may actually WRITE (the durable per-row set
      // when an install happened, the declared set otherwise) -- the SAME helper the formatter uses, so
      // "there is something to write" and "here is what gets written" cannot disagree.
      if (!rowsForWrite(d).empty()) {
        return true;
      }
      // 🔴 ONE CACHE FORM (2026-09-30 ruling; option D landed): an EMPTY ROW SET is a durable
      // artifact in its own right -- "this key was measured, and the stock configuration is the best
      // this device has" -- so an outcome whose row declares the effect `rows` may write `ROWS 0`.
      // The TABLE decides, not this function: a `negative:` row still needs its marker, and a
      // `nothing:` row is not admitted by this arm.  The completeness gate asks THIS predicate
      // (firstUnconcludedKind), so "may write" and "is complete" cannot disagree about an empty set.
      const char * effect = declaredEffectFor(d);
      if (effect != nullptr && string(effect) == "rows") {
        return true;
      }
      return d.negative && negativeReasonKnown(d.negativeReason);
    }

    // The CLOSED SET of decision outcomes and whether each one may be REMEMBERED (step "3-4").
    //
    // Why a table and not a bool: the rule is "a measurement-level failure is never written", and
    // the failure modes are not interchangeable -- a timeout, an IO error, a lost device and a run
    // with no winner all mean "this attempt did not produce an answer", while a structural
    // all-reject means "there is no answer to produce" and a completed measurement means "here is
    // the answer".  Only those last two may be written, and they are different KINDS of writable
    // (a row set vs a NEGATIVE entry), which is why the class is declared as well -- the class is
    // what section 6.4's rule table names, and what step 4/5 will branch on.
    //
    // Class (b) -- a run the RUNTIME GATE refused -- was declared NOT writable until the gate / tau /
    // soft-gate parameters entered policyFingerprint, because changing a gate parameter would
    // otherwise leave the key permanently skipped (section 6.4 items 1/5 -- the same argument R17 makes
    // for the file's reason).  🔴 ONE MEANING FOR THE DEPLOYED NAME: this comment used to say "every
    // candidate rejected by the validation gate", which is a DIFFERENT claim; the name's carrier is
    // settled by the ruling row in tune-gaps.md A1-A (the P12 runtime spike-rate gate), and
    // `rejectBySpikeGate` below is the only thing that mints it.
    // THAT CONDITION IS NOW MET: the gate/statistics constants are rows of POLICY_CONSTANTS and
    // policyFingerprint() appends them, so a gate change changes the key by construction instead of by
    // trusting that the revision moved (R-28 step (1), 2026-09-26).  The writable flag is TRUE and the
    // same-spelling NEGATIVE reason is in NEGATIVE_REASONS in THIS batch (R-28 step (2)) -- the two are
    // indivisible: a same-spelling reason whose outcome row stayed unwritable would be a name the cache
    // writer can never emit, and a writable row with nothing that produces it would be a state with no
    // producer.
    //
    // The lookup is FAIL-CLOSED (see outcomeWritable): an outcome this build does not know is not
    // writable, so a caller that invents a state cannot write by accident.
    struct DecisionOutcomeSpec {
      const char* outcome;
      // `klass` has exactly three roles, and NONE of them is the runtime verdict:
      //   (1) R19's whitelist key in the guardrail (t3-guardrails.py WRITABLE_CLASSES);
      //   (2) the self-test's FIXTURE SELECTOR (:1084/:1092 choose which shape to build);
      //   (3) a documentation label on the outcome name.
      // The production decision path never reads it: `outcomeWritable` uses `writable` + `condition`
      // only.  A rename is therefore LOUD twice over, never silent -- R19 reddens (whitelist), and
      // the self-test reddens too (a class with no fixture builder makes `outcomeWritable` disagree
      // with `o.writable`), which is what makes "add a writable class, forget its fixture"
      // impossible to land green.
      const char* klass;       // "structural" / "completed" / "gate" / "measurement"
      bool writable;
      // The EXTRA conditions on top of `writable`, as a comma-separated LIST from the closed set in
      // `CONDITION_SPECS`: "" = none, "device-only" = every rejection reason must be a device property
      // (G-D20), "measured-here" = the candidate set must be this process's own measurement (W3-P1),
      // "swept-here" = it must cover the declared domain rather than a sampling surface (W4b).
      // They live in the table on purpose -- a condition expressed as a branch on `klass`, or as
      // a clause inside one of the gates, would let a rename or a refactor silently change what may be
      // written, which is the "implicit knowledge in a branch" the first design rule bans.  The
      // writer reports the FIRST named condition that does not hold, so a refusal names its clause.
      const char* condition;
      // 🆕 (7d): THE DURABLE EFFECT, as DECLARED data.  Section 4.5 of the plan requires outcome =>
      // "what lands on disk" to be a TOTAL FUNCTION, and this column is that function:
      //   `rows`              -- the durable artifact is the row set;
      //   `negative:<reason>` -- it is a NEGATIVE entry and the reason is named HERE, so a fixture can
      //                         be driven BY THIS STRING instead of hand-assembling the state (the way
      //                         a fixture ends up asserting the wrong thing);
      //   `nothing:<why>`     -- this attempt leaves no durable claim, and the why is the plan table own
      //                         wording (`in-progress` / `transient` / `rerun`).
      // 🔴 THE RULE THAT MAKES IT A CRITERION: every `structural` and `completed` row must NOT be
      // `nothing:` -- a finished run that remembers nothing is the "pay for the measurement on every
      // startup" shape this project exists to prevent.  R19 pins the closed set, that rule, and the
      // fact that the self-test READS the column.
      const char* effect;
    };
    const DecisionOutcomeSpec DECISION_OUTCOMES[] = {
      // A structural all-reject is a KIND-LEVEL, STICKY claim ("this key has no injectable slot"), so
      // it needs all three: the reasons must be about the DEVICE (device-only), the candidate set must
      // be OURS (measured-here), and it must be a SWEEP rather than a sample (swept-here).  The last
      // two are ORTHOGONAL to the first -- who produced the set, and how completely it was searched,
      // are different questions.
      { "structural-all-reject", "structural",  true,  "device-only,measured-here,swept-here", "negative:no-injection-slot" },
      // A completed measurement writes ROWS, not a verdict about what does not exist.  The W4b coverage
      // condition is named here TOO, deliberately: the question is not about the outcome name but about
      // the candidate set, and leaving it off this row would mean an unmarked run could write rows as if
      // the set had been swept -- the same silent drift, one row over.  Today this row cannot fire
      // either (nothing installs a row yet); it is what keeps the rule attached to the SET rather than
      // to one outcome's spelling.
      { "measured",              "completed",   true,  "measured-here,swept-here",            "rows" },
      // 🆕 (7d) BRANCH (2): THE COMPLETED MEASUREMENT THAT FOUND NO WINNER.  It is declared here with
      // its PRODUCER in the same batch (section 2.3: an OUTCOME name may never precede the code that can
      // set it) and its class is `completed` -- a measurement that finished.
      // 🔴 ITS DURABLE EFFECT IS `rows`, AND THE ROW SET IS EMPTY (2026-09-30 ruling: ONE cache form,
      // "the complete set of this machine's best parameters").  "No candidate is not-worse than stock" IS
      // an answer -- it says "use the stock configuration everywhere" -- and `ROWS 0` plus its
      // ROWSET_HASH is how that answer is written; the durable NEGATIVE entry it used to write was the
      // SECOND cache form the ruling deletes.  That is exactly what the old class-only fixture selector
      // could not see (it would have exercised this row with a rows fixture), which is why the effect
      // column and the fixture dispatch land together.
      // Its conditions are the ones a WHOLE-KEY negative claim needs: this process measured the set
      // (measured-here) and the set is a sweep rather than a sample (swept-here) -- the same two the
      // `measured` row carries, because the coverage question belongs to the SET, not to the outcome
      // spelling.  The PRUNE condition that used to be the third is gone with the layer (option D).
      { "measured-no-gain",      "completed",   true,  "measured-here,swept-here",            "rows" },
      // R-28 step (2): THE ONE ROW WHOSE FLAG IS TRUE AND WHOSE CLASS IS `gate`.  It carries
      // `measured-here` for the same reason the rows above do (W3-P1): what a gate refusal remembers --
      // "this device's block times spike too often for the comparison to be trusted" -- is a claim
      // about the DEVICE, so only a candidate set THIS process measured may mint it.  A cache file
      // channel cannot even supply the inputs (a file carries no spike count), and the provenance
      // condition is what makes that structural instead of hopeful.
      // Its durable effect is a NEGATIVE entry and NEVER a row set: `rejectBySpikeGate` clears the rows
      // for exactly that reason.  Note the class token is unchanged -- the class is the guardrail's
      // whitelist key and the self-test's fixture selector, not the runtime verdict.
      { "gate-reject",           "gate",        true,  "measured-here",                                          "negative:gate-reject" },
      { "running",               "measurement", false, "",                                                       "nothing:in-progress" },
      { "timeout",               "measurement", false, "",                                                       "nothing:transient" },
      { "io-error",              "measurement", false, "",                                                       "nothing:transient" },
      { "device-lost",           "measurement", false, "",                                                       "nothing:transient" },
      { "no-winner",             "measurement", false, "",                                                       "nothing:rerun" },
      // A1 (probe batch): `unstable` means "we do NOT know" -- the re-measurement did not re-confirm
      // the decision inside the stability boundary (STABILITY_BOUNDARY_SE, tunestats.h) -- which is a
      // THIRD thing next to "no answer was produced" (timeout / io-error / device-lost / no-winner)
      // and "there is no answer to produce" (structural-all-reject).  It is NOT writable, and the
      // reason is the rule this table exists for: writing "unresolved" into the file would state,
      // permanently, a fact about the DEVICE that the measurement never established.  Declaring it as
      // a row (rather than leaving it in a comment where it is decided) is what makes the state gate
      // refuse it BY NAME -- and the self-test iterates this declared table, so a `measurement` row
      // is exercised in both directions at birth without a new fixture.
      // 🔴 NOTHING PRODUCES IT TODAY, and that is stated rather than left to the reader: the producer is
      // the stability step of the decision path, which lands with the measurement primitive in this
      // batch, and until it does this row is a WINDOW ITEM -- tracked as a ledger row in tune-gaps.md
      // section A1-A (the list of names the spec declares before their carrier exists).  An OUTCOME name
      // may not precede its producer, unlike a NEGATIVE reason whose reserved class above exists
      // precisely because the parser must accept names no writer has produced yet; registering these
      // two in the SAME batch as their producers is what keeps the row from being reverse drift, and the
      // ledger is what makes that an obligation someone can check instead of prose.
      { "unstable",              "measurement", false, "",                                                       "nothing:rerun" },
    };

    // ── P12: THE RUNTIME SPIKE-RATE GATE (R-28 step (2), 2026-09-26) ──────────────────────────────
    //
    // WHERE THE JUDGEMENT LIVES (R-10): the primitive MEASURES and REPORTS -- its result carries
    // `spikes` / `blocks` / `objects` and says nothing about any gate -- and the COMPARISON happens
    // HERE, on the shell side.  That is why the outcome below is `gate-reject` while the primitive's own
    // `VK_TUNE_PROBE_NOT_MEASURED` keeps its single meaning ("I could not produce a usable
    // comparison").  Merging the two names would let "this measurement is too spiky to trust" be read
    // as "there was nothing to measure", which is the opposite claim.
    //
    // FAIL-CLOSED, AND IN WHICH SENSE (the correction R-4(ii) forces into the open): the DECISION reads
    // the interval's UPPER bound (`high`) -- never the point estimate, never the lower bound -- because
    // "we cannot rule out a rate above the measured support" is what refuses a write.  The CONSTRUCTION
    // is NOT conservative in the same direction, and the loud says so: `rateInterval` is the i.i.d.
    // Wilson interval and spikes are CLUSTERED rather than drifting (tunestats.h states that assumption
    // next to the estimator), so the interval is if anything too NARROW and the bound therefore too EASY
    // to pass.  Direction fail-closed / construction not -- the two halves must never be read as one.
    //
    // The instrument is the EXISTING one (`TuneStats::rateInterval`, the same interval family the
    // decisions use, with the declared `ALPHA`) because R-9(i) forbids inventing a metric here;
    // R-4(ii)'s clustering-aware interval is registered, not implemented.
    struct SpikeGateVerdict {
      bool reject = false;                  // true => this measurement may not become a durable row set
      TuneStats::RateInterval interval {};  // the i.i.d. interval the decision actually read
      uint32_t spikes = 0;                  // the counting inputs, carried so the loud can NAME them
      uint32_t blocks = 0;
      uint32_t nObjects = 0;                // not an input to the interval: it is the one ingredient a
                                            // clustering-aware interval will need, so it is printed
                                            // (a reading, not a criterion -- the gap stays visible)
    };

    // The judgement, and ONLY the judgement: no logging, no Decision, no file.  `blocks == 0` leaves
    // `interval.ok` false ("an empty denominator is not a rate of zero"), and that is a REJECT rather
    // than a pass: a caller that asks before it owns a denominator gets the fail-closed answer instead
    // of "0 of 0 spikes, how calm".
    SpikeGateVerdict judgeSpikeRate(uint32_t spikes, uint32_t blocks, uint32_t nObjects) {
      SpikeGateVerdict v;
      v.spikes = spikes;
      v.blocks = blocks;
      v.nObjects = nObjects;
      v.interval = TuneStats::rateInterval((long) spikes, (long) blocks, TuneStats::ALPHA);
      v.reject = !v.interval.ok || v.interval.high > TuneStats::SPIKE_RATE_GATE;
      return v;
    }

    // Turn a rejected measurement into the durable verdict.  THE one production writer of the
    // `gate-reject` state and of its same-spelling reason: R17 counts this assignment site, and R42
    // counts the marker it sets alongside `finalizeOutcome`'s.
    //
    // Four things this function owns, in the order they matter:
    //   (1) IT CLEARS THE ROW SET.  A gate refusal says "this run's measurement quality cannot support
    //       ANY durable claim", so rows measured in the same run must not survive it; the verdict is
    //       about the measurement, not about which candidate happened to look fast.
    //   (2) IT SETS THE MARKER WITH ITS OWN ORIGIN.  `negativeOrigin` records WHO may take a marker away,
    //       and the finalizer recomputes only its own (AllReject).  Leaving this one at `None` would be
    //       the "one field, two writers, no record of who wrote it" sharing that field exists to
    //       prevent, and would let the finalizer's reset wipe a verdict it never reached.
    //   (3) IT SETS THE OUTCOME, so the state gate and the file agree by construction rather than by the
    //       caller remembering to set two fields that must match.
    //   (4) IT SAYS THE READING OUT LOUD -- interval AND bound AND the counting inputs.  R-28 requires
    //       the loud to name the measured interval and the bound; `nObjects` rides along because it is
    //       the missing ingredient of the registered clustering-aware interval.
    //
    // NOT SET HERE, deliberately: `allRejectReasonsAreDevice` (the row's condition is `measured-here`;
    // inventing a second satisfied condition would make the fixture pass for the wrong reason) and
    // `rowInstalledSeen`/`rowInstalled` (the gate judges the measurement, which is upstream of every
    // install, so there is no mask to reconcile -- and if a caller ever ran this after an install,
    // `rowsForWrite`'s size mismatch is the fail-closed, LOUD answer rather than a silent guess).
    //
    // ⚠️ WHAT THIS FUNCTION DOES NOT SETTLE, split by which half has a CARRIER (R-28 step (3) tightened
    // this note, because a half that can be executed must not be left as a sentence):
    //   * the order against the THREE-BRANCH decision is FIXED by the ruling -- the gate judges the
    //     MEASUREMENT, which is upstream of the install accounting, so it runs FIRST and a gate-rejected
    //     run never reaches the three branches.
    //   * the order against the FINALIZER on an EMPTY-INSTALL decision is now EXECUTED by the self-test's
    //     (k) case: the finalizer must return Nothing and leave this verdict and its marker alone.
    //   * what still has NO carrier is the protocol side (⑤): a row that installs AFTER a gate rejection.
    //     That half stays a registered window obligation, not an implied guarantee.
    // 🔴 THE MARKER AND THE LOUD ARE TWO RESPONSIBILITIES (split 2026-09-26, R-29 close-out).
    // The marker is the STATE a gate rejection leaves; the NOTICE is the EVENT it reports.  They were
    // one function, and the self-test's POLICY-row probe -- which needs a decision the FORMATTER can
    // write and therefore installs this marker -- ended up reporting a gate event for a measurement it
    // never made: `NOTICE: : spike-rate gate: ...` with an EMPTY device name (the synthetic decision has
    // none), a third copy of the sentence the smoke pins at exactly two, and the last red leg of the
    // close-out run.  The literal stays in ONE production site (this setter), which is what the census
    // counts (`WANT_REASON_WRITERS`), so splitting costs no rule anything.
    void setGateRejectMarker(Decision & d) {
      d.rows.clear();
      d.negative = true;
      d.negativeReason = "gate-reject";
      d.negativeScope.clear();
      d.negativeScope.push_back("mm");
      d.negativeOrigin = Decision::NegativeOrigin::SpikeGate;
      d.outcome = "gate-reject";
    }
    void rejectBySpikeGate(Decision & d, const SpikeGateVerdict & v) {
      setGateRejectMarker(d);
      logLine("NOTICE: " + d.deviceName + ": spike-rate gate: the measured i.i.d. spike interval ["
              + to_string(v.interval.low) + ", " + to_string(v.interval.high)
              + "] has its UPPER bound above the declared bound "
              + to_string(TuneStats::SPIKE_RATE_GATE) + " (spikes=" + to_string(v.spikes)
              + " blocks=" + to_string(v.blocks) + " nObjects=" + to_string(v.nObjects)
              + "), so NO row set is written for this key; the measurement's quality is a durable "
                "property of this device+policy, so a NEGATIVE gate-reject entry is written instead -- "
                "that entry IS the backoff (the next run reads it and does not pay for the measurement "
                "again).  The interval assumes INDEPENDENT blocks and spikes are clustered, so it is if "
                "anything too NARROW: the DIRECTION is fail-closed (the upper bound decides), the "
                "CONSTRUCTION is not conservative");
    }

    // ── (7c) THE JUDGEMENT LAYER ────────────────────────────────────────────────────────────────────
    // WHAT IT IS: the step that turns the measurement's paired differences into the two answers the
    // decision needs -- "is this comparison trustworthy at all" and "which candidate is not worse than
    // stock on EVERY covered shape" (tune-measurement.md section 2.3.1's live criteria).  It DECIDES
    // NOTHING about what is written: the three branches and the write are (7d)'s, which is why this
    // function neither calls the write path nor assigns an outcome -- the spike gate's marker is minted
    // by `rejectBySpikeGate`, the ONE setter that owns it.
    //
    // WHERE EVERY CRITERION COMES FROM (no new statistics and no second spelling of any of them):
    //   * the interval family IS `TuneStats::pairedSignedRank` (the ONE interval: median +/- t * 1.2533 *
    //     MAD / sqrt(n)).  The shell must not assemble that formula: its two constants appear only inside
    //     tunestats.cpp, and R48 pins that (a literal here would be a second answer to "what is an
    //     interval");
    //   * non-inferiority is `TuneStats::notWorseThan(r, TOLERANCE)`, and the SHAPE dimension is a
    //     CONJUNCTION (intersection-union) -- so "the winner" is a candidate that passes on EVERY covered
    //     shape, and no multiplicity correction is applied on that dimension;
    //   * the sentinel is `TuneStats::equivalentWithin(r, TOLERANCE)`: the whole interval INSIDE
    //     +-tolerance.  "The sentinel did not win" was DELETED as a criterion (failing to reject equality
    //     is not evidence of equality), and the sentinel measures NOISE, not bias -- its job is to say
    //     whether this comparison is comparable at all;
    //   * the stability boundary is `TuneStats::decisionUnstable(r, TOLERANCE, ALPHA)`, and by the ruling
    //     it applies ONLY to the claim that would be written -- never to the gate or the sentinel, whose
    //     flip costs machine time rather than a wrong durable claim;
    //   * the gate is `judgeSpikeRate` + `rejectBySpikeGate` (landed with R-28), now fed THIS run's own
    //     counts instead of a fixture's.
    //
    // THE RULING'S PRIORITY, and it is visible in the shape of this function: the gate judges the
    // MEASUREMENT and therefore runs FIRST (a gate-rejected run never reaches the branches); an incomplete
    // pair set is the branches' business (`pairsComplete`), and the boundary is not consulted for a pair
    // whose interval does not exist (`ciMissing` is counted apart from `winnerUnstable`, because "no
    // data" and "too close to the line" are different answers -- the ruling says so in as many words).
    void judgeMeasurement(Decision & d) {
      Decision::Judgement & j = d.judgement;
      j.attempted = true;
      const Decision::Measurement & m = d.measurement;
      // (a) THE GATE FIRST.
      const SpikeGateVerdict gate = judgeSpikeRate(m.spikes, m.blocks, m.objects);
      j.gateSpikes = gate.spikes;
      j.gateBlocks = gate.blocks;
      j.gateObjects = gate.nObjects;
      j.gateLow = gate.interval.low;
      j.gateHigh = gate.interval.high;
      j.gateRejected = gate.reject;
      if (gate.reject) {
        rejectBySpikeGate(d, gate);
        return;
      }
      // (7e-1) THE NORMALISER EXISTS NOW, so nothing is parked here any more.  The ruling was option (i):
      // the primitive writes `d_out` block-relative (that shape own anchor median as the divisor, with the
      // `> 0` and `>= 2` measured-block guard), which is the unit `TOLERANCE` lives in -- so the interval verdicts below run
      // unconditionally and `units_missing` is gone from this layer line.  The third path (derive the scale
      // from the measured differences) was REJECTED and is registered, because it degenerates exactly on the
      // two cases this layer judges: `median(d)` is ~0 for the A/A sentinel (the divisor blows up) and
      // NEGATIVE for a slower candidate (the sign of the criterion flips).
      // (b) THE INTERVALS.  The samples are the primitive's dense layout with NaN for "never measured",
      //     and the differences are ALREADY paired (t_stock - t_object inside one rep).  Object 0 is the
      //     anchor (NaN by design), object 1 is the A/A sentinel, and candidate `c` on covered shape `s`
      //     sits at `2 + c*nShapes + s` -- the mapping the measurement built, read back here.
      const uint32_t nShapes = (uint32_t) m.shapes.size();
      const uint32_t nCands = (uint32_t) d.candidateGeoms.size();
      if (nShapes == 0 || m.objects < 2 || m.reps == 0 || m.samples.empty()) {
        return;   // nothing to judge: no pair is counted and `winner` stays -1 (the branches answer)
      }
      j.verdictsComputed = true;
      // 🔴 R51 / (A): ALL TIES ARE A BOUNDED INTERVAL, NOT "NO DATA" -- AND NOT "EQUAL".  When every
      // paired difference is EXACTLY zero the signed-rank interval cannot be formed, but that is not
      // "nothing was measured": it is "the difference is below this run's ruler", and the ruler is the
      // achieved block's microsecond quantisation (1/hot_us).  B-0 measured that floor at 1/1202 = 0.083%
      // against a 1% tolerance, so the honest statement is the BOUND [0, resolution], which the tolerance
      // comparison then judges: 0.083% < 1% => "equivalent within tolerance" -- positive, bounded, and it
      // stops the A/A sentinel from being unprovable by construction.
      // THE FAIL-CLOSED HALF IS THE COMPARISON: if the floor were COARSER than the tolerance the rule must
      // not fire (a zero difference would not bound anything the criterion asks about), which is why the
      // resolution is compared here instead of assumed.  A pair that fails that test stays `ciMissing`,
      // which is what branch (2) refuses to read as "nothing won".
      const double resolutionRel = m.hotUs > 0.0 ? (1.0 / m.hotUs) : 0.0;   // the heaviest shape's, kept
                                                                              // as the fallback reading
      // (7e-3) THE FLOOR AS A READING: how many covered shapes have a block below the declared minimum,
      // and which ruler is the lightest one seen.  A silent gate would leave "the ruler was too coarse"
      // invisible, which is exactly how two causes came to share one counter.
      for (size_t sIdx = 0; sIdx < m.shapes.size(); sIdx++) {
        const double aUs = sIdx < m.anchorUs.size() ? m.anchorUs[sIdx] : 0.0;
        if (!(aUs >= TuneStats::MIN_BLOCK_TIME_MS * 1000.0)) {
          j.shapesBelowFloor++;
        }
        if (aUs > 0.0 && (j.minAnchorUs == 0.0 || aUs < j.minAnchorUs)) {
          j.minAnchorUs = aUs;
        }
      }
      const auto intervalFor = [&](uint32_t objectIndex, uint32_t shapeIndex, TuneStats::PairedResult & out) {
        // (7e-3) THE SHAPE'S OWN RULER, and the floor that decides whether it supports a claim at all.  A
        // shape whose block is below `MIN_BLOCK_TIME_MS` cannot bound anything the tolerance asks about
        // (its quantum is comparable to the tolerance), so the bounded rule below must not fire for it --
        // and WHEN it does not fire, this is what says which of the two causes it was.
        const double shapeAnchorUs = shapeIndex < m.anchorUs.size() ? m.anchorUs[shapeIndex] : 0.0;
        const double shapeRes = shapeAnchorUs > 0.0 ? (1.0 / shapeAnchorUs) : 0.0;
        // ⚠️ UNITS (measured 2026-09-29, from the raw list): `MIN_BLOCK_TIME_MS` is
        // MILLISECONDS (2.0) and every anchor median here is MICROSECONDS, so the first form of this
        // comparison was true for any real block -- the gate could not fire, and the same loop printed
        // `shapes_below_floor=0` beside `min_anchor_us=1747`.  The declared constant is the source of
        // the threshold; this line is the conversion, and `ci_coarse` is the reading that proves it.
        const bool shapeRulerOk = shapeAnchorUs >= TuneStats::MIN_BLOCK_TIME_MS * 1000.0;
        std::vector<double> series;
        uint32_t finite = 0;
        double maxAbs = 0.0;
        const size_t base = ((size_t) shapeIndex * (size_t) m.objects + (size_t) objectIndex) * (size_t) m.reps;
        for (uint32_t r = 0; r < m.reps; r++) {
          const double v = m.samples[base + (size_t) r];
          if (std::isfinite(v)) {
            finite++;
            series.push_back(v);
            const double a = v < 0.0 ? -v : v;
            if (a > maxAbs) {
              maxAbs = a;
            }
          }
        }
        // (R52) WHAT THIS PAIR CONTRIBUTED, counted on the READ side: a pair with samples is one the
        // judgement could actually judge, and a pair with none is recorded WITH its address -- because
        // "the pairing is wrong" is only actionable if the log names the slot it went wrong at.
        if (finite >= 2) {
          j.pairsMeasured++;
        }
        else if (j.firstMissingShape < 0) {
          j.firstMissingShape = (int) shapeIndex;
          j.firstMissingObject = (int) objectIndex;
        }
        out = TuneStats::pairedSignedRank(series, TuneStats::ALPHA);
        // 🔴 THE BOUND COMES FROM THE DATA, not from a ruler constant -- and it is measured, twice over.
        // FIRST FORM (all differences exactly zero) never fired: the real series is mostly exact zeros
        // with a few +-1 us entries (`ties_bounded=0` while 14 pairs had no interval).  SECOND FORM (every
        // difference inside 1/hot_us) also missed, because each SHAPE has its own quantum (the primitive
        // divides by that shape own anchor median) and the shell only knows the heaviest shape's.
        // So the bound is the observed band itself: if the whole series lies within +-maxAbs and maxAbs is
        // still inside the tolerance, then "the difference is at most maxAbs" is a positive, bounded claim
        // -- which the tolerance comparison then judges -- and it is at least `finite >= PAIRS_FLOOR`
        // observations wide.  Anything wider than the tolerance is NOT bounded this way: it stays
        // `ciMissing` (no data), which is the fail-closed half branch (2) depends on.
        if (!out.ciOk && finite >= (uint32_t) TuneStats::PAIRS_FLOOR && maxAbs > 0.0
            && maxAbs <= TuneStats::TOLERANCE && shapeRulerOk && shapeRes > 0.0
            && shapeRes <= TuneStats::TOLERANCE) {
          out.hl = 0.0;
          out.ciLow = -maxAbs;
          out.ciHigh = maxAbs;
          out.ciOk = true;
          out.pairs = (long) finite;
          j.tiesBounded++;
        }
        else if (!out.ciOk) {
          // THE DECOMPOSITION (7e-3), and it is a PARTITION BY CONSTRUCTION: every pair that ends up with
          // no interval lands in exactly one bucket, so the parts add up to `ciMissing` and the next
          // question becomes askable.  The first form required `finite >= 2` and therefore answered zero for
          // every bucket while `ci_missing` read 14 -- MEASURED, and the reason `ci_never` exists: those
          // pairs had NO finite samples at all (the primitive never wrote their slot), which is a different
          // question from any ruler.
          if (finite < 2) {
            j.ciNever++;
          }
          else if (!shapeRulerOk || shapeRes <= 0.0 || shapeRes > TuneStats::TOLERANCE) {
            j.ciMissingCoarse++;
          }
          else {
            j.ciMissingWide++;
          }
        }
      };
      // (c) THE SENTINEL, one interval per covered shape, and EVERY one of them has to pass: "this
      //     comparison is comparable" is a statement about the whole run, not about its best shape.
      bool sentinelOk = true;
      for (uint32_t s = 0; s < nShapes; s++) {
        TuneStats::PairedResult r {};
        intervalFor(1u, s, r);
        j.sentinelPairs++;
        if (!r.ciOk || !TuneStats::equivalentWithin(r, TuneStats::TOLERANCE)) {
          sentinelOk = false;
        }
      }
      j.sentinelEquivalent = sentinelOk;
      // (d) THE CANDIDATES, shape by shape; the first candidate that passes on EVERY covered shape wins.
      bool winnerFound = false;
      for (uint32_t c = 0; c < nCands; c++) {
        bool allShapes = true;
        bool anyUnstable = false;
        bool haveMargin = false;
        double worstMargin = 0.0;
        for (uint32_t s = 0; s < nShapes; s++) {
          TuneStats::PairedResult r {};
          // 🔴 THE IDENTITY THE MEASUREMENT ACTUALLY BUILT (measured 2026-09-29).  Rows -- and therefore
          // objects 2.. -- are SHAPE-MAJOR: the measurement walks the covered shapes and, for each, its
          // candidates, so candidate c on shape s is object 2 + s*nCands + c.  The judgement read it
          // as 2 + c*nShapes + s, which is the same set of indices in a different order, so the counts
          // looked fine and the PAIRINGS were wrong: 14 of 18 (shape, candidate) combinations pointed at
          // objects that belong to another shape, and the primitive had never measured them there -- which
          // is exactly what pairs_illegal=90 says (6 shapes x 15 objects that are not theirs) and what
          // ci_never=14 counted.  A bijection is not enough: the pairing is the measurement.
          intervalFor(2u + s * nCands + c, s, r);
          j.candidatePairs++;
          if (!r.ciOk) {
            // NO INTERVAL IS NOT "UNSTABLE": it is missing data, and the branches answer for it.
            j.ciMissing++;
            allShapes = false;
            continue;
          }
          if (TuneStats::notWorseThan(r, TuneStats::TOLERANCE)) {
            j.notWorsePairs++;
            const double margin = r.ciLow + TuneStats::TOLERANCE;
            if (!haveMargin || margin < worstMargin) {
              worstMargin = margin;
              haveMargin = true;
            }
          }
          else {
            allShapes = false;
          }
          if (TuneStats::decisionUnstable(r, TuneStats::TOLERANCE, TuneStats::ALPHA)) {
            anyUnstable = true;
          }
        }
        if (allShapes && !winnerFound) {
          winnerFound = true;
          j.winner = (int) c;
          j.winnerUnstable = anyUnstable;
          j.winnerMargin = worstMargin;
        }
      }
    }

    // ── R-28 step (3): THE CANDIDATE-DROP ACCOUNTING AND THE THIRD BRANCH ─────────────────────────
    //
    // WHAT A DROP IS, and why only TWO of the three buckets are in the table (tune-measurement.md
    // section 2.6(a), restated in t3-shell-plan.md:442): a DETERMINISTIC rejection -- `illegal` (the rule
    // table says this geometry cannot exist) or `unsupported` (this device/family cannot build it) --
    // never entered the denominator at all, because the candidate was never going to be compared; an
    // ACCIDENTAL or budget-shaped one (`build-failed`) did enter it, so it LEAVES the judgeable set and
    // has to be counted.  The name below is therefore the GATING set, and its count is what the
    // suppression loud prints.
    //   🔴 `illegal` IS NOT A MEMBER, and that is the rule rather than an omission: folding it in would
    //   make "the denominator shrank because our own rule table refused everything" look like "we measured
    //   what we set out to measure" -- the false green section 2.6(a) warns about.  Its count lives on the
    //   probe side (`vk_tune_probe_result::pairs_illegal`), whose own log line is registered as R-11's
    //   window item and is deliberately NOT discharged (or pre-empted) by this batch.
    //
    // The table PAIRS each name with the Decision field that counts it, and the loud is BUILT from the
    // table: a new drop reason therefore cannot be added without appearing in the line a reader greps,
    // and the line cannot name a reason whose count nobody keeps.
    struct CandidateDropSpec {
      const char* name;
      uint32_t Decision::*count;
    };
    const CandidateDropSpec CANDIDATE_DROPS[] = {
      { "build-failed", &Decision::droppedBuildFailed },
    };

    // The drop list EXACTLY as the suppression loud carries it, built from the table above so the line
    // and the declared set cannot drift apart.
    string formatCandidateDrops(const Decision & d) {
      string out;
      for (const CandidateDropSpec & c : CANDIDATE_DROPS) {
        out += (out.empty() ? "" : ", ") + string(c.name) + "=" + to_string(d.*(c.count));
      }
      return out;
    }

    // ONE place the measurement's accounting enters the Decision (R-12: a new field needs a producer and
    // a consumer in the same batch).  `enumeratedCandidates` is a PARAMETER rather than a field of the
    // result, because the result describes what was MEASURED: how many candidates the caller's enumerator
    // produced for this key is a fact about the caller's set, and only the caller knows it.  The two
    // definitions are written HERE, with the first producer, rather than left to the reader -- the plan
    // fixes the fields' SHAPE (`cands=<measured>/<enumerated>`), not their sources.
    // `r.objects` is deliberately NOT the numerator.  It is the number of OBJECTS the primitive timed,
    // and that array carries arms which are not candidates at all (the stock anchor and the A/A
    // sentinel -- see the measurement's own comment).  Feeding it in would print a numerator ABOVE its
    // denominator, which section 2.6 calls the wiring bug this pair of fields exists to expose.  So the
    // caller states how many of the ENUMERATED candidates were really measured, and the two numbers are
    // then ONE quantity ("of N enumerated candidates, M were measured") instead of two units side by
    // side.
    void noteMeasurementAccounting(Decision & d, const vk_tune_probe_result & r,
                                   uint32_t enumeratedCandidates, uint32_t measuredCandidates) {
      d.pairsComplete = r.pairs_complete;
      d.droppedBuildFailed = r.dropped_build_failed;
      d.candidatesMeasured = measuredCandidates;
      d.candidatesEnumerated = enumeratedCandidates;
      d.pairsLegal = r.pairs_legal;
      d.shapesCount = r.shapes;
    }

    // THE THIRD BRANCH (tune-measurement.md section 2.6(a)): "nothing was better" AND the pair set is
    // incomplete.  It is ONE decision point rather than a sentence scattered through the branches, and
    // that is the requirement -- the clause that PICKS THE OUTCOME is the clause that reads
    // `pairsComplete`, so the flag cannot be consulted somewhere else and ignored here.
    //
    //   * "nothing was better" is read as `rowsForWrite(d).empty()`, i.e. through the SAME helper the
    //     write gate and the formatter use, so "no winner" cannot mean one thing here and another there.
    //   * BRANCH ① IS NOT AFFECTED BY INCOMPLETENESS: a candidate that won may still be written, because
    //     what was dropped was an ALTERNATIVE (section 2.6(a) states this explicitly -- reading the flag
    //     as "suppress the whole write" would let one slow candidate void a safe row set).
    //   * BRANCH ② (a complete measurement with no winner) is NOT this function's business and is not in
    //     this build: its outcome (`measured-no-gain`) and its writer (`NEGATIVE no-gain`) land with the
    //     protocol that produces them, and R17 pins `no-gain`'s writer count at 0 until then.  So a
    //     decision whose set is complete is left EXACTLY as it was found, not re-labelled here.
    //   * BRANCH ③'s outcome is `no-winner` -- an already DECLARED, unwritable outcome -- so this branch
    //     adds no state and no gate, and "we could not finish the comparison" can never be written down
    //     as a property of the device.
    //
    // ⚠️ ORDERING AGAINST THE FINALIZER: the finalizer recomputes the outcome from the INSTALL accounting
    // at every install and again at the once point.  It LEAVES this branch's outcome alone as long as the
    // decision installs nothing (`install.declared == 0`), which is what "no winner" means -- and the (l)
    // case of the self-test EXECUTES that rather than registering it, because the empty-install half has a
    // carrier here.  What has no carrier is the protocol side (⑤): a branch-③ decision that somehow hands
    // rows to the installer.  That half stays a registered window obligation.
    bool noGainSuppressedOnIncompleteSet(Decision & d) {
      if (d.pairsComplete) {
        return false;
      }
      if (!rowsForWrite(d).empty()) {
        return false;
      }
      d.outcome = "no-winner";
      logLine("NOTICE: " + d.deviceName
              + ": no-gain suppressed: candidate set incomplete (" + formatCandidateDrops(d) + ")");
      return true;
    }

    // ── (7d) THE THREE BRANCHES (tune-measurement.md section 2.6(a)) ────────────────────────────────
    // WHAT THIS IS: the selection that turns the judgement into an OUTCOME.  It is the only producer of
    // `measured-no-gain` and the only place the three branches are ordered, and it decides nothing about
    // the measurement itself: by the ruling the gate ran first (it judges the measurement), the interval
    // layer second, and this third.
    //
    // THE BRANCHES, with the plan wording:
    //   ① a candidate that is not worse than stock on EVERY covered shape => `measured`, and the row set
    //      is written (completeness does NOT veto it: what an incomplete set drops is an ALTERNATIVE,
    //      section 2.6(a)).  BUT the row-set claim requires the WINNER CONFIRMATION step first -- same
    //      paired test, same alpha, FRESH blocks, section 2.3 (4) -- and this build has not declared the
    //      confirmation budget (`R_confirm` is written into the cost model and never given a value), so
    //      the confirmation CANNOT run.  Fail-closed: no row set, and the outcome is the `no-winner`
    //      class, which is what A2 requires of a failed or missing confirmation ("confirmation failure =>
    //      the `no-winner` class, never a sticky negative entry": a confirmation that did not happen is
    //      NO EVIDENCE, not evidence of equality).
    //   ② no candidate is not-worse AND the pair set is complete => `measured-no-gain`, whose durable
    //      effect (declared in the outcome table) is the `NEGATIVE no-gain` entry.
    //   ③ otherwise => `no-winner`, and `noGainSuppressedOnIncompleteSet` is the ONE owner of that
    //      sentence and of its loud (re-implementing it here would give one rule two spellings).
    //
    // TWO THINGS THIS FUNCTION DELIBERATELY DOES NOT DO.  It does not run when the judgement produced no
    // verdicts at all (the interval layer is parked on the missing block-relative normaliser): that is
    // NO DATA, and A2 forbids minting a sticky negative entry from it -- the outcome stays as it was and
    // the write gate refuses it by name.  And it never touches a decision the gate already rejected: the
    // gate verdict is about the measurement, which is upstream of every count here.
    // The confirmation needs the DEVICE, so the backend travels with the decision: it is the same handle
    // `installForHandle` already has, and a null one is a legal argument (the branch then cannot confirm
    // and falls to the `no-winner` class, which is what the self-test exercises without a device).
    void concludeBranches(Decision & d, ggml_backend_t backend) {
      const Decision::Judgement & j = d.judgement;
      if (j.gateRejected) {
        return;   // the gate settled it (and cleared the rows) before this ran
      }
      if (!d.measurement.ok || !j.verdictsComputed) {
        return;   // nothing was compared: the outcome stays `running`, and `writeCacheIfAllowed` refuses
                  // it with its own sentence (which is why no second loud is minted here)
      }
      if (j.winnerUnstable) {
        // 🔴 A3 ②, AS A GATE (plan tune-measurement.md:105/109-111).  The winning claim sits within
        // `STABILITY_BOUNDARY_SE` standard errors of the tolerance line, so the outcome is the DECLARED,
        // NON-WRITABLE `unstable` plus this loud -- and NO row set.  MEASURED before this branch existed
        // (the first live winner): `winner=2`, `winner_unstable=1`, `winner_margin=0.0018`, and the run was
        // stopped by the CONFIRMATION step instead, i.e. the boundary criterion fired and had no power while
        // the layer claimed a winner the priority order says it may not claim.  The plan's line 121 is the
        // sharper half: `unstable` may sit in the outcome table only because its PRODUCER lands in this
        // batch, and until now there was no producer at all.
        // WHY IT IS REGRESSION TO THE MEAN AND NOT THE WINNER'S CURSE (operator's closure on that run, and
        // why the two are not the same guard): with t = 1.9646 at 512 pairs and the tolerance line as the
        // reference, `SE = (ciHigh - ciLow)/(2t) = 0.0036 = 2 x 0.0018`, so the margin sat EXACTLY at one SE
        // -- "a barely-crossing candidate returns to centre when re-measured", rather than "the maximum of
        // noise was selected".  Both send you back to fresh blocks; only this one explains a margin that was
        // already at the boundary when it was selected.
        if (d.outcome == "running") {
          d.outcome = "unstable";
        }
        logLine("NOTICE: " + d.deviceName + ": candidate " + to_string(j.winner) + " is not worse than stock "
                "on every covered shape, but its claim sits at the stability boundary (margin "
                + to_string(j.winnerMargin) + " against STABILITY_BOUNDARY_SE="
                + to_string(TuneStats::STABILITY_BOUNDARY_SE) + " standard errors), so the outcome is the "
                  "declared non-writable `unstable` and NO row set is written: a claim that would not survive "
                  "its own re-measurement is not a claim this layer is allowed to remember");
        return;
      }
      if (j.winner >= 0) {
        // ① THE WINNER, WITH ITS CONFIRMATION STEP IN FRONT OF IT (section 2.3 (4)): the same paired test,
        // the same alpha, and blocks the SELECTION never saw.  The budget is `TuneStats::PAIRS_FLOOR` --
        // the same DECLARED value the pair floor uses, which is the whole rationale of the operator ruling
        // (2026-09-29): reusing the one declared value means there is no second constant to keep in step,
        // and if the confirmation turns out to be under-powered the value to change is `PAIRS_FLOOR`
        // itself (with the policy version bumped in the same batch), not a new knob.
        //
        // WHY ONE CALL PER COVERED SHAPE: the row set's claim is a CONJUNCTION over shapes, and the
        // alignment variant a shape dispatches is a function of ITS OWN k (`vk_tile_row_align_for`), while
        // the object array is shared by every shape of a call.  One call per shape keeps "the variant
        // measured" and "the variant that takes effect" the same thing (K1) instead of measuring one
        // shape's variant on another shape.  The cost is the same blocks (`nShapes x 3 x R_confirm`), which
        // is the term the cost model already carries.
        //
        // A CONFIRMATION THAT CANNOT RUN IS NOT EVIDENCE.  A refused call (no device, no preconditions), an
        // unformable interval, or a shape whose interval does not clear the tolerance all fall to the
        // `no-winner` class and NEVER mint a negative entry (A2: what a sticky negative entry needs is
        // positive evidence, and a winner chosen from the blocks it was selected on is the winner-curse
        // shape section 2.3 exists to refuse).
        const uint32_t nShapes = (uint32_t) d.measurement.shapes.size();
        bool confirmed = d.measurement.ok && nShapes > 0 && backend != nullptr;
        uint32_t confPairs = 0;
        uint32_t confRejected = 0;
        for (uint32_t s = 0; s < nShapes && confirmed; s++) {
          const vk_tune_probe_shape & shape = d.measurement.shapes[s];
          vk_tune_row_spec row {};
          row.src0_type = d.candidateRowSrc0;
          row.src1_type = d.candidateRowSrc1;
          row.key_m = shape.m;
          row.key_k = shape.k;
          row.path = d.candidatePath;
          row.geom = d.candidateGeoms[(size_t) j.winner];
          vk_tune_probe_object objs[3];
          for (uint32_t a = 0; a < 2; a++) {
            objs[a].stock = true;      // the anchor and the A/A sentinel, as in the measurement
            objs[a].rowIndex = 0;
            objs[a].aligned = false;
          }
          const uint32_t alignFor = vk_tile_row_align_for(row.path, row.geom);
          objs[2].stock = false;
          objs[2].rowIndex = 0;
          objs[2].aligned = alignFor > 0 && (row.key_k % alignFor) == 0;
          vk_tune_probe_params cp {};
          cp.shapes = &shape;
          cp.nShapes = 1;
          cp.objects = objs;
          cp.nObjects = 3;
          cp.shots = d.precheckShots;
          cp.warmupBlocks = (uint32_t) TuneStats::WARMUP_BLOCKS;
          cp.reps = (uint32_t) TuneStats::PAIRS_FLOOR;   // the declared pair floor, reused (see above)
          cp.stallBlocks = (uint32_t) TuneStats::STALL_BLOCKS;
          cp.coverageComplete = d.coverageComplete;
          cp.shotsPerShape = nullptr;
          cp.nShotsPerShape = 0;
          // A DIFFERENT rotation cycle than the selection used, so these are blocks the selection never
          // saw: the seed is the declared one, the cycle number is what changes.
          cp.seed = TuneStats::PROBE_ROTATION_SEED;
          cp.spikeBandLo = TuneStats::SPIKE_BAND_LO;
          cp.spikeBandHi = TuneStats::SPIKE_BAND_HI;
          std::vector<double> confD((size_t) cp.nObjects * (size_t) cp.reps, 0.0);
          vk_tune_probe_result cres {};
          // ⚠️ NOT named `cerr`: the guardrail that pins "the shell writes its lines through `logLine`"
          // scans for that token, and a local buffer called `cerr` reads as a `std::cerr` site outside the
          // writer (measured: R10e-shell went red on these three lines).  The name is the lesson.
          char confErr[512];
          confErr[0] = '\0';
          if (!::ggml_backend_vk_tune_probe_candidate(backend, &row, 1, &cp, confD.data(), confD.size(),
                                                      &cres, confErr, sizeof(confErr))
              || cres.st != VK_TUNE_PROBE_OK) {
            confirmed = false;   // the call refused: no device, no precondition, no comparison
            break;
          }
          std::vector<double> confSeries;
          for (uint32_t r = 0; r < cp.reps; r++) {
            const double dv = confD[(size_t) (2 * cp.reps) + r];   // the winner object's slot
            if (std::isfinite(dv)) {
              confSeries.push_back(dv);
            }
          }
          const TuneStats::PairedResult cr = TuneStats::pairedSignedRank(confSeries, TuneStats::ALPHA);
          confPairs++;
          if (!cr.ciOk || !TuneStats::notWorseThan(cr, TuneStats::TOLERANCE)) {
            confRejected++;
            confirmed = false;   // this shape failed to re-confirm the winner
          }
        }
        if (confirmed) {
          // THE ROW SET, one per covered shape, in the minimal WRITABLE form (the shape own key, the
          // catalogue own family pair) -- exactly what the measurement measured, now backed by blocks the
          // selection never saw.
          d.rows.clear();
          for (uint32_t s = 0; s < nShapes; s++) {
            vk_tune_row_spec row {};
            row.src0_type = d.candidateRowSrc0;
            row.src1_type = d.candidateRowSrc1;
            row.key_m = d.measurement.shapes[s].m;
            row.key_k = d.measurement.shapes[s].k;
            row.path = d.candidatePath;
            row.geom = d.candidateGeoms[(size_t) j.winner];
            d.rows.push_back(row);
          }
          d.outcome = "measured";
          logLine("NOTICE: " + d.deviceName + ": candidate " + to_string(j.winner) + " confirmed on fresh "
                  "blocks (" + to_string(confPairs) + " covered shape(s), R_confirm="
                  + to_string((long) TuneStats::PAIRS_FLOOR) + "), so the row set is written");
          return;
        }
        if (d.outcome == "running") {
          d.outcome = "no-winner";
        }
        logLine("NOTICE: " + d.deviceName + ": candidate " + to_string(j.winner) + " is not worse than stock "
                "on every covered shape, but the winner confirmation did NOT pass ("
                + to_string(confRejected) + " of " + to_string(confPairs) + " shape(s) failed to re-confirm, "
                  "R_confirm=" + to_string((long) TuneStats::PAIRS_FLOOR) + " blocks), so NO row set is "
                  "written: a winner selected on the same blocks cannot confirm itself, and a confirmation "
                  "that did not happen is not evidence of anything (A2: the `no-winner` class, never a "
                  "sticky negative entry)");
        return;
      }
      if (!d.pairsComplete) {
        noGainSuppressedOnIncompleteSet(d);   // ③, owned there (sentence + loud)
        return;
      }
      if (j.ciMissing > 0) {
        // 🔴 MEASURED (7e-1, the first run after the normaliser landed): `pairsComplete` is the
        // PRIMITIVE's flag about MEASURED pairs, and it says nothing about whether the intervals could be
        // FORMED.  A pair whose 98 paired differences are all exactly zero at the microsecond resolution
        // yields no interval at all, and reading that as "nothing won" is reading NO DATA as a verdict --
        // A2 forbids a sticky negative entry from it in as many words ("a confirmation that did not happen
        // is not evidence").  So branch (2) requires BOTH: every legal pair measured AND every judged pair
        // evaluable.  The run that produced this line read `cand_pairs=18 ci_missing=14` and minted
        // `no-gain` anyway; the count is now the reason it does not.
        if (d.outcome == "running") {
          d.outcome = "no-winner";
        }
        logLine("NOTICE: " + d.deviceName + ": no-gain suppressed: " + to_string(j.ciMissing) + " of "
                + to_string(j.candidatePairs) + " candidate pair(s) produced no interval at all (their "
                  "paired differences are below the measurement resolution), so this run cannot claim that "
                  "nothing won and no durable verdict is written (A2: missing evidence is not evidence)");
        return;
      }
      // ② THE HARD CONCLUSION: every covered shape had every legal pair measured and no candidate is
      //    not-worse.  The durable answer is the EMPTY ROW SET (`ROWS 0`), which is why this branch sets
      //    NO marker: with ONE cache form there is nothing else this key could remember (2026-09-30).
      d.outcome = "measured-no-gain";
      logLine("NOTICE: " + d.deviceName + ": measured, no gain: every legal pair of every covered shape "
              "was measured and no candidate is not-worse than stock, so the durable verdict for this key "
              "is the EMPTY ROW SET (`ROWS 0` with its ROWSET_HASH: use the stock configuration "
              "everywhere), which is the one cache form this build writes");
    }

    // The effect column, read by NAME (see the forward declaration above the write gate).  ONE reader
    // for the three askers -- the write gate (may an empty row set be written?), the writer (what to
    // emit) and the self-test (does each row's fixture hold the row's own effect?) -- so the column
    // cannot drift.
    const char * declaredEffectFor(const Decision & d) {
      for (const DecisionOutcomeSpec & o : DECISION_OUTCOMES) {
        if (d.outcome == o.outcome) {
          return o.effect;
        }
      }
      return nullptr;
    }

    // Does a decision really hold the durable effect its outcome row declares?  The effect column is
    // DECLARED data, so something has to READ it -- otherwise it is a label that can drift from what the
    // writer does.  The self-test calls this for every row (and an unknown effect string is never
    // satisfied, the same fail-closed shape as an unknown condition name).
    bool effectMatches(const Decision & d, const char * effect) {
      const string e = effect;
      if (e == "rows") {
        // `rows` = "the durable artifact is this decision's row set"; with ONE CACHE FORM that includes
        // the EMPTY set (`ROWS 0`: measured, and stock is the best this device has).  So the question is
        // whether the row set may be written at all -- the write gate's own question, ASKED rather than
        // re-spelled, so the column and the writer cannot disagree.
        return !d.negative && cacheEntryWritable(d);
      }
      if (e.rfind("negative:", 0) == 0) {
        return d.negative && d.negativeReason == e.substr(string("negative:").size());
      }
      if (e.rfind("nothing:", 0) == 0) {
        return !d.negative && rowsForWrite(d).empty();
      }
      return false;
    }

    // The extra conditions named by a table row (G-D20 + W3-P1).  Fail-closed: EVERY name in the list
    // must hold, and an unknown name is never satisfied, so a typo in the table cannot make something
    // writable.
    bool outcomeConditionOk(const Decision & d, const char * condition) {
      for (const string & name : splitConditions(condition)) {
        if (!conditionHolds(d, name)) {
          return false;
        }
      }
      return true;
    }

    bool outcomeWritable(const Decision & d) {
      for (const DecisionOutcomeSpec & o : DECISION_OUTCOMES) {
        if (d.outcome == o.outcome) {
          return o.writable && outcomeConditionOk(d, o.condition);
        }
      }
      return false;
    }

    // WHICH condition refused (W3-P1): the writer's message must name its clause, the same way
    // rowsForWrite's mismatch names the sizes and the fold's message names the differing byte.
    // nullptr means "no NAMED condition is the reason" -- an unwritable row, an undeclared outcome, or
    // an unknown condition name -- and in all three the state gate's own sentence applies, EXCEPT that
    // W3b splits the unknown-name case out: `unknownName` is filled in for it, because there the generic
    // sentence is false in the same way it was false for provenance (the outcome it refuses is the one
    // the sentence says may be remembered).
    const ConditionSpec * firstUnsatisfiedCondition(const Decision & d, string * unknownName = nullptr) {
      if (unknownName != nullptr) {
        unknownName->clear();
      }
      for (const DecisionOutcomeSpec & o : DECISION_OUTCOMES) {
        if (d.outcome != o.outcome) {
          continue;
        }
        if (!o.writable) {
          return nullptr;
        }
        for (const string & name : splitConditions(o.condition)) {
          const ConditionSpec * c = findCondition(name);
          if (c == nullptr) {
            // Fail-closed: a name this build does not implement is never satisfied, and the writer says
            // so in its own words rather than borrowing a sentence that contradicts it.
            if (unknownName != nullptr) {
              *unknownName = name;
            }
            return nullptr;
          }
          if (!c->holds(d)) {
            return c;
          }
        }
        return nullptr;
      }
      return nullptr;
    }

    // ── W4b (⑤ 第一刀): THE DECLARED VALUE DOMAIN AND THE COUNT-ONLY PROBE ──────────────────────────
    // Why this exists before anything is measured: the enumerator's job is to answer "is there ANY
    // geometry this device can install", and an honest answer has to say HOW MUCH was searched.  The count
    // is what decides whether the declared domain can be swept or has to be sampled (w4b-preview.md §9),
    // and it must come from the REAL predicate -- a second implementation in a script would be the second
    // derivation that R34 and F17 forbid.
    //
    // WHERE THE NUMBERS COME FROM (both READ from the source, not guessed):
    //   * the values are the distinct LITERAL entries of the stock `*_warptile*` tables in
    //     ggml-vulkan.cpp (:5142-5160 for the cm2 branch);
    //   * the five-element ORDER is {BLOCK_SIZE, BM, BN, BK, enable_smaller_matrices}, pinned by
    //     mul_mm_cm2.comp:31-36 (the spec constant ids) and spelled out at ggml-vulkan.cpp:5055 -- and
    //     independently by the validity header's own note (vulkan-tile-validity.h:42-44).
    // 🔴 The scalar and cm1 paths have NO declared domain ON PURPOSE: their stock tables are built from
    // dynamic expressions (`subgroup_size_8`, `tm_l`, `s_warptile_wm`, ...), so their numeric domains cannot
    // be read off as literals -- and inventing them here would be exactly the "second spelling of the
    // domain" F17 bans.  The probe reports them LOUDLY as undeclared instead of counting a guess.
    // The fields the domain does not sweep come from the SHADER's own declared defaults
    // (mul_mm_cm2.comp:31-36), which is the one legal source for a value nobody enumerates.
    struct AxisDomain {
      const char* name;
      int axis;
      const uint32_t* values;
      size_t count;
    };
    const uint32_t CM2_BS_VALUES[]      = { 128, 256 };
    const uint32_t CM2_BM_VALUES[]      = { 32, 64, 128 };
    const uint32_t CM2_BN_VALUES[]      = { 64, 128, 256 };
    // 🔴 OPTION D (2026-09-30 ruling 1): the cm2 BK axis DECLARES {64} only.  This declaration IS the
    // old C4-BK exclusion -- the rule that said it (and the PRUNE ledger that counted it) is deleted, so
    // the measured exclusion lives here, with its evidence: BK=128 is LEGAL (stock itself builds
    // `s_warptile_mmq` with BK=128) but measured 33 ms pathological vs 1579.8 us at BK=64
    // (census-t05-shapes.md section 4.4, mm total).  Trying it cannot change "the best", so the search
    // domain is 36 geometries, not 72.  Re-open it only by re-running that BK sweep on this device.
    const uint32_t CM2_BK_VALUES[]      = { 64 };
    const uint32_t CM2_SMALLER_VALUES[] = { 0, 1 };
    const AxisDomain CM2_DOMAIN[] = {
      { "BS",      VK_TILE_AXIS_BS,      CM2_BS_VALUES,      2 },
      { "BM",      VK_TILE_AXIS_BM,      CM2_BM_VALUES,      3 },
      { "BN",      VK_TILE_AXIS_BN,      CM2_BN_VALUES,      3 },
      { "BK",      VK_TILE_AXIS_BK,      CM2_BK_VALUES,      1 },
      { "smaller", VK_TILE_AXIS_SMALLER, CM2_SMALLER_VALUES, 2 },
    };
    const size_t CM2_DOMAIN_SIZE = sizeof(CM2_DOMAIN) / sizeof(CM2_DOMAIN[0]);

    // W4b commit 2: THE STOCK PRESET VECTORS, as literals.  These are the five-element vectors
    // ggml-vulkan.cpp builds its cm2 pipelines from (:5142-5160) and they are TRANSCRIBED rather than
    // re-derived, because an expression here would be the "second spelling of the domain" F17 bans.  The
    // exact value sets are re-derived from those very lines by guardrail R44, and the DISTINCT tuples are
    // covered by R44's matching clause, so a typo in this table is a red guardrail rather than a silently
    // narrower search.  They are DISTINCT (the mmq_k family is a byte-for-byte copy of mmq), which is
    // what makes "how many presets were searched" a number worth stating on its own line.
    const uint32_t CM2_PRESET_VECTORS[][5] = {
      { 256, 128, 256, 64, 1 },   // l_warptile      / l_warptile_mmq      / l_warptile_mmq_k   (:5142/:5150/:5158)
      { 256, 128, 128, 64, 0 },   // m_warptile                                                     (:5143)
      { 128,  64,  64, 64, 0 },   // s_warptile                                                     (:5144)
      { 256, 128, 128, 64, 1 },   // m_warptile_mmq  / m_warptile_mmq_k                             (:5151/:5159)
      { 256,  32,  64, 128, 0 },  // s_warptile_mmq  / s_warptile_mmq_k                             (:5152/:5160)
    };
    const size_t CM2_PRESET_COUNT = sizeof(CM2_PRESET_VECTORS) / sizeof(CM2_PRESET_VECTORS[0]);

    // The values the domain does NOT sweep: the shader's own defaults (mul_mm_cm2.comp:31-36).
    vk_tile_geometry cm2BaseGeometry() {
      vk_tile_geometry g = {};
      g.BM = 64; g.BN = 64; g.BK = 16;
      g.WM = 32; g.WN = 32; g.WMITER = 2;
      g.TM = 4;  g.TN = 2;  g.BS = 256;
      g.pad = -1; g.reshape = -1; g.smaller = -1;
      return g;
    }

    // M1 (t1-predicate.md §0, first meta-discipline).  BOTH directions, because each one alone lies:
    //   (a) an axis we intend to sweep must BE enumerable for the path: sweeping a PINNED or INERT axis is
    //       a no-op that would inflate the count and prove nothing;
    //   (b) every ENUMERABLE axis of the path must HAVE a declared domain: that half is what keeps a missing
    //       declaration from looking like a complete enumeration, and a silent under-count is exactly the
    //       failure F17's completeness predicate exists to prevent.
    bool auditDomain(const AxisDomain* dom, size_t n, int path, string& bad) {
      for (size_t i = 0; i < n; i++) {
        if (vk_tile_axis_class(path, dom[i].axis) != VK_TILE_AXIS_ENUMERABLE) {
          bad = string(dom[i].name) + " is declared but is not ENUMERABLE on this path";
          return false;
        }
      }
      static const int ALL_AXES[] = { VK_TILE_AXIS_BS, VK_TILE_AXIS_BM, VK_TILE_AXIS_BN, VK_TILE_AXIS_BK,
                                      VK_TILE_AXIS_WM, VK_TILE_AXIS_WN, VK_TILE_AXIS_WMITER, VK_TILE_AXIS_TM,
                                      VK_TILE_AXIS_TN, VK_TILE_AXIS_WARP, VK_TILE_AXIS_SPEC_PAD,
                                      VK_TILE_AXIS_SPEC_RESHAPE, VK_TILE_AXIS_SMALLER };
      for (size_t a = 0; a < sizeof(ALL_AXES) / sizeof(ALL_AXES[0]); a++) {
        if (vk_tile_axis_class(path, ALL_AXES[a]) != VK_TILE_AXIS_ENUMERABLE) {
          continue;
        }
        bool declared = false;
        for (size_t i = 0; i < n; i++) {
          if (dom[i].axis == ALL_AXES[a]) { declared = true; break; }
        }
        if (!declared) {
          bad = "axis id " + to_string(ALL_AXES[a]) + " is ENUMERABLE on this path but has NO declared "
                "domain (a sweep would silently under-count)";
          return false;
        }
      }
      return true;
    }

    // The count-only probe: the FULL Cartesian product over the declared domain, every geometry put through
    // the REAL predicate (`vk_tile_candidate_reason`, the shared header's ONE implementation), with refusals
    // bucketed by the predicate's own reason string -- that distribution is the evidence for how the domain
    // should be declared at all (w4b-preview.md §9, "每谓词杀伤分布").
    // It builds NOTHING and writes NOTHING: no pipeline, no file, no decision.  That is what makes it usable
    // before any of the format work, and why it cannot crash a device the way a build can.
    string enumerateCandidateCounts(uint32_t deviceSubgroupSize) {
      const int path = VK_TILE_PATH_CM2;
      string bad;
      if (!auditDomain(CM2_DOMAIN, CM2_DOMAIN_SIZE, path, bad)) {
        // LOUD and explicit: a failed audit must never read as "nothing to report".
        return string("cm2 domain REFUSED by the M1 audit: ") + bad
             + " (nothing was counted; fix the declaration, do not lower the audit)";
      }
      size_t domainSize = 1;
      for (size_t i = 0; i < CM2_DOMAIN_SIZE; i++) {
        domainSize *= CM2_DOMAIN[i].count;
      }
      vector<pair<string, size_t>> kills;   // reason -> count
      size_t legal = 0;
      for (size_t i0 = 0; i0 < CM2_DOMAIN[0].count; i0++) {
        for (size_t i1 = 0; i1 < CM2_DOMAIN[1].count; i1++) {
          for (size_t i2 = 0; i2 < CM2_DOMAIN[2].count; i2++) {
            for (size_t i3 = 0; i3 < CM2_DOMAIN[3].count; i3++) {
              for (size_t i4 = 0; i4 < CM2_DOMAIN[4].count; i4++) {
                vk_tile_geometry g = cm2BaseGeometry();
                g.BS      = CM2_DOMAIN[0].values[i0];
                g.BM      = CM2_DOMAIN[1].values[i1];
                g.BN      = CM2_DOMAIN[2].values[i2];
                g.BK      = CM2_DOMAIN[3].values[i3];
                g.smaller = (int) CM2_DOMAIN[4].values[i4];
                const char* why = vk_tile_candidate_reason(path, g, vk_tile_wave_for(deviceSubgroupSize));
                if (why == nullptr) {
                  legal++;
                }
                else {
                  const string r(why);
                  bool seen = false;
                  for (size_t k = 0; k < kills.size(); k++) {
                    if (kills[k].first == r) { kills[k].second++; seen = true; break; }
                  }
                  if (!seen) {
                    kills.push_back(make_pair(r, (size_t) 1));
                  }
                }
              }
            }
          }
        }
      }
      // Highest kill count first: "which clause is doing the cutting" should be the first thing a reader
      // sees.  A selection sort over a handful of reasons is honest and has no hidden cap.
      for (size_t i = 0; i + 1 < kills.size(); i++) {
        for (size_t j = i + 1; j < kills.size(); j++) {
          if (kills[j].second > kills[i].second) { swap(kills[i], kills[j]); }
        }
      }
      ostringstream os;
      os << "path=cm2 wave=" << vk_tile_wave_for(deviceSubgroupSize) << " axes=" << CM2_DOMAIN_SIZE << " domain=" << domainSize
         << " legal=" << legal << " killed=" << (domainSize - legal);
      if (!kills.empty()) {
        os << " by reason:";
        for (size_t i = 0; i < kills.size(); i++) {
          os << " [" << kills[i].second << "] " << kills[i].first << ";";
        }
      }
      // The undeclared paths are NAMED, not omitted: "we did not count it" and "it has no candidates" must
      // never read the same (lesson #46).
      os << " undeclared=scalar,cm1 (their stock tables are dynamic expressions, not literals)";
      return os.str();
    }

    // ── W4b (⑤ first cut, COMMIT 2): THE CANDIDATE CATALOGUE, THE PRUNE LEDGER, AND THE WRITER ─────
    // The probe above only COUNTS the declared domain and reports the predicate's kill distribution.  This
    // is the part that actually SEARCHES -- and therefore the part that has to state, mechanically, HOW
    // MUCH it searched and WHICH rules narrowed the search, because those two facts are exactly what
    // decide whether an all-reject may be remembered (DECISION_OUTCOMES' four conditions).
    //
    // KINDS: the search domain is declared PER KIND, and `mm` is the only kind this tuner has.  The
    // distinction is load-bearing rather than decorative: `mmq`/`mmq_k` above are real geometry families
    // in the stock sources, but a family whose KIND is not registered here has no domain at all, so
    // nothing about it can be searched, sampled, or concluded.  That is what keeps "not applicable to
    // this binary" from quietly becoming "an empty family that proves something".
    // 🔴 Why the quant families are out of scope is a KIND statement, not a geometry statement:
    // tune-overview.md section 7.2 -- the id path and the quant paths are not touched at all.  Do NOT
    // restate it as "their vectors have six elements" or "their first element is an expression"; those are
    // shape arguments, and a shape argument invites somebody to widen the shape later.
    struct GeometryFamily {
      const char* name;      // "non-quant" / "mmq" / "mmq_k"
      const char* kind;      // a TUNE_KINDS[] name.  A family whose kind is not registered is NOT searched.
      const AxisDomain* axes;
      size_t axisCount;
      // 🔴 WHICH STOCK VECTORS BELONG TO THIS FAMILY.  Membership is DATA, not a derivation: the vectors
      // above are the deduplicated literals of ggml-vulkan.cpp:5142-5160, and "which of them does this
      // family build" is a fact about those source lines, not something the axis-value sets can answer.
      // An earlier draft filtered the whole vector list by the family's declared values, which put
      // `s_warptile_mmq` (BK=128) into the non-quant family -- a wrong answer that only showed up as an
      // off-by-one in the catalogue's counts.
      // 🆕 (7f), 2026-09-29: WHAT CHANGED IS THE ROLE, NOT THE DATA.  The search no longer walks these
      // indices -- it enumerates `axes` above -- because the ruling defines `swept-here` as the declared
      // domain rather than as this family's stock surface.  The membership stays DECLARED because it is
      // still the transcription R44 compares against ggml-vulkan.cpp (`the preset table equals the stock
      // tuples`), and deleting it would delete that comparison rather than move it.  A reader looking for
      // "what was searched" wants the walk in `runCandidateCatalog`; this table answers "what does stock
      // build for this family", which is a different question with the same subject.
      const size_t* presetIndices;
      size_t presetCount;
      bool applicable;       // per THIS binary: a fixed predicate (policy), never a runtime sample
      const char* note;      // why it is applicable or not -- printed, so it cannot be silently dropped
    };

    // The stock cm2 vector list above, indexed.  `l/m/s_warptile` are the non-quant family; `mmq` and
    // `mmq_k` build from their own five (three shared with the non-quant family by having the same values,
    // two of their own).
    const size_t CM2_PRESETS_NONQUANT[] = { 0, 1, 2 };
    const size_t CM2_PRESETS_MMQ[]      = { 0, 1, 2, 3, 4 };
    const size_t CM2_PRESETS_MMQ_K[]    = { 0, 1, 2, 3, 4 };

    // The non-quant cm2 family is the one whose geometry this binary really can select: four src0-type
    // pipelines (f32 / f32_f16 / f16 / f16_f32) are created from ONE warptile vector set
    // (ggml-vulkan.cpp:5724-5727), and each of them is created twice, for the {f16,f32} accumulator
    // (CREATE_MM2, :5710-5713).  The accumulation axis therefore does NOT split the domain -- the same
    // five-element vector feeds every variant -- which is why the declaration is per GEOMETRY family and
    // not per (src0 type, acc) pair, and why no accumulator has to be guessed here.
    // 🔴 The quant families are NOT applicable to this binary, and the reason is a fact about what it
    // compiles (tune-overview.md section 7.2 / :262: KataGo's weights are f16 and the mmq/q8_1 pipelines
    // are never built here), not a shape argument about their vectors.  They are DECLARED rather than
    // omitted so the catalogue's line says "0 searched, and here is why" instead of staying silent.
    const GeometryFamily GEOMETRY_FAMILIES[] = {
      { "non-quant", "mm", CM2_DOMAIN, CM2_DOMAIN_SIZE,
        CM2_PRESETS_NONQUANT, sizeof(CM2_PRESETS_NONQUANT) / sizeof(CM2_PRESETS_NONQUANT[0]), true,
        "cm2's f32/f32_f16/f16/f16_f32 pipelines are created from ONE warptile vector set "
        "(ggml-vulkan.cpp:5724-5727) in both f16acc and f32acc variants (:5710-5713), so this domain "
        "covers every accumulation variant this binary can select" },
      { "mmq", "mm", CM2_DOMAIN, CM2_DOMAIN_SIZE,
        CM2_PRESETS_MMQ, sizeof(CM2_PRESETS_MMQ) / sizeof(CM2_PRESETS_MMQ[0]), false,
        "KataGo never compiles the quantization pipelines (its weights are f16), so no row this tuner "
        "could install would reach them: out of scope by tune-overview.md section 7.2" },
      { "mmq_k", "mm", CM2_DOMAIN, CM2_DOMAIN_SIZE,
        CM2_PRESETS_MMQ_K, sizeof(CM2_PRESETS_MMQ_K) / sizeof(CM2_PRESETS_MMQ_K[0]), false,
        "same as mmq: Qi_K quantization, not compiled on this backend" },
    };
    const size_t GEOMETRY_FAMILY_COUNT = sizeof(GEOMETRY_FAMILIES) / sizeof(GEOMETRY_FAMILIES[0]);

    // ── the CANDIDATE-TABLE VERDICTS, as DATA ────────────────────────────────────────────────────────
    // The predicate (`vk_tile_candidate_reason`) answers "is this geometry SELF-CONSISTENT".  It says so
    // itself, in its own header: "Bounds beyond these sanity checks (BK sweep verdict, bucket membership)
    // belong to the candidate table, not here" (vulkan-tile-validity.h:270-271).  This IS that table --
    // the first two entries of it -- and it exists as DATA for one reason: a rule that lives only in a
    // comment cannot be checked, and R44's second clause has to be able to say "the declared values
    // satisfy the candidate table's verdicts".
    //
    //
    //   E1 (t1-predicate.md:136): `WM == 16 && WN == 16 && BS >= 256` HUNG THE DRIVER, and the custom:
    //     parser lets it through.  A MUST rule (an unusable configuration), and it is already covered by
    //     "skip that combination" rather than by vk_tile_candidate_reason -- which is exactly why it has
    //     to live here.  🔴 NOT REACHED by tonight's catalogue (the geometry presets never carry WM/WN,
    //     and the five-axis domain does not sweep them), so it is RECORDED as data now and MUST land
    //     before the SCALAR domain is ever widened: that is where those two axes become enumerable.
    // A rule is one of two SHAPES, and which one it is determines how it is checked -- so the shape is a
    // tag on the data rather than a name the checker compares against.  (The first draft switched on
    // `r.name`, which meant a second rule could never be added without editing the checker; a rule table
    // that only works for the rules already in it is a table nobody can extend.)
    enum class RuleShape { AxisValue, ForbiddenCombo };
    struct AxisValueRule {
      const int* allowed;
      size_t count;
    };
    // The ForbiddenCombo rules no longer carry their own numbers, and that is the point: the ONE
    // observed driver boundary (E1) is stated in `vulkan-tile-validity.h`
    // (vk_tile_e1_blacklist_hit / vk_tile_e1_blacklist_reason), because it has TWO readers -- this
    // table, which keeps the combination out of the ENUMERATED set, and the backend's install
    // firewall, which must refuse a row that came from a FILE (the cache) and therefore never
    // passed through enumeration.
    // Why a FUNCTION POINTER and not the three ints it used to hold: two hand-written copies of the
    // same boundary drift, and the drift is invisible -- a rule whose numbers say `== 256` while the
    // firewall's says `>= 256` leaves a whole half-line reachable through whichever gate is weaker.
    // Delegating makes "the enumerator and the firewall agree" structural instead of a review habit
    // (G-D26, the same reason the "path is neither scalar, cm1 nor cm2" sentence has one spelling).
    // The path scoping comes along with it: the shared predicate asks vk_tile_axis_class() whether
    // WM/WN/BS are row-settable ENUMERABLE axes on the row's path, so a cm2 row is out of scope by
    // the axis table's own answer rather than by a second, hardcoded path test here.
    using ComboHitFn = bool (*)(int path, const vk_tile_geometry&);
    struct ForbiddenCombo {
      ComboHitFn hit;      // the shared predicate; never nullptr for an enabled rule
    };
    struct CandidateRule {
      const char* name;
      const char* kind;
      RuleShape shape;
      const char* axis;    // the axis fieldA names; the refusal sentence says which value was dropped
      AxisValueRule values;      // shape == AxisValue
      ForbiddenCombo combo;      // shape == ForbiddenCombo
      const char* evidence;  // where the verdict comes from
    };
    const CandidateRule CANDIDATE_RULES[] = {
      { "E1-blacklist", "mm", RuleShape::ForbiddenCombo, nullptr, { nullptr, 0 },
        { vk_tile_e1_blacklist_hit },
        "t1-predicate.md:136 (E1): WM==16 && WN==16 && BS>=256 hung the driver, and the custom: "
        "validation lets it through; the boundary lives in vulkan-tile-validity.h so that this table "
        "and the backend's install firewall (ggml_vk_tuned_row_capability_reason) cannot drift" },
    };
    const size_t CANDIDATE_RULE_COUNT = sizeof(CANDIDATE_RULES) / sizeof(CANDIDATE_RULES[0]);

    // The geometry + the rule outcome of ONE candidate.  `ok` is exactly `negativeProvable`'s input: a
    // candidate the rules rejected may not be measured, so it is neither legal nor excluded.
    struct Candidate {
      vk_tile_geometry geom;
      bool legal;
    };

    bool axisValueAllowed(const int* values, size_t n, int v) {
      for (size_t i = 0; i < n; i++) {
        if (values[i] == v) {
          return true;
        }
      }
      return false;
    }

    // ONE reader for "the value of the named axis in this geometry".  The axis is spelled as the DATA
    // spells it (the same names CM2_DOMAIN uses), so a rule and the domain cannot use two vocabularies.
    // Returns 0 for an axis this build does not know, which every rule treats as "cannot judge" (skip).
    int ruleAxisValue(const vk_tile_geometry& g, const char* axis) {
      if (axis == nullptr) {
        return 0;
      }
      const string a(axis);
      if (a == "BS")      { return (int) g.BS; }
      if (a == "BM")      { return (int) g.BM; }
      if (a == "BN")      { return (int) g.BN; }
      if (a == "BK")      { return (int) g.BK; }
      if (a == "smaller") { return (int) g.smaller; }
      if (a == "WM")      { return (int) g.WM; }
      if (a == "WN")      { return (int) g.WN; }
      return 0;
    }

    // The candidate-table verdict for ONE geometry.  `mustName` non-empty <=> a MUST rule rejected it:
    // an unusable configuration, not a preference.  The MUST violation is NOT an exclusion -- it is
    // unusable, so it belongs in neither the candidate set nor any ledger (the PRUNE direction and its
    // ledger are deleted: option D, 2026-09-30 ruling 1).
    enum class RuleResolution { Admissible, RejectedByMust };

    RuleResolution candidateRuleVerdict(int rulePath, const vk_tile_geometry& g, string& mustName) {
      mustName.clear();
      for (size_t i = 0; i < CANDIDATE_RULE_COUNT; i++) {
        const CandidateRule& r = CANDIDATE_RULES[i];
        if (r.name == nullptr) {
          continue;
        }
        if (r.shape == RuleShape::AxisValue) {
          const int v = ruleAxisValue(g, r.axis);
          if (v == 0 || axisValueAllowed(r.values.allowed, r.values.count, v)) {
            continue;
          }
          mustName = r.name;
        }
        else if (r.shape == RuleShape::ForbiddenCombo) {
          // The verdict comes from the SHARED predicate, and so does the sentence (the firewall
          // returns the same one).  What stays local is only the VOCABULARY the ledger and the log
          // speak: rule names and axis values, which are this table's business.  `ruleAxisValue` is
          // still used for that -- an axis this build does not know returns 0, which is why the
          // readout is spelled as the raw numbers rather than as a claim about why they are wrong.

          // The path is not in the geometry: the catalogue is a per-family walk, and the family's
          // own path is what the shared predicate needs to ask the axis table whether these are
          // row-settable axes at all.  Passed in rather than assumed, so a cm2 catalogue walk cannot
          // inherit a scalar-only boundary.
          if (r.combo.hit == nullptr || !r.combo.hit(rulePath, g)) {
            continue;
          }
          mustName = r.name;
        }
      }
      if (!mustName.empty()) {
        return RuleResolution::RejectedByMust;
      }
      return RuleResolution::Admissible;
    }

    // The catalogue for ONE geometry family: it walks the DECLARED stock vector set (not the Cartesian
    // product -- that is the probe's counting surface) and puts every preset through the REAL predicate and
    // then through the candidate table.  It builds nothing: no pipeline, no file, no measurement.
    //
    // `d` receives the fact the write side needs: `candidateCoverage` (HOW MUCH was searched).  It is set
    // HERE and nowhere else, which is what makes it a property of the enumeration rather than of whoever
    // calls it.
    vector<Candidate> runCandidateCatalog(Decision* d) {
      if (d != nullptr) {
        // 🔴 THE ONE SETTER of `candidateSearched` ((7a)).  At ENTRY, because the fact it records is "this
        // process ran the search for this key" -- whether the walk then produced a candidate is a second
        // fact (`rowSource`) and how much it covered is a third (`candidateCoverage`).  It is NOT derived
        // from either, and the loader never writes it: on a cache hit the load overwrites `rowSource`, and
        // that is exactly the case in which one field would become a lie about the other (R45).
        d->candidateSearched = true;
      }
      vector<Candidate> out;
      const int path = VK_TILE_PATH_CM2;
      for (size_t f = 0; f < GEOMETRY_FAMILY_COUNT; f++) {
        const GeometryFamily& fam = GEOMETRY_FAMILIES[f];
        if (!fam.applicable || !tuneKindKnown(fam.kind)) {
          continue;   // not this binary's family / not a registered kind: nothing is searched or concluded
        }
        string bad;
        if (!auditDomain(fam.axes, fam.axisCount, path, bad)) {
          // LOUD.  A domain that cannot be audited must never read as "this family has no candidates".
          if (d != nullptr) {
            d->note = string("the ") + fam.name + " candidate domain was REFUSED by the M1 audit: " + bad;
          }
          continue;
        }
        vector<string> scopeHere;
        for (size_t k = 0; k < sizeof(TUNE_KINDS) / sizeof(TUNE_KINDS[0]); k++) {
          scopeHere.push_back(TUNE_KINDS[k].name);
        }
        // 🔴 THE SEARCH DOMAIN IS THE DECLARED DOMAIN -- (7f), 2026-09-29.  This walk used to enumerate
        // this family's three STOCK PRESET VECTORS (`CM2_PRESETS_*`), and the comment that stood here
        // argued that the Cartesian product was the wrong thing to search.  The ruling that changed it is
        // tune-measurement.md's 2026-09-29 `swept` decision: "`swept-here` means the DECLARED PRESET
        // DOMAIN, fully enumerated", priced at 3 -> 36 candidate pairs.  The two mistakes that comment
        // warned about are still guarded, by construction rather than by argument:
        //   * `presets=72` (the two-answers failure) cannot recur because no declared number is read here
        //     at all -- `declared`/`postRule` are re-walked by `formatCandidateCatalog` FROM THIS SAME AXIS
        //     TABLE, so the search and the count are one set with three readers;
        //   * pulling the quant family's vectors in cannot recur because the axes walked are `fam.axes`
        //     (this family's own declaration), never a filter over some shared list.
        // 🔴 THREE READERS OF THIS ONE SET, and they must stay one set: this walk (what was searched),
        // `formatCandidateCatalog`'s re-walk (`declared`/`postRule`), and `enumerateCandidateCounts` (the
        // count-only probe with the predicate's kill distribution).  A second way to enumerate the domain
        // is the defect R27's "one set for count / edit / judge" rule exists to prevent.
        const auto considerGeometry = [&](const vk_tile_geometry& g) {
          Candidate c;
          c.geom = g;
          c.legal = false;
          string mustName;
          const RuleResolution verdict = candidateRuleVerdict(path, g, mustName);
          if (verdict == RuleResolution::RejectedByMust) {
            return;   // unusable: it belongs in neither the candidate set nor the set this run searched
          }
          if (vk_tile_candidate_reason(path, g, vk_tile_wave_for(32)) != nullptr) {
            return;
          }
          c.legal = true;
          out.push_back(c);
        };
        // The enumerated set, shape for shape, straight off the declared axis table (the same nesting
        // `formatCandidateCatalog` uses, so the two walks cannot drift apart).
        for (size_t v0 = 0; v0 < fam.axes[0].count; v0++) {
          for (size_t v1 = 0; v1 < fam.axes[1].count; v1++) {
            for (size_t v2 = 0; v2 < fam.axes[2].count; v2++) {
              for (size_t v3 = 0; v3 < fam.axes[3].count; v3++) {
                for (size_t v4 = 0; v4 < fam.axes[4].count; v4++) {
                  vk_tile_geometry g = cm2BaseGeometry();
                  g.BS      = fam.axes[0].values[v0];
                  g.BM      = fam.axes[1].values[v1];
                  g.BN      = fam.axes[2].values[v2];
                  g.BK      = fam.axes[3].values[v3];
                  g.smaller = (int) fam.axes[4].values[v4];
                  considerGeometry(g);
                }
              }
            }
          }
        }
        // The coverage of THIS run, and the kinds it covered: the stock vector set is a SAMPLING SURFACE,
        // not a sweep of the Cartesian domain the probe counts.  Set here, at the one place that knows what
        // was searched -- and OUTSIDE the `if (d != nullptr)` guard's subject, because the guard belongs on
        // the WRITE to the Decision, not on the walk that decides what was searched.
        // ⚠️ `negativeScope` is latent tonight and that is stated rather than hidden: no write path can
        // emit a marker yet (the conditions refuse one, and the completeness gate refuses the file), so
        // this value is not read by anything today.  It is set because it is a FACT ABOUT THE SEARCH, and
        // because the writer refuses an unscoped marker -- when a marker does become reachable, the scope
        // it needs is already the truthful one instead of a literal somebody picks.
        if (d != nullptr) {
          // 🔴 THE PRODUCTION SETTER OF `rowSource = "measured"` (W4b commit 2).  It is the whole point of
          // this batch's write side: the enumerator is the ONLY thing that may say "these candidates are
          // this process's own measurement", and until this cut nothing did (R42 counted 0 production
          // setters and the self-test's named helper was the only way to mint the state).  Set here, in the
          // same breath as the coverage, because the two are one claim: THIS process produced THIS set.
          // 🔴 (7a): this site is on the PRODUCTION path now (the catalogue is a persistent line rather
          // than a bit-8 diagnostic), so "only the enumerator may say it" is REACHABLE rather than merely
          // counted.  R42(d)'s census is unchanged (production 1 / self-test 1); its MEANING changed.
          d->rowSource = "measured";
          // 🔴 (7f): `swept`, and the claim is the one the write gate reads.  It is TRUE here by
          // construction rather than by assertion: the walk above enumerates `fam.axes` to exhaustion
          // (every combination, no early exit), and the only way to reach this line is that the M1 audit
          // of that very domain passed.
          // The old value was `sampled`, and it was right while the walk enumerated three stock vectors.
          d->candidateCoverage = "swept";
          // (7b) step 5: WHICH PATH these geometries belong to, recorded by the one function that knows
          // it (the path is spelled once, at the top of this catalogue).  The measurement builds its
          // candidate rows from that path and must not pick a second one.
          d->candidatePath = path;
          // ... and the PAIR such a row may declare, which is this path's own family and not the
          // shape's: the cm2 tuned-row modules cover f16 x f16 only, while a KataGo shape's declared
          // pair is normally f16 x f32.  Declared here, next to the path, because both are "what the
          // catalogue proposes", and a wrong value is refused loudly by the primitive (R-13 and the
          // shader lookup), never silently measured.
          d->candidateRowSrc0 = (int) VK_TUNE_SRC_F16;
          d->candidateRowSrc1 = (int) VK_TUNE_SRC_F16;
          // R-25's input flag, stated by the component that knows: this walk is the declared STOCK VECTOR
          // SURFACE, a proper subset of the slot's shape set -- so the set handed over is NOT complete, and
          // writing `false` here is a measurement of that fact rather than a default nobody filled in.
          // (vulkantuner.h: it is NOT `candidateCoverage` and NOT `pairsComplete`.)
          d->coverageComplete = false;
          if (d->negativeScope.empty()) {
            d->negativeScope = scopeHere;
          }
        }
        break;   // ONE live family today; a second one joins with its own kind and its own coverage rule
      }
      return out;
    }

    // The catalogue's own line (the probe above reports the DECLARED domain; this reports the SEARCH).
    string formatCandidateCatalog(const Decision& d, const vector<Candidate>& cands) {
      // The declared / post-rule pair PER FAMILY, recomputed from the declaration and the rule table
      // rather than stored, so the two numbers can never disagree with what they describe.  Per family
      // and not in one running variable: with three families in the table, a single pair would report
      // the LAST family's numbers against every family's name.
      size_t declaredCount[GEOMETRY_FAMILY_COUNT] = {};
      size_t postRuleCount[GEOMETRY_FAMILY_COUNT] = {};
      size_t calls = 0, inconsistent = 0;
      for (size_t f = 0; f < GEOMETRY_FAMILY_COUNT; f++) {
        const GeometryFamily& fam = GEOMETRY_FAMILIES[f];
        if (!fam.applicable || !tuneKindKnown(fam.kind)) {
          continue;
        }
        size_t prod = 1;
        for (size_t i = 0; i < fam.axisCount; i++) {
          prod *= fam.axes[i].count;
        }
        declaredCount[f] = prod;
        // Re-walk the product so the "after the rules" number is measured, not derived from the ledger.
        for (size_t v0 = 0; v0 < fam.axes[0].count; v0++) {
          for (size_t v1 = 0; v1 < fam.axes[1].count; v1++) {
            for (size_t v2 = 0; v2 < fam.axes[2].count; v2++) {
              for (size_t v3 = 0; v3 < fam.axes[3].count; v3++) {
                for (size_t v4 = 0; v4 < fam.axes[4].count; v4++) {
                  vk_tile_geometry g = cm2BaseGeometry();
                  g.BS      = fam.axes[0].values[v0];
                  g.BM      = fam.axes[1].values[v1];
                  g.BN      = fam.axes[2].values[v2];
                  g.BK      = fam.axes[3].values[v3];
                  g.smaller = (int) fam.axes[4].values[v4];
                  string mustName;
                  calls++;
                  // The path is the one this catalogue walks, spelled ONCE above it (both call sites
                  // pass it now): the shared ForbiddenCombo predicate needs it to ask the axis table
                  // whether WM/WN/BS are row-settable here, and a defaulted path would be a second
                  // answer to a question the caller already knows.
                  const RuleResolution v = candidateRuleVerdict(VK_TILE_PATH_CM2, g, mustName);
                  // 🔴 A PERMANENT cross-check, and it stays.  The failure this batch hit was a
                  // disagreement between "a rule named a geometry" and "the geometry was refused" (the
                  // first version's `return mustName.empty()` admitted everything a PRUNE had just
                  // excluded).  The verdict and its out-parameter are asserted against EACH OTHER here,
                  // offline, forever, so a rule table that stops refusing what it names is a red line
                  // rather than a plausible-looking number.
                  if (v == RuleResolution::RejectedByMust) {
                    if (mustName.empty()) { inconsistent++; }
                    continue;
                  }
                  if (!mustName.empty()) { inconsistent++; }
                  // `postRule` counts what survives the CANDIDATE TABLE -- it is the
                  // domain this build would be willing to build, before any question of provenance or
                  // coverage.  Counting only what the predicate also accepts would make this number equal
                  // to `legal` below and hide the effect of the rules, which is the one thing the pair
                  // `declared -> postRule` exists to show.
                  postRuleCount[f]++;
                }
              }
            }
          }
        }
      }
      size_t legal = 0;
      for (size_t i = 0; i < cands.size(); i++) {
        if (cands[i].legal) {
          legal++;
        }
      }
      ostringstream os;
      os << "candidate catalog: path=cm2";
      for (size_t f = 0; f < GEOMETRY_FAMILY_COUNT; f++) {
        const GeometryFamily& fam = GEOMETRY_FAMILIES[f];
        os << " family=" << fam.name << " kind=" << fam.kind
           << (fam.applicable && tuneKindKnown(fam.kind) ? " applicable=1" : " applicable=0")
           << " declared=" << declaredCount[f]
           << " postRule=" << postRuleCount[f];
      }
      // The count's token is `candidates=`, NOT `searched=`.  The DECISION line has owned `searched=`
      // since (7a) and it prints a BOOLEAN (`candidateSearched` = "this process ran the search"), so
      // reusing the name here would make one token a count on one line and a flag on the other -- and
      // the two values (36 vs 1) never collide as strings, so no grep or criterion would report it.
      // It is the same class as `coverage=` (a string here, a 0/1 echo on the measurement line).
      os << " candidates=" << cands.size() << " legal=" << legal;
      os << " coverage=" << d.candidateCoverage
         << " remaining=unknown (the format carries no coverage field; coverage IS remembered now that"
            " the search enumerates the declared domain, and the file's own conditions decide whether a"
            " row set or a marker may be written)";
      // The offline cross-check, on the SAME line so the two readings cannot be compared from different
      // runs: `calls` = geometries the candidate table judged, and `inconsistent` = 0 means the rule
      // function's verdict and its out-parameters agreed every time.
      os << " (postRule-check calls=" << calls
         << " inconsistent=" << inconsistent << ")";
      return os.str();
    }

    // ---- the KIND registry and the completeness gate (C4) --------------------------------
    // A cache file is a statement about the WHOLE key, so it may only be written when every kind
    // this identity can select for has a CONCLUSION in this decision: rows this process really
    // installed, or a structural all-reject.  The alternative -- writing the kinds decided so far
    // and merging the rest in later -- is exactly the read-modify-write the ruling forbids: it
    // needs a partial write, a read-back, or a "keep the old rows" rule, and every one of those
    // makes "the file exists" mean less than it says.
    //
    // The kind SET is a function of the IDENTITY, not of the run: KATAGO_VK_FA* is a POLICY knob,
    // so toggling it changes the identity (POLICY is in the header) and one identity never
    // alternates between "mm only" and "mm+fa".  What CAN differ is which kinds a single run
    // concluded (a split `-tune`, or a later per-kind pass), and that is what this gate refuses.
    //
    // TODAY there is exactly one kind, and its conclusion predicate is `cacheEntryWritable`'s, so
    // this gate is deliberately REDUNDANT: it lands before the second kind exists precisely so
    // that adding one cannot be done without a rule here.  A kind with no arm below is an INTERNAL
    // error and refuses the write loudly -- "treat it as concluded" is the one behaviour that
    // would turn the whole rule into a decoration.
    // The KIND registry itself (TUNE_KINDS + tuneKindKnown) is declared next to NEGATIVE_REASONS at the
    // top of this namespace: the parser and the candidate catalogue both need it, and it must be ONE list.
    // The name of the first declared kind that has NO conclusion, or nullptr when complete.
    // `internal` separates "this measurement is partial" (a legitimate state, and the caller says
    // so) from "this build has no rule for that kind" (a defect): both refuse the write.
    const char* firstUnconcludedKind(const Decision & d, bool & internal) {
      internal = false;
      for (const TuneKindSpec & k : TUNE_KINDS) {
        const string name(k.name);
        if (name == "mm") {
          if (!cacheEntryWritable(d)) {
            return k.name;
          }
        }
        else {
          internal = true;
          return k.name;
        }
      }
      return nullptr;
    }

    bool allKindsConcluded(const Decision & d) {
      bool internal = false;
      return firstUnconcludedKind(d, internal) == nullptr;
    }

    // The probe face belongs to the backend (KATAGO_VK_T2_PROBE, bit 8 = the tuned-row surface,
    // which includes its validator self-test).  The shell reads the SAME variable with the SAME
    // grammar -- `all` or a base-10 bitmask, see vk_t2_probe's comment -- so one env asks both
    // sides for their self-checks.  Never reached under KATAGO_VK_TUNE=0: makeDecision returns
    // before this (and the backend suppresses its probes there for the same reason).
    unsigned long t2ProbeBits() {
      const char* v = getenv("KATAGO_VK_T2_PROBE");
      if (v == nullptr || v[0] == '\0') {
        return 0ul;
      }
      // `all` stays 0xF: bit 16 (the W4e probe leg) must NOT be part of it, because a full-probe run
      // also turns bit 4 and bit 8 on and the measurement primitive refuses to measure under either --
      // so "all" would ask for a measurement the primitive is right to refuse.
      return (strcmp(v, "all") == 0) ? 0xFul : strtoul(v, nullptr, 10);
    }

    bool t2ProbeBit8() {
      return (t2ProbeBits() & 8ul) != 0;
    }

    bool t2ProbeBit16() {
      return (t2ProbeBits() & 16ul) != 0;
    }

    // The USER-FACING mode, as an identity string.  ONE producer: the file name's hash segment and
    // the header's readable line both come from here, so they cannot drift.  Today the only
    // dimension is the fp16 request; a FUTURE dimension appends "|k=v" and the file name does NOT
    // get longer (the segment is a fixed-length sha16 either way).  Each value's spelling is
    // enabled_t::toString(), i.e. the SAME one the backend prints in its `[prec] mode=` line.
    string modeString(enabled_t useFP16Mode) {
      return "fp16=" + useFP16Mode.toString();
    }

    // The policy fingerprint (t1-schema.md section 4): build revision + policy version + the
    // policy-relevant env knobs.  Deliberately assembled HERE rather than asked of the
    // backend, because the interface is capped at four functions (R1) and none of these facts
    // needs a Vulkan device to read.  The env values are the *requests*; the resolved variant
    // also depends on the device, which is already in the fingerprint/name, so the pair
    // (device, policy) is fully covered.
    string policyFingerprint() {
    // ── THE DECLARED POLICY CONSTANTS (R-28 step (1), 2026-09-26) ────────────────────────────────
    // ONE table, read by policyFingerprint() (it BUILDS the string from these rows) and by the smoke's
    // A33 clause (it checks that the `POLICY` row the real formatter emits carries them).  `text` is
    // computed FROM the constant, so a value and its spelling cannot drift apart, and removing the loop
    // or a row cannot land green: the emitted `POLICY` row stops carrying the constant.
    //
    // WHY THESE ARE IN THE POLICY AT ALL, and it is the plan's own requirement rather than this batch's
    // idea: tune-gaps.md says "代码 / 谓词 / 规则表 / 枚举器 / 判据 / 门参数 -> b<rev16>; **门参数进
    // policyFingerprint 才能写 class-(b)**" (tune-gaps.md:208, restated at :288).  "class (b)" is the
    // deployed outcome name `gate-reject`, whose carrier is the P12 runtime spike-rate gate (settled by
    // the ruling row tune-gaps.md A1-A; an older reading of this sentence said "every candidate rejected
    // by the validation gate", and one meaning per name is the rule here).  THAT is the condition the
    // outcome table's own comment said it was waiting for, and R-28 step (2) is the batch that both
    // satisfied it (these rows are in the POLICY string, so a gate change changes the key by
    // construction) and spent it (the `gate-reject` row is writable as of that step).
    //
    // The names are LOWERCASE on purpose: every KNOBS[] name is uppercase, so the two segments of POLICY
    // are distinguishable at a glance -- and the smoke's default-leg check relies on exactly that (it
    // asserts that no UPPERCASE |NAME= carries a value, which is what "the default sets no dev knob"
    // means now that the string legitimately contains valued lowercase rows).
    // WHY NOT LEAN ON `b<rev16>` ALONE: the revision is a COARSE automatic identity -- it moves only if a
    // commit moves AND the build actually picks the new revision up, and this repo has produced the
    // opposite (a silently failed build, a regenerated gitinfo: source changed, revision unchanged).
    // In that state a file written under the OLD gate constant would MATCH the stale-revision binary and
    // be adopted -- the one hole the revision chain cannot see.  A value in this string cannot be stale.
    // t1-schema.md's hand-bump list for TUNER_POLICY_VERSION stays for every OTHER code-level semantic
    // change; these rows are simply no longer on it, because here the invalidation is automatic.
    //
    // WHAT BELONGS HERE: a value that changes what a DURABLE VERDICT means -- the spike gate and its
    // band, and the statistics the interval and the block definition are computed with.
    // What does NOT: the self-test's trial count (it changes no verdict), the rotation seed (it changes
    // the ORDER, which is already covered by the revision), and everything that is an env knob.
    struct PolicyConstantSpec {
      const char* name;
      std::string text;
    };
    const PolicyConstantSpec POLICY_CONSTANTS[] = {
      { "spike_rate_gate", std::to_string(TuneStats::SPIKE_RATE_GATE) },
      { "spike_band_lo",   std::to_string(TuneStats::SPIKE_BAND_LO)   },
      { "spike_band_hi",   std::to_string(TuneStats::SPIKE_BAND_HI)   },
      { "alpha",           std::to_string(TuneStats::ALPHA)           },
      { "tolerance",       std::to_string(TuneStats::TOLERANCE)       },
      { "min_block_ms",    std::to_string(TuneStats::MIN_BLOCK_TIME_MS) },
      { "pairs_floor",     std::to_string(TuneStats::PAIRS_FLOOR)     },
      // (7b) / N4: the pre-check's own three numbers.  They belong in this string for the same reason
      // the rows above do -- they decide what a durable verdict MEANS (the warm-up count and K change
      // the block sequence the comparison is computed from), and `stall_blocks` is here by ruling so
      // that changing it invalidates instead of silently mixing regimes.  Their derivations live next
      // to them in tunestats.h, because a value in this string is only auditable if its origin is.
      { "warmup_blocks",   std::to_string(TuneStats::WARMUP_BLOCKS)   },
      { "hot_blocks_k",    std::to_string(TuneStats::HOT_BLOCKS_K)    },
      { "stall_blocks",    std::to_string(TuneStats::STALL_BLOCKS)    },
    };

      // The DEV-SURFACE knobs: everything that can change what this tuner may select (or the
      // computation it selects for), so a dev A/B run cannot silently reuse a cache measured
      // under a different setting.  Added in one cut from the census tool, which is the ONE
      // source for the list (t3-census-env.py; R30 keeps it a subset check, R13k pins the
      // names): the alternative -- proving per knob that the field it gates is already carried
      // by the device fingerprint -- needs statement-scoped extraction per knob, while adding
      // it costs one line and can only cause an extra re-tune.  POLICY lives in the HEADER, so
      // this list grows the header string and never the file name.
      static const char* const KNOBS[] = {
        "KATAGO_VK_FA", "KATAGO_VK_FA_TILE", "KATAGO_VK_FA_PRECISE", "KATAGO_VK_FA_NULL",
        "KATAGO_VK_RMS_SUBGROUP", "KATAGO_VK_M4_FUSION", "KATAGO_VK_MMPREC_F32",
        "GGML_VK_DISABLE_COOPMAT", "GGML_VK_DISABLE_COOPMAT2", "KATAGO_VK_MM_TILE",
        "GGML_VK_DISABLE_BFLOAT16", "GGML_VK_DISABLE_COOPMAT2_DECODE_VECTOR", "GGML_VK_DISABLE_DOT2",
        "GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_FUSION", "GGML_VK_DISABLE_GRAPH_OPTIMIZE",
        "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "GGML_VK_DISABLE_MMVQ", "GGML_VK_DISABLE_MULTI_ADD",
        "GGML_VK_DISABLE_OCP_FP4", "GGML_VK_FORCE_MMVQ", "KATAGO_VK_CM2_BK",
        "KATAGO_VK_CM2_BM", "KATAGO_VK_LAYOUTDBG_EDGE", "KATAGO_VK_LAYOUT_HINT",
        "KATAGO_VK_NARROW_BARRIER", "KATAGO_VK_QKV_NOCONT", "KATAGO_VK_ROPE_DISABLE",
        "KATAGO_VULKAN_LAYOUTDBG", "KATAGO_VULKAN_NOALLOC",
      };
      // NOTE: the build revision is truncated to 16 characters and that is NOT the lossy kind
      // of truncation the knob values below must avoid -- a SHA is a fixed-length prefix-free
      // identifier, so no two revisions share their first 16 hex characters in practice, and
      // keeping it readable in the log is worth more than the extra 48 characters.  Do not
      // "fix" it with the hash rule below: the revision must stay greppable.
      string s = "b" + sanitizeValue(string(GIT_REVISION), 16)
               + "|p" + std::to_string(TUNER_POLICY_VERSION);
      for (const char* knob : KNOBS) {
        const char* v = getenv(knob);
        s += "|";
        s += knob;
        s += "=";
        if (v == nullptr || v[0] == '\0') {
          s += "-";
        }
        else {
          // SHORT VALUES AS-IS, LONG VALUES HASHED -- and the hash is taken over the RAW value.
          // Two earlier mistakes are fixed here: truncating (lossy AND silent: two
          // KATAGO_VK_MM_TILE=custom:... values differing past the cap produced the same
          // fingerprint, so a policy change would have reused the cache), and hashing the
          // SANITISED value (`a\x01b` and `a\x02b` both sanitise to `a_b`, i.e. two different
          // policy values would collide).  The length recorded is the RAW length, for the same
          // reason.  shortHash() output is hex, so it needs no sanitising to stay line-safe.
          const size_t MAX_VALUE = 24;
          const string raw(v);
          if (raw.size() <= MAX_VALUE) {
            // Short enough to read: keep it, but make it line-safe.
            s += sanitizeValue(raw, MAX_VALUE);
          }
          else {
            s += "#" + shortHash(raw) + "/" + std::to_string(raw.size());
          }
        }
      }
      // The declared judgement-layer constants, appended LAST so that a reader of the string sees the
      // knobs (env, human-set) before the values that are baked into this binary.  They are appended by
      // the loop rather than spelled here: POLICY_CONSTANTS is the one place a value and its spelling
      // live, and the loop's absence is visible in the formatter's own output (the self-test probe
      // reports the emitted `POLICY` row and the smoke's A33 votes on it).
      for (const PolicyConstantSpec & c : POLICY_CONSTANTS) {
        s += "|";
        s += c.name;
        s += "=";
        s += c.text;
      }
      return s;
    }

    // FNV-1a over the rows' identity+geometry, spelled out field by field (never over the
    // struct's bytes: padding would make the hash depend on the compiler).  This is what
    // ROWSET_HASH declares, and on the cache channel a mismatch means the file does not
    // describe the rows it contains.
    //
    // FNV-1a 64 with the STANDARD parameters: offset basis 0xCBF29CE484222325 = 14695981039346656037,
    // prime 0x100000001B3 = 1099511628211.  They MUST be the standard values (t1-schema.md 1.3):
    // ROWSET_HASH is a cross-repo contract, so a port that implements FNV-1a from the standard has
    // to agree with us byte for byte.  The empty row set (the four zero bytes of the row count) is
    // pinned by cacheWriteSelfTest() below -- that is the leg that keeps this constant honest.
    string rowSetHash(const vector<vk_tune_row_spec>& rows) {
      uint64_t h = 14695981039346656037ull;
      auto feed = [&h](uint32_t v) {
        for (int i = 0; i < 4; i++) {
          h ^= (v >> (8 * i)) & 0xffu;
          h *= 1099511628211ull;
        }
      };
      feed((uint32_t) rows.size());
      for (const vk_tune_row_spec& r : rows) {
        feed((uint32_t) r.path);
        feed((uint32_t) r.src0_type);
        feed((uint32_t) r.src1_type);
        feed(r.key_m);
        feed(r.key_k);
        feed(r.geom.BM); feed(r.geom.BN); feed(r.geom.BK);
        feed(r.geom.WM); feed(r.geom.WN); feed(r.geom.WMITER);
        feed(r.geom.TM); feed(r.geom.TN); feed(r.geom.BS);
        feed((uint32_t) r.geom.pad);
        feed((uint32_t) r.geom.reshape);
        feed((uint32_t) r.geom.smaller);
      }
      ostringstream os;
      os << hex << h;
      return os.str();
    }

    // ISO-8601 UTC, e.g. 2026-09-13T21:04:05Z.  Informational only: nothing is invalidated by
    // it (POLICY/BASELINE do that), but a human reading the cache wants to know its age.
    string createdAtNow() {
      const std::time_t now = std::time(nullptr);
      std::tm tmUtc {};
#if defined(_WIN32)
      gmtime_s(&tmUtc, &now);
#else
      gmtime_r(&now, &tmUtc);
#endif
      char buf[32];
      std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmUtc);
      return string(buf);
    }

    // Size of the persistent pipeline cache this process LOADED (t1-schema.md section 1,
    // "pipeline cache growth").  Reading the env var rather than asking the backend keeps
    // the interface at four functions (R1) and keeps this purely an accounting question.
    // The "after" half of the accounting can only be read at the next startup, so it is
    // not reported here -- reporting a same-run number as "after" would be fiction.
    bool pipelineCacheBytesBefore(size_t& out) {
      // ONE reader for path-valued env vars (r6, FileUtils::readPathEnv): getenv here would narrow
      // a non-ASCII path to the ANSI code page and this function would then quietly report
      // "unknown" for a cache file that exists.
      const string path = FileUtils::readPathEnv("GGML_VK_PIPELINE_CACHE");
      if (path.empty()) {
        return false;
      }
      ifstream in;
      if (!FileUtils::tryOpen(in, path, ios_base::in | ios_base::binary)) {
        return false;
      }
      in.seekg(0, ios_base::end);
      const streamoff len = in.tellg();
      if (len < 0) {
        return false;
      }
      out = (size_t)len;
      return true;
    }

    // ---- the ONE row-set file format (see vulkantuner.h for the field list) ----------
    // The cache file goes through these functions, so "the file the tuner writes" and "the
    // file a developer drops at the cache path" are the same thing -- there is no second
    // parser to drift.
    struct RowsFile {
      bool foundVersion = false;
      int version = 0;
      bool foundTunerVersion = false;
      int tunerVersion = 0;
      bool foundFingerprint = false;
      string fingerprint;
      bool foundPolicy = false;
      string policy;
      bool foundBaseline = false;
      string baseline;
      bool foundModel = false;
      string model;
      bool foundBoard = false;
      int boardX = 0, boardY = 0;
      bool foundMaxBatch = false;
      int maxBatch = 0;
      bool foundRowSetHash = false;
      string declaredRowSetHash;
      bool foundCreated = false;
      string created;
      bool foundMode = false;
      string mode;
      bool foundDeclaredRows = false;
      long declaredRows = -1;
      vector<vk_tune_row_spec> rows;
      bool negative = false;
      string negativeReason;
      // W4b: WHICH KINDS the NEGATIVE line speaks for, from the closed set of TUNE_KINDS[] names.  It is
      // REQUIRED on the line (an unscoped marker would have to be read as "some kind", the fail-open
      // default the closed set refuses), so an empty vector here can only mean "no NEGATIVE line".
      vector<string> negativeScope;
      // Human-readable keys that were ABSENT.  The loader REFUSES a file carrying any (a
      // machine-written cache has no excuse), and the refusal NAMES them, so "missing" is
      // never read as "verified".
      vector<string> absentKeys;
    };

    // ── W3b: MAY THE LOADER BELIEVE A FILE'S NEGATIVE ENTRY? ─────────────────────────────────────────
    // The read side of the same rule W3-P1 closed on the write side.  There is ONE channel left (the
    // cache), and it is not the same kind of statement as a set THIS process measured:
    //   cache  -- this device's own MEMORY.  It is key-checked (fingerprint, version, policy, baseline,
    //             the outer keys, ROWSET_HASH), it is the file the writer maintains, and the operator's
    //             "I do not trust this" switch is KATAGO_VK_TUNE_RETUNE.  Believing it is the DESIGN.
    //   a file whose rows this process did not measure -- the write side refuses to mint a verdict from
    //             one ("a set that came from a FILE is not evidence that we measured this key"), so a
    //             file that ASSERTS a marker does not make the write side re-assert it.
    // Named and printable for the same reason `negativeProvable` is: R43 pins that the ONE adoption site
    // CITES this predicate instead of re-deriving the rule from a boolean expression in place.
    bool negativeAdoptable(const RowsFile& file) {
      return file.negative && file.rows.empty();
    }

    // Read the whole file rather than using FileUtils::readFile: that helper throws on failure
    // and this path must return an error string instead (every load failure here is fail-open +
    // loud, never an exception out of the hook).  Do not "simplify" this back to readFile().
    bool readWholeFile(const string& path, string& out, string& err) {
      ifstream in;
      if (!FileUtils::tryOpen(in, path, ios_base::in | ios_base::binary)) {
        err = "cannot open " + path;
        return false;
      }
      ostringstream os;
      os << in.rdbuf();
      if (in.bad()) {
        err = "read error on " + path;
        return false;
      }
      out = os.str();
      return true;
    }

    // A durable, atomic write: same-directory tmp file, flushed to the DEVICE (not just to
    // the OS cache), then renamed over the target with a few retries (Windows can fail a
    // rename while another process holds the target open).  There is no fsync helper in
    // core/FileUtils, so the flush is platform-specific -- the precedent is contribute.cpp,
    // which uses FlushFileBuffers for the same reason.
    //
    // What this guarantees: a reader never sees a half-written file, and the bytes are on
    // disk before the rename.  What it does NOT guarantee: that the rename itself is
    // durable (no directory fsync); a power loss can still lose the whole file.  That is
    // acceptable here because the file is a CACHE -- losing it costs one re-tune, which is
    // the same fail-open path as a missing file.
    bool writeFileAtomic(const string& path, const string& content, string& err) {
      // The tmp name carries the pid and a per-process sequence.  Without them two
      // processes writing the same target would share one tmp name, and A's rename could
      // publish B's bytes as A's result (the in-file outer key would then refuse it, but a
      // refusal caused by a race is a bug that should not exist).  The shell deliberately
      // does NOT use the backend's ctx ordinal here: that identity is backend-owned (R7)
      // and the shell must not invent a second numbering.
      static std::atomic<uint32_t> tmp_seq {0};
#if defined(_WIN32)
      const long pid = (long) _getpid();
#else
      const long pid = (long) getpid();
#endif
      const string tmp = path + ".tmp." + std::to_string(pid) + "."
                       + std::to_string(tmp_seq.fetch_add(1u));
      FILE* f = FileUtils::tryFopen(tmp, "wb");
      if (f == nullptr) {
        err = "cannot open " + tmp + " for writing";
        return false;
      }
      const size_t written = content.empty() ? 0 : fwrite(content.data(), 1, content.size(), f);
      const bool wroteAll = (written == content.size());
      const bool flushed = (fflush(f) == 0);
      bool synced = true;
#if defined(_WIN32)
      synced = _commit(_fileno(f)) == 0;
#else
      synced = fsync(fileno(f)) == 0;
#endif
      fclose(f);
      if (!wroteAll) {
        err = "short write to " + tmp;
        FileUtils::tryRemoveFile(tmp);
        return false;
      }
      if (!flushed || !synced) {
        err = "could not flush " + tmp + " to disk";
        FileUtils::tryRemoveFile(tmp);
        return false;
      }
      for (int attempt = 0; attempt < 3; attempt++) {
        if (FileUtils::tryRename(tmp, path)) {
          return true;
        }
      }
      err = "could not rename " + tmp + " to " + path;
      FileUtils::tryRemoveFile(tmp);
      return false;
    }

    // Parse one ROW line.  Field count and integer syntax are checked here; the GEOMETRY is
    // checked by vk_tile_candidate_reason(), the same table the enumerator uses, so a bad
    // row is refused with the identical reason string in both places (t3-shell-plan.md
    // section 1.7 and section 6.1 item 11).
    // `deviceSubgroupSize` comes from Decision.deviceSubgroupSize (injected by the backend through
    // the fingerprint call).  It is a PARAMETER, not a global: one process can hold several devices,
    // and a global would let one device's wave judge another device's rows (G-C3's second pitfall).
    bool parseRowLine(const string& rest, size_t lineNo, uint32_t deviceSubgroupSize, vk_tune_row_spec& out, string& err) {
      istringstream is(rest);
      int src0 = 0, src1 = 0, path = 0;
      long long keyM = 0, keyK = 0;
      long long g[12] = {0};
      string pathToken;
      if (!(is >> pathToken >> src0 >> src1 >> keyM >> keyK)) {
        err = "ROW line " + to_string(lineNo) + ": expected <path> <src0> <src1> <key_m> <key_k>";
        return false;
      }
      for (int i = 0; i < 12; i++) {
        if (!(is >> g[i])) {
          err = "ROW line " + to_string(lineNo) + ": expected 12 geometry fields after key_k";
          return false;
        }
      }
      // The path token is the vk_tile_path enum; accepting the spelling as well keeps a
      // hand-written fixture readable and costs three comparisons.
      if (pathToken == "scalar") path = VK_TILE_PATH_SCALAR;
      else if (pathToken == "cm1") path = VK_TILE_PATH_CM1;
      else if (pathToken == "cm2") path = VK_TILE_PATH_CM2;
      else path = atoi(pathToken.c_str());
      if (path != VK_TILE_PATH_SCALAR && path != VK_TILE_PATH_CM1 && path != VK_TILE_PATH_CM2) {
        err = "ROW line " + to_string(lineNo) + ": unknown path '" + pathToken + "'";
        return false;
      }
      if (keyM < 0 || keyK < 0) {
        err = "ROW line " + to_string(lineNo) + ": key_m/key_k must be >= 0 (0 = whole slot)";
        return false;
      }
      out.src0_type = src0;
      out.src1_type = src1;
      out.key_m = (uint32_t) keyM;
      out.key_k = (uint32_t) keyK;
      out.path = path;
      out.geom.BM = (uint32_t) g[0];
      out.geom.BN = (uint32_t) g[1];
      out.geom.BK = (uint32_t) g[2];
      out.geom.WM = (uint32_t) g[3];
      out.geom.WN = (uint32_t) g[4];
      out.geom.WMITER = (uint32_t) g[5];
      out.geom.TM = (uint32_t) g[6];
      out.geom.TN = (uint32_t) g[7];
      out.geom.BS = (uint32_t) g[8];
      out.geom.pad = (int) g[9];
      out.geom.reshape = (int) g[10];
      out.geom.smaller = (int) g[11];
      // The geometry table needs the wave: BS is derived as num_warps * wave, so whether a row is
      // legal depends on the device it is FOR.  Keeping the clause here (instead of dropping it at
      // parse time) is what preserves the SSOT property -- one bad row, one identical reason string
      // -- for the devices the rows are meant for.
      // R1b-2b close-out (2026-09-17): the CONSUMER guards the wave, not the caller.  Why here and
      // not only at the Decision: `vk_tile_wave_for(0)` returns 8 (it must stay identical to stock's
      // `subgroup_size_8 = max(sg,8)`, so it can never be changed to return 0), which means an
      // unfilled subgroup size would otherwise be judged as wave 8 -- silently legal-or-not for the
      // wrong hardware.  The predicate's own `wave == 0` clause stays as the second line of defence;
      // this one is what a future caller (the ⑤ enumerator) cannot forget, because the value is
      // consumed here.
      if(deviceSubgroupSize == 0) {
        err = "ROW line " + to_string(lineNo) + ": the device's subgroup/wave size is unknown (0), "
              "so no geometry can be judged (a 0 here means the caller never filled it in)";
        return false;
      }
      const char* reason = vk_tile_candidate_reason(path, out.geom, vk_tile_wave_for(deviceSubgroupSize));
      if (reason != nullptr) {
        err = "ROW line " + to_string(lineNo) + ": illegal geometry: " + reason;
        return false;
      }
      return true;
    }

    bool parseRowsFile(const string& text, uint32_t deviceSubgroupSize, RowsFile& out, string& err) {
      istringstream is(text);
      string line;
      size_t lineNo = 0;
      while (getline(is, line)) {
        lineNo++;
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
          line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
          continue;
        }
        if (line.compare(0, 8, "VERSION ") == 0) {
          out.foundVersion = true;
          out.version = atoi(line.c_str() + 8);
        }
        else if (line.compare(0, 14, "TUNER_VERSION ") == 0) {
          out.foundTunerVersion = true;
          out.tunerVersion = atoi(line.c_str() + 14);
        }
        else if (line.compare(0, 12, "FINGERPRINT ") == 0) {
          out.foundFingerprint = true;
          out.fingerprint = line.substr(12);
        }
        else if (line.compare(0, 7, "POLICY ") == 0) {
          out.foundPolicy = true;
          out.policy = line.substr(7);
        }
        else if (line.compare(0, 9, "BASELINE ") == 0) {
          out.foundBaseline = true;
          out.baseline = line.substr(9);
        }
        else if (line.compare(0, 12, "ROWSET_HASH ") == 0) {
          out.foundRowSetHash = true;
          out.declaredRowSetHash = line.substr(12);
        }
        else if (line.compare(0, 8, "CREATED ") == 0) {
          out.foundCreated = true;
          out.created = line.substr(8);
        }
        else if (line.compare(0, 6, "MODEL ") == 0) {
          out.foundModel = true;
          out.model = line.substr(6);
        }
        else if (line.compare(0, 5, "MODE ") == 0) {
          // AFTER the MODEL branch on purpose, and the trailing space is LOAD-BEARING: "MODE " is
          // the only thing that separates this key from "MODEL ", so compare(0, 4, "MODE") would
          // swallow MODEL lines.  The branch order is belt-and-braces on top of that.
          out.foundMode = true;
          out.mode = line.substr(5);
        }
        else if (line.compare(0, 9, "MAXBATCH ") == 0) {
          out.foundMaxBatch = true;
          out.maxBatch = atoi(line.c_str() + 9);
        }
        else if (line.compare(0, 6, "BOARD ") == 0) {
          const string v = line.substr(6);
          const size_t x = v.find('x');
          if (x == string::npos) {
            err = "BOARD line " + to_string(lineNo) + ": expected <X>x<Y>, got '" + v + "'";
            return false;
          }
          out.foundBoard = true;
          out.boardX = atoi(v.substr(0, x).c_str());
          out.boardY = atoi(v.substr(x + 1).c_str());
        }
        else if (line.compare(0, 5, "ROWS ") == 0) {
          out.foundDeclaredRows = true;
          out.declaredRows = atol(line.c_str() + 5);
          if (out.declaredRows < 0) {
            err = "ROWS line " + to_string(lineNo) + ": negative row count";
            return false;
          }
        }
        else if (line.compare(0, 9, "NEGATIVE ") == 0) {
          // Same rule as an unknown key (see the final else): a claim this tuner cannot interpret
          // must not be read as if it had said nothing.  Be precise about WHAT the reason decides
          // (this comment used to overclaim): every reason currently declares the SAME invalidation
          // rule (the whole key set -- see NEGATIVE_REASONS), so the reason does not change how
          // long the claim lives TODAY.  What it changes is what the claim MEANS, and that is
          // enough to refuse a string we do not know: "this device structurally cannot be tuned"
          // and an arbitrary sentence are not the same assertion, and only one of them is safe to
          // act on.  Declaring the rule per reason is what keeps a FUTURE reason with a different
          // lifetime from being added without saying so (R17 pins both fields).
          // W4b: the line is `NEGATIVE <reason> <kind>[,<kind>...]` and BOTH tokens are closed sets --
          // the reason from NEGATIVE_REASONS (what the claim MEANS, and how long it lives) and the scope
          // from TUNE_KINDS (which kinds it is ABOUT).  The scope is what keeps a key-level marker from
          // asserting a kind this run never measured; it is REQUIRED rather than optional, because an
          // absent scope would have to be read as "all kinds" -- a fail-open default with the same shape
          // as accepting an unknown reason.  A malformed line (missing scope, empty token, repeated kind)
          // is refused for the same reason an unknown key is: a claim we cannot interpret must not load
          // as if it had said nothing.
          const string payload = line.substr(9);
          const size_t sp = payload.find(' ');
          const string reason = (sp == string::npos) ? payload : payload.substr(0, sp);
          const string scopeText = (sp == string::npos) ? string() : payload.substr(sp + 1);
          if (!negativeReasonKnown(reason)) {
            err = "NEGATIVE line " + to_string(lineNo) + ": unknown reason '" + reason
                + "' (the reason is a closed set; see NEGATIVE_REASONS)";
            return false;
          }
          if (scopeText.empty()) {
            err = "NEGATIVE line " + to_string(lineNo) + ": no scope (expected 'NEGATIVE <reason> "
                  "<kind>[,<kind>...]'): an unscoped marker does not say which kind it is about, and "
                  "reading it as 'all kinds' would assert kinds this run never measured";
            return false;
          }
          out.negativeScope.clear();
          {
            string cur;
            for (size_t i = 0; i <= scopeText.size(); i++) {
              if (i == scopeText.size() || scopeText[i] == ',') {
                if (cur.empty()) {
                  err = "NEGATIVE line " + to_string(lineNo) + ": empty kind in scope '" + scopeText + "'";
                  return false;
                }
                if (!tuneKindKnown(cur)) {
                  err = "NEGATIVE line " + to_string(lineNo) + ": unknown kind '" + cur
                      + "' in scope (the kind set is TUNE_KINDS; a marker about a kind this build does "
                        "not know must not load)";
                  return false;
                }
                for (size_t k = 0; k < out.negativeScope.size(); k++) {
                  if (out.negativeScope[k] == cur) {
                    err = "NEGATIVE line " + to_string(lineNo) + ": kind '" + cur
                        + "' repeated in scope '" + scopeText + "'";
                    return false;
                  }
                }
                out.negativeScope.push_back(cur);
                cur.clear();
              }
              else {
                cur.push_back(scopeText[i]);
              }
            }
          }
          out.negative = true;
          out.negativeReason = reason;
        }
        else if (line.compare(0, 4, "ROW ") == 0) {
          if (out.rows.size() >= MAX_ROWS_PER_FILE) {
            err = "line " + to_string(lineNo) + ": more than " + to_string(MAX_ROWS_PER_FILE)
                  + " ROW lines (the cap the writer also enforces)";
            return false;
          }
          vk_tune_row_spec row;
          if (!parseRowLine(line.substr(4), lineNo, deviceSubgroupSize, row, err)) {
            return false;
          }
          out.rows.push_back(row);
        }
        else {
          // An unknown key is refused, not ignored: a file written by a NEWER tuner must
          // fail loudly rather than be read as if the extra field did not matter.
          err = "line " + to_string(lineNo) + ": unknown key '" + line.substr(0, line.find(' ')) + "'";
          return false;
        }
      }
      if (out.foundDeclaredRows && out.declaredRows != (long) out.rows.size()) {
        // This is the truncation guard: a file that lost rows in transit would otherwise
        // be read as a perfectly good, shorter row set.
        err = "declared ROWS " + to_string(out.declaredRows) + " but parsed "
              + to_string(out.rows.size()) + " ROW lines (truncated or hand-edited file?)";
        return false;
      }
      // The ADVISORY set is built from the declared array, but WHICH FIELD each key is read from
      // cannot come from that array, so the mapping is a table right here.  That mapping is the
      // part that can drift (the previous comment claimed the parser "cannot drift", which was
      // never true of this mapping), and the failure mode is nasty: a key in the array with no
      // mapping is reported ABSENT for every file -- including the ones this tuner just wrote --
      // so every cache load is refused with a message about missing header keys, i.e. the wrong
      // cause.  An unmapped key therefore FAILS THE PARSE and says so.
      struct RequiredKeyFound {
        const char* key;
        bool found;
      };
      const RequiredKeyFound requiredKeyFound[] = {
        { "MODEL",         out.foundModel },
        { "BOARD",         out.foundBoard },
        { "MAXBATCH",      out.foundMaxBatch },
        { "ROWSET_HASH",   out.foundRowSetHash },
        { "CREATED",       out.foundCreated },
        { "VERSION",       out.foundVersion },
        { "TUNER_VERSION", out.foundTunerVersion },
        { "MODE",          out.foundMode },
      };
      for (const char* key : REQUIRED_FOR_CACHE) {
        bool mapped = false;
        bool found = false;
        for (const RequiredKeyFound& rf : requiredKeyFound) {
          if (string(rf.key) == key) {
            mapped = true;
            found = rf.found;
            break;
          }
        }
        if (!mapped) {
          err = string("internal: REQUIRED_FOR_CACHE key '") + key + "' has no found-flag mapping"
                " (a required key with no field to read is reported absent for every file, which"
                " would refuse this tuner's own caches for the wrong reason)";
          return false;
        }
        if (!found) {
          out.absentKeys.push_back(string(key));
        }
      }
      return true;
    }

    // The serialised form.  Kept next to the parser so a field added to one is obviously
    // missing from the other.
    string formatRowsFileImpl(const Decision& d) {
      // W2 (decisions C/D): what gets written is the DURABLE PER-ROW SET, not the declared set -- a row
      // this process did not install must not be written down (w2-preview section 5 item 4: two
      // contexts with different subsets write the INTERSECTION, and dropped rows are NOT in the file;
      // ruling D asserts rowsDropped == 0 on our own files).  The rule itself lives in rowsForWrite(),
      // which the write GATE also calls, so "there is something to write" and "here is what gets
      // written" cannot drift apart.
      //
      // 🔴 The hash below is computed from the rows ACTUALLY EMITTED, not from d.rows: the reader
      // compares the declared hash with the hash of what it parsed, so a filtered file carrying the
      // unfiltered hash would refuse to load -- i.e. a cache that silently never loads.
      const vector<vk_tune_row_spec> rows = rowsForWrite(d);
      ostringstream os;
      os << "VERSION " << SCHEMA_VERSION << "\n";
      os << "TUNER_VERSION " << TUNER_VERSION << "\n";
      os << "FINGERPRINT " << d.fingerprint << "\n";
      os << "POLICY " << d.policyFingerprint << "\n";
      os << "BASELINE " << d.baselineVersion << "\n";
      os << "MODEL " << d.modelIdentity << "\n";
      os << "BOARD " << d.nnXLen << "x" << d.nnYLen << "\n";
      os << "MAXBATCH " << d.maxBatchSize << "\n";
      os << "ROWSET_HASH " << rowSetHash(rows) << "\n";
      os << "CREATED " << createdAtNow() << "\n";
      // The mode is written as READABLE TEXT even though the NAME carries only its hash: the name
      // routes, and this line is what refutes a hand-renamed or cross-machine-copied file.  It is
      // one key holding the whole string (not one key per dimension), so a future mode dimension
      // grows this VALUE and never the key set -- the comparison stays "entire value equal".
      os << "MODE " << d.modeString << "\n";
      // One file = one key, so "this key has rows" and "this key was all-rejected" are
      // mutually exclusive statements.  The writer refuses to emit both; the reader still
      // copes with a hand-edited file containing both (rows win, loudly).  The remaining
      // case -- neither -- has no producer at all since R-29: the cache write path cannot
      // create one (cacheEntryWritable() refuses it before this function is called), so the
      // refusal below is a guard rather than a branch any run can reach.
      // W2: this test now runs on the rows actually emitted.  When every row was dropped the emitted
      // set is EMPTY and a NEGATIVE-only file is exactly the right statement -- which is the
      // structural-all-reject outcome; the old `d.rows` spelling would have refused to write it.
      //
      // 🔴 W4b: the refusal is limited to the SAME KIND, and the general rule is now stated (rather than
      // left to be re-derived): a file may carry ROW lines of one kind AND a NEGATIVE entry about
      // ANOTHER, because those are statements about DIFFERENT kinds and the scope on the NEGATIVE line is
      // what keeps them from contradicting each other.  What the format forbids is one kind being both
      // "here are the rows" and "there is nothing here".  Since rows are mm rows today and TUNE_KINDS has
      // exactly one member, every existing shape is still refused exactly as before -- the behaviour is
      // unchanged, and adding `fa` will not need a second SCHEMA_VERSION bump to make mixing legal.
      auto rowsKind = [](const vk_tune_row_spec& r) -> const char* {
        // The row's kind is implied by which row family it is.  Today one family, so one answer; the
        // mapping belongs HERE, next to the writer that has to decide, and it must move with TUNE_KINDS.
        (void) r;
        return "mm";
      };
      if (!rows.empty() && d.negative) {
        bool sameKind = false;
        for (size_t i = 0; i < d.negativeScope.size(); i++) {
          for (size_t r = 0; r < rows.size(); r++) {
            if (d.negativeScope[i] == rowsKind(rows[r])) {
              sameKind = true;
              break;
            }
          }
          if (sameKind) {
            break;
          }
        }
        // An UNSCOPED marker cannot be told apart from one about the rows' own kind, so it is refused
        // too: the safe direction is the one that cannot write a self-contradicting file.  (The writer
        // refuses an empty scope outright further down; this branch is what keeps the CONTENT rule
        // fail-closed rather than depending on which check happens to run first.)
        if (d.negativeScope.empty() || sameKind) {
          throw StringError("VulkanTuner: refusing to write a file with both ROW and NEGATIVE for the "
                            "SAME kind (one file = one key: for one kind the two are exclusive "
                            "statements; rows of one kind next to a marker about ANOTHER are allowed)");
        }
      }
      // The writer enforces the SAME cap the parser does (MAX_ROWS_PER_FILE).  Refusing to
      // write an over-cap set is the only honest option: writing it would produce a file
      // that the reader then rejects, i.e. a cache that silently never loads.
      if (rows.size() > MAX_ROWS_PER_FILE) {
        throw StringError("VulkanTuner: refusing to write more than "
                          + std::to_string(MAX_ROWS_PER_FILE) + " rows");
      }
      os << "ROWS " << rows.size() << "\n";
      for (const vk_tune_row_spec& r : rows) {
        os << "ROW " << r.path << " " << r.src0_type << " " << r.src1_type
           << " " << r.key_m << " " << r.key_k
           << " " << r.geom.BM << " " << r.geom.BN << " " << r.geom.BK
           << " " << r.geom.WM << " " << r.geom.WN << " " << r.geom.WMITER
           << " " << r.geom.TM << " " << r.geom.TN << " " << r.geom.BS
           << " " << r.geom.pad << " " << r.geom.reshape << " " << r.geom.smaller << "\n";
      }
      if (d.negative) {
        // The NEGATIVE segment.  Without this the negative channel had no WRITER at all: a
        // decision that was (or had loaded) a negative entry written as `ROWS 0` and nothing
        // else -- i.e. the marker was silently dropped and the round trip was broken, which is
        // the same "it looked fine" failure the read side already refuses to make.
        //
        // The reason is taken from the Decision (never a literal here, so R17 can require that)
        // and re-checked against the closed set: the parser refuses an unknown reason, so a file
        // this writer produces is always re-readable.  `d.rows` is empty whenever `d.negative` is
        // set (the guard above throws otherwise), which is also why the ROWSET_HASH written above
        // is the hash of the EMPTY row set -- a deterministic value, not a placeholder.
        if (!negativeReasonKnown(d.negativeReason)) {
          throw StringError("VulkanTuner: refusing to write a NEGATIVE entry with reason '"
                            + d.negativeReason + "' (it is not in the closed set, and a file this "
                            "tuner cannot re-read must not be written)");
        }
        // W4b: the SCOPE is mandatory and closed-set too, and the writer enforces it for the same reason
        // the parser does -- a file this tuner produces must be one it can read back.  Duplicates are
        // collapsed AFTER checking, so the emitted line is canonical without the writer having to trust
        // whatever built the Decision.
        if (d.negativeScope.empty()) {
          throw StringError("VulkanTuner: refusing to write an UNSCOPED NEGATIVE entry (reason '"
                            + d.negativeReason + "'): the marker must name the kinds it is about, "
                            "because 'no scope' cannot be read as 'all kinds'");
        }
        string scopeLine;
        for (size_t i = 0; i < d.negativeScope.size(); i++) {
          if (!tuneKindKnown(d.negativeScope[i])) {
            throw StringError("VulkanTuner: refusing to write a NEGATIVE entry whose scope names the "
                              "unknown kind '" + d.negativeScope[i] + "' (the kind set is TUNE_KINDS)");
          }
          bool seen = false;
          for (size_t k = 0; k < i; k++) {
            if (d.negativeScope[k] == d.negativeScope[i]) { seen = true; break; }
          }
          if (seen) {
            continue;
          }
          scopeLine += (scopeLine.empty() ? "" : ",") + d.negativeScope[i];
        }
        os << "NEGATIVE " << d.negativeReason << " " << scopeLine << "\n";
      }
      return os.str();
    }

    string formatDecisionLine(const Decision& d, int nnXLen, int nnYLen, int maxBatchSize) {
      ostringstream os;
      os << "decision"
         << " device=" << d.deviceName
         << " deviceIdx=" << d.deviceIdx
         << " S=" << nnXLen << "x" << nnYLen
         << " mb=" << maxBatchSize
         << " model=" << d.modelIdentity
         << " policy=" << d.policyFingerprint
         << " baseline=" << d.baselineVersion
         << " rows=" << d.rowCount
         << " source=" << d.rowSource
         // (7a): the SECOND provenance fact, printed next to the first ON PURPOSE.  `source=` is the final
         // source of the candidate set (the loader may have overwritten it); `searched=` is "this process
         // ran the candidate search".  Stored is not enough here: the cache-hit leg is precisely where the
         // pair has to be readable, because `source=cache searched=1` is the only end-to-end witness that
         // the loader moved one field and left the other alone.
         << " searched=" << (d.candidateSearched ? 1 : 0)
         // R-28 step (3) / section 2.6: the TWO denominators, with names that cannot be confused.
         // `cands=` is candidates MEASURED over candidates ENUMERATED (a numerator above the denominator
         // is a wiring bug a reader can SEE, which is why they are printed as a pair), and `pairs=` is
         // legal pairs over shapes.  They are READINGS, not judgements: 0/0 says "no measurement has been
         // accounted for", never "a complete measurement of nothing".
         << " cands=" << d.candidatesMeasured << "/" << d.candidatesEnumerated
         << " pairs=" << d.pairsLegal << "/" << d.shapesCount
         // (7b) step 4: the two readings that consume the shape-set interface's two outputs (R-12: a
         // field only its producer reads must land with its consumer).  `graph_shapes` is the coverage
         // set's size AFTER R-21's filter and `graph_nodes` is the row-table-side MUL_MAT dispatch count
         // those shapes account for; the smoke pins both on a probe leg, so a walk that found nothing
         // reads as 0 rather than as a smaller set.
         << " graph_shapes=" << d.graphShapes.size()
         << " graph_nodes=" << d.graphNodes
         << " tuneMode=" << d.tuneModeText
         << " cache_state=" << d.cacheState
         << " tune_ms=" << d.tuneMs
         << " cache=" << d.cachePath
         << " cache_exists=" << (d.cacheFileExists ? 1 : 0)
         << " pipeline_cache_bytes_before="
         << (d.hasPipelineCacheBytes ? to_string(d.pipelineCacheBytesBefore) : string("unknown"))
         << " fingerprint=" << d.fingerprint
         << " (" << d.note << ")";
      return os.str();
    }

  }  // namespace

  size_t backendFingerprint(ggml_backend_t backend, char* out, size_t outSize, uint32_t* sgOut) {
    return ::ggml_backend_vk_tune_fingerprint(backend, out, outSize, sgOut);
  }

  bool backendInstallRows(ggml_backend_t backend, const vk_tune_row_spec* rows, size_t nRows,
                          unsigned char* rowInstalled, unsigned rowInstalledCap, string& err,
                          vk_tune_install_result& out) {
    vector<char> errBuf(512, '\0');
    const bool ok = ::ggml_backend_vk_tune_install_rows(backend, rows, nRows, errBuf.data(), errBuf.size(),
                                                        rowInstalled, rowInstalledCap, &out);
    if (!ok) {
      err = string(errBuf.data());
    }
    return ok;
  }

  // THE single cache write path (t3-shell-plan.md section 9.1 R16, step "3-3").
  //
  // Why a named function rather than an `if` at the call site: the write permission has to be one
  // recognisable NAME for the guardrail to check, and the check is STRUCTURAL -- this body is the
  // only place a cache writeFileAtomic call may sit, and both permission predicates have to be
  // named in it BEFORE the write.  It is deliberately NOT declared in vulkantuner.h: the header
  // says what the caller (vulkanbackend.cpp) needs, and this is an internal step of the shell.
  //
  // NO CALLER YET, and that is the documented state of step 3-3 rather than an oversight.  The
  // first class-(a) verdict ("this device structurally cannot be tuned") can only come from the
  // enumerator, which is step 5, so at this cut every runnable path ends with "nothing to say" and
  // MUST NOT write.  The consequence is recorded instead of hidden (t1-schema.md section 1): step
  // 3 lands the function and the rules, step 4 lands the end-to-end, and the POSITIVE leg for this
  // write path (a synthetic class-(a) all-reject producing a cache file whose header matches the
  // schema) is step 4's acceptance item.  What this cut CAN check is the negative side: R16's
  // structure, plus the behaviour legs that plant a cache and require it byte-identical and that
  // run with no cache file at all and require none to appear.
  //
  // The CALLER logs; this function only reports -- and that is now TRUE rather than aspirational:
  // formatRowsFile() REFUSES some decisions by throwing (both ROW and NEGATIVE in one file; a
  // reason outside the closed set), so the conversion to an error string happens HERE, once.  The
  // read path states the same contract for the same reason ("every load failure is fail-open +
  // loud, never an exception out of the hook"), and a caller that later moves outside the
  // decision's try/catch must not be able to turn a cache refusal into a crash.  Keeping the
  // function silent is also what keeps one writer per line (section 9.3-9): the message is born
  // with the path that can actually produce it, which is the caller (step 4).
  bool writeCacheIfAllowed(const Decision& d, const TunePolicy& policy, string& err) {
    if (!cacheWriteAllowed(policy)) {
      // Read-only (cfg vulkanTune=off, or any future second meaning of "read-only"): the cache may
      // be READ but never written.  Reaching here is a bug in the caller, so it is refused loudly
      // rather than silently skipped.
      err = "the cache is read-only in this mode";
      return false;
    }
    if (!outcomeWritable(d)) {
      // THE state gate (step 3-4).  "We ran out of time" must never be remembered as "this device
      // cannot be tuned", and neither must a lost device, an IO failure or a run with no winner:
      // those describe THIS attempt, and a cache file describes the DEVICE.  Only a completed
      // measurement or a structural all-reject gets past here (see DECISION_OUTCOMES), and an
      // outcome nobody declared is refused too (fail-closed).
      //
      // W3-P1/W3b: when a NAMED condition is what refused, the message IS that condition's own sentence
      // (it lives in CONDITION_SPECS), because the generic one says the opposite here ("a structural
      // all-reject may be remembered" while refusing exactly one) and would send the next reader to the
      // wrong code.  The provenance sentence is deliberately INDEPENDENT of the content refusal below
      // ("nothing to write: no rows and no negative entry"): in this state that sentence is FALSE -- the
      // negative entry IS there, it is simply not provable -- and the acceptance leg greps for THIS one
      // by its registered marker, so a writer that fell through to the other sentence must not look green.
      // The third case is an unknown condition name (W3b): `unknownCondition` comes back non-empty, and
      // the generic sentence is false there too (same reason as provenance), so it gets the internal one.
      string unknownCondition;
      const ConditionSpec * bad = firstUnsatisfiedCondition(d, &unknownCondition);
      if (!unknownCondition.empty()) {
        err = "internal: this build has no rule for the condition '" + unknownCondition
              + "' named by the '" + d.outcome + "' table row, and an unimplemented condition is never "
                "satisfied -- so the write is refused (the closed set in CONDITION_SPECS and R42's "
                "both-sides clause are what keep this from being a silent fail-open)";
      }
      else if (bad != nullptr && bad->refusal != nullptr) {
        err = bad->refusal(d);
      }
      else {
        err = "the decision is not in a writable state ('" + d.outcome + "'): only a completed "
              "measurement or a structural all-reject may be remembered";
      }
      return false;
    }
    if (!cacheEntryWritable(d)) {
      // 🔴 ONE CACHE FORM (2026-09-30): "nothing to write" NO LONGER means "no rows and no negative
      // entry" -- an EMPTY ROW SET is a durable artifact (`ROWS 0`), and `cacheEntryWritable` admits it
      // for every outcome whose row declares the effect `rows`.  So this gate now fires only for an
      // outcome whose declared effect is NOT a row set and which carries no provable negative entry:
      // a state the table itself says cannot produce a file.  The predicate stays in this body on
      // purpose (R16: the writer states its gates where it writes).
      err = "nothing to write: this outcome's declared durable effect is not a row set and no provable "
            "negative entry is carried (a file that says neither would load as 'this key is already "
            "decided')";
      return false;
    }
    if (!allKindsConcluded(d)) {
      // The completeness gate (C4).  The message names the KIND, because "the file was not
      // written" is not actionable on its own: a partial measurement is a legitimate state that
      // the next run fixes, while a kind with no rule is a defect in this build.
      bool internal = false;
      const char* missing = firstUnconcludedKind(d, internal);
      const string kindName = missing != nullptr ? missing : "?";
      err = internal
        ? ("internal: no conclusiveness rule for kind '" + kindName + "' (a declared kind with no "
           "rule would make the completeness gate a decoration, so the write is refused)")
        : ("the decision is incomplete: kind '" + kindName + "' has no conclusion (no rows and no "
           "structural all-reject), and a cache file states the WHOLE key -- writing only the "
           "kinds decided so far would silently drop the rest, so nothing is written");
      return false;
    }
    // The per-key directory is created HERE and nowhere else, so that "nothing was decided =>
    // nothing is written" covers the directory too (A30 asserts the whole key path is absent).
    // The store ROOT is created when the decision is taken, so ONE level is enough -- which is
    // exactly what MakeDir::make does: it throws when the parent is missing and returns quietly
    // when the directory already exists.
    const size_t lastSlash = d.cachePath.find_last_of("/\\");
    if (lastSlash != string::npos && lastSlash > 0) {
      try {
        MakeDir::make(d.cachePath.substr(0, lastSlash));
      }
      catch (const std::exception& e) {
        err = string("cannot create the cache directory: ") + e.what();
        return false;
      }
    }
    // WRITE-PATH: cache
    try {
      return writeFileAtomic(d.cachePath, formatRowsFile(d), err);
    }
    catch (const std::exception& e) {
      err = string("the cache entry was refused: ") + e.what();
      return false;
    }
  }

  // Exercise the outcome table and the write gates THROUGH the real function, once per process,
  // under probe bit 8.
  //
  // Why this exists (2026-09-13 review of step 3-4): "a measurement-level failure is never
  // written" has no non-vacuous executor at this step -- the cache write path has no caller and
  // there is no measurement loop yet, so an end-to-end leg asserting "no file was created" would
  // pass because nothing ever ATTEMPTS a write, not because the rule works (the vacuous judgement
  // section 9.2 keeps catching).  This calls the real writer with a real Decision for every
  // declared outcome and requires exactly the declared verdict:
  //   * an unwritable outcome must be refused by the write path itself, with a write-allowed
  //     policy (so the refusal comes from the state gate, not from the mode);
  //   * a writable outcome must pass the STATE gate and be stopped only by the read-only policy --
  //     the positive direction is therefore checked without ever writing a file that no real
  //     verdict produced (the real positive leg belongs to step 4).
  // Then it requires that nothing appeared at the probe path.  The four E2E legs that cannot be
  // written yet (timeout / io-error / device-lost / no-winner) are registered for step 5 WITH a
  // mandatory precondition: prove the failure happened first, then assert the file was not created.
  // These are free functions in namespace VulkanTuner, so the ones the self-test drives have to be
  // declared before it.  The finalizer is defined further down (it reads the install accounting, which
  // is what the lower half of this file is organised around); reportVerdict is the ONE place its verdict
  // is turned into a loud line.
  Decision::FinalizeVerdict finalizeOutcome(Decision& d);

  // The durable per-row set, folded over this device's contexts.  `mask` carries one byte per declared
  // row; `mismatch` means the accumulated mask and this context's mask describe different row sets --
  // which CANNOT happen today (the row set is fixed per decision) and is therefore handled FAIL-CLOSED,
  // by returning an empty mask of the NEW size.  🔴 `mismatchAcc` is only meaningful when `mismatch` is
  // true (it is the OLD length, reported so the caller can name both numbers in its warning).
  struct RowFold {
    std::vector<uint8_t> mask;
    bool mismatch;
    std::size_t mismatchAcc;
  };

  // The COMPUTATION of the fold: no logging, no state.  The caller owns the Decision's fields and the
  // ONE message (see the call site in installForHandle for why the message does not live here).
  // Executed by the self-test's (g) and (h) cases -- (g) the per-row AND, (h) the mismatch fail-closed --
  // which is what makes w2-preview.md's long-registered "flip & to |" mutation deterministic at all: the
  // mask is written in the backend AFTER every `continue` and BEFORE the pipeline build
  // (ggml-vulkan.cpp:21400-21401), so two contexts of one device normally agree and no smoke leg can
  // separate `&` from `|`.
  RowFold foldRowInstalled(const std::vector<uint8_t>& acc, bool seen, const std::vector<uint8_t>& ctx) {
    if (!seen) {
      return { ctx, false, 0 };
    }
    if (acc.size() == ctx.size()) {
      std::vector<uint8_t> out(ctx.size());
      for (size_t i = 0; i < ctx.size(); i++) {
        out[i] = (uint8_t) (acc[i] & ctx[i]);   // AND: durable only if EVERY context installed this row
      }
      return { out, false, 0 };
    }
    return { std::vector<uint8_t>(ctx.size(), 0), true, acc.size() };
  }

  void cacheWriteSelfTest(const TunePolicy & policy, const string & cacheDir) {
    TunePolicy readOnly = policy;
    readOnly.off = true;
    const string probePath = cacheDir + "/self-test-must-not-exist.txt";
    int declared = 0, refused = 0, allowed = 0;
    bool ok = true;
    string firstBad;
    // R-29 Cut 5: the moved `A1_export` POLICY assertion's two readings (see the probe block at
    // the end of this function).  Declared here so the final verdict can include them.
    bool policyRowOk = false;
    // 🔴 `policyRowText` starts EMPTY, and the "did not run" sentence is applied where the
    // verdict is REPORTED (see the probe block below).  It used to START with that sentence,
    // while the read-back was guarded by `policyRowText.compare(0, 5, "FAIL:") != 0` -- i.e. by
    // "has not failed yet".  Sentinel and failure text were the same string, so the guard was
    // false on every path and the read-back NEVER RAN: the self-test said "the POLICY-row probe
    // did not run" in every process, and both A33 legs went red on that sentence (measured
    // 2026-09-26).  Empty is now the only "not decided yet", and each outcome sets its own text.
    string policyRowText;
    // R0 legs: pin rowSetHash() to the STANDARD FNV-1a 64 parameters AND to the wire protocol
    // (field set, field order, 4-byte little-endian feed).  Both expected values come from an
    // INDEPENDENT implementation (python, logs-t3/1b2b/r0-*), never from this function:
    //   empty row set (the four zero bytes of the row count) -> basis * prime^4 mod 2^64
    //                                                        = 0x4d25767f9dce13f5
    //   one legal row (count + 17 fields = 18 feeds)          = 0xe3cde8d1133b7112
    // The empty vector alone does NOT pin the protocol: four zero bytes leave the digest
    // insensitive to the xor order, the byte order and the field order.  Sensitivity proof --
    // both values DIFFER from the expected one, so a weakened vector cannot pass by accident:
    //   swap two field positions -> 0x12b27bd83877cd52 ;  feed the bytes big-endian -> 0x85f45fe6df78624c
    // LITERAL DISCIPLINE (t1-schema.md 1.3): these two literals may only come from the
    // independent implementation.  REFERENCE them; never recompute them in place and never edit
    // them to match a red leg -- "make the literal agree with the code" is the ONE way this
    // check can silently become a no-op.
    {
      vk_tune_row_spec v {};
      v.path = VK_TILE_PATH_SCALAR;
      v.src0_type = 1;   // f16
      v.src1_type = 1;   // f16
      v.key_m = 192;
      v.key_k = 512;
      v.geom.BM = 128; v.geom.BN = 128; v.geom.BK = 32;
      v.geom.WM = 32;  v.geom.WN = 64;  v.geom.WMITER = 2;
      v.geom.TM = 2;   v.geom.TN = 2;   v.geom.BS = 256;
      v.geom.pad = 1;  v.geom.reshape = 0; v.geom.smaller = 0;
      // The vector's row must be LEGAL, and that is checked here rather than claimed: a vector
      // over an illegal row would still hash, but it would stop describing a row the tuner can
      // actually produce (and if the predicate ever started rejecting it, nobody would notice).
      // R1b-2b: the wave is an INPUT now, so this check must exercise BOTH generations -- and it
      // SYNTHESIZES them rather than reading the injected value: this function has no device (it runs
      // from the cache-dir self-test) and the wave-dependent clauses still need a writer on both
      // sides.  `v` is the wave32 fixture (BM=128,BN=128,WM=32,WN=64,WMITER=2,TM=2,TN=2,BS=256 =>
      // num_warps = 4*2 = 8 = 256/32), so wave64 must REFUSE it; `v64` is the mirror
      // (WM=WN=64,WMITER=4,TM=TN=4 => num_warps = 2*2 = 4 = 256/64), so wave32 must refuse it.
      // The cross-check is the point: a fixture that only ever passes under one wave would also pass
      // if the parameter were ignored entirely, so it would prove nothing about the wave.
      const uint32_t wave32 = vk_tile_wave_for(32);
      const uint32_t wave64 = vk_tile_wave_for(64);
      const char * why     = vk_tile_candidate_reason(v.path, v.geom, wave32);              // want legal
      const char * why64no = vk_tile_candidate_reason(v.path, v.geom, wave64);              // want refused
      vk_tune_row_spec v64 = v;
      v64.geom.WM = 64; v64.geom.WN = 64; v64.geom.WMITER = 4; v64.geom.TM = 4; v64.geom.TN = 4;
      v64.geom.BS = 256;
      const char * why64ok = vk_tile_candidate_reason(v64.path, v64.geom, wave64);          // want legal
      const char * why32no = vk_tile_candidate_reason(v64.path, v64.geom, wave32);          // want refused
      const string gotEmpty = rowSetHash(vector<vk_tune_row_spec>());
      const string gotOne   = rowSetHash(vector<vk_tune_row_spec>{ v });
      const string wantEmpty = "4d25767f9dce13f5";
      const string wantOne   = "e3cde8d1133b7112";
      if (why != nullptr || why64no == nullptr || why64ok != nullptr || why32no == nullptr
          || gotEmpty != wantEmpty || gotOne != wantOne) {
        ok = false;
        firstBad = "rowSetHash vector: legal=" + string(why == nullptr ? "yes" : why)
                   + " empty=" + gotEmpty + " (want " + wantEmpty + ")"
                   + " one-row=" + gotOne + " (want " + wantOne + ")"
                   + " wave cross-check: v@32=" + (why == nullptr ? "legal" : "REFUSED")
                   + " v@64=" + (why64no == nullptr ? "LEGAL (BUG)" : "refused")
                   + " v64@64=" + (why64ok == nullptr ? "legal" : "REFUSED (BUG)")
                   + " v64@32=" + (why32no == nullptr ? "LEGAL (BUG)" : "refused");
      }
    }
    // W3-P1: THE ONE fixture that may say "these candidates are this process's own measurement".
    // Production has no such setter -- the enumerator is step 5 -- so this helper is the only place the
    // state can be minted, and R42 counts exactly that (production 0 / self-test 1) instead of trusting
    // a comment.  A named helper rather than inline assignments keeps the count a count of STATES
    // rather than of text occurrences.
    auto aMeasuredSet = [](Decision & x) { x.rowSource = "measured"; };
    // W4b: the other state the remaining condition needs, minted the same way.  It must be set
    // EXPLICITLY even though its satisfying value is "swept": `candidateCoverage` starts at the
    // fail-closed "unknown", so a fixture that only relied on the default would satisfy `swept-here` for
    // the wrong reason.
    auto aSweptSet = [](Decision & x) { x.candidateCoverage = "swept"; };

    // One breaker per condition NAME in the closed set: the refusing direction of a condition can only
    // be tested by making THAT condition false.  A name with no breaker returns false, and the caller
    // turns that into a FAIL rather than a skip -- an untestable condition is precisely what this loop
    // exists to prevent (a declared condition that nothing can falsify is a comment with a name).
    auto breakCondition = [](const string & name, Decision & x) {
      if (name == "device-only")   { x.allRejectReasonsAreDevice = false; return true; }
      if (name == "measured-here") { x.rowSource = "none";                 return true; }
      if (name == "swept-here")    { x.candidateCoverage = "sampled";      return true; }

      return false;
    };

    for (const DecisionOutcomeSpec & o : DECISION_OUTCOMES) {
      Decision d;
      d.outcome = o.outcome;
      d.cachePath = probePath;
      d.nnXLen = 19;
      d.nnYLen = 19;
      d.maxBatchSize = 8;
      d.fingerprint = "self-test";
      d.policyFingerprint = "self-test";
      d.baselineVersion = "1";
      d.modelIdentity = "self-test";
      if (o.writable && string(o.klass) == "structural") {
        // The one writable shape that says something without rows.
        d.negative = true;
        d.negativeReason = "no-injection-slot";
        // G-D20: "the device has no injectable slot" is writable only when every rejection reason
        // is a device property; the fixture has to say so explicitly (unset => not writable).
        d.allRejectReasonsAreDevice = true;
        // W3-P1: and the row names a SECOND condition, so the fixture must satisfy that one too --
        // a fixture that only met `device-only` would fail this loop's positive direction, which is
        // how "the table gained a condition nobody's fixture knows about" announces itself.
        aMeasuredSet(d);
        // W4b/option D: the row now names THREE (device-only, measured-here, swept-here), and the scope is
        // mandatory for the writer, so the fixture carries it.  Each new condition gets its state set HERE
        // rather than left to a default: the positive direction of this loop is what makes forgetting one
        // of them a FAIL instead of a silent pass (the same position W3-P1 landed in).
        aSweptSet(d);
        d.negativeScope.clear();
        d.negativeScope.push_back("mm");
      }
      else if (o.writable && string(o.klass) == "gate") {
        // The `gate` class's writable shape, MINTED BY THE REAL GATE rather than assembled here.  A
        // fixture that set `negative` / `negativeReason` by hand would prove the TABLE is self-consistent
        // while proving nothing about the function that has to produce that state -- and the two would
        // be free to drift apart, which is the whole failure mode of a fixture.
        //
        // `aMeasuredSet` is required and that is the row's own condition doing its job: a gate refusal
        // concluded from a candidate set this process did not measure is not a device fact (W3-P1).
        // The numbers carry MARGIN on purpose -- 25 spikes in 100 blocks is a rate whose whole interval
        // sits ABOVE the bound, so this is not a coin flip at the boundary (the (k) case below points the
        // same judgement at a calm reading and requires the other answer).
        // The device name is set because THIS fixture prints (its loud names the device): leaving it
        // empty produces "NOTICE: : spike-rate gate: ...", which reads as a formatting bug rather than
        // as a synthetic decision.  No other fixture in this loop prints anything, so none of them has
        // ever needed this line -- which is exactly why it is easy to forget here.
        d.deviceName = "self-test";
        aMeasuredSet(d);
        rejectBySpikeGate(d, judgeSpikeRate(25u, 100u, 6u));
      }
      else if (o.writable && string(o.klass) == "completed") {
        // 🔴 (7d): THE COMPLETED CLASS IS NO LONGER ONE SHAPE.  The rows-file fixture below is right for a
        // row whose durable effect is `rows`, and WRONG for one whose effect is a NEGATIVE entry: the
        // first version of this loop selected by CLASS alone, so `measured-no-gain` would have been
        // exercised by a rows fixture and the loop would have gone green while asserting the wrong
        // durable effect -- "a green self-test row that asserts the wrong durable effect is worse than a
        // missing row" (section 2.3 of the plan).  So the fixture is now selected by (class, effect), and
        // the negative shape is MINTED BY THE REAL PRODUCER (`concludeBranches`), for the same reason the
        // gate fixture above is minted by the real gate: a hand-assembled state proves the table is
        // self-consistent and nothing about the function that has to produce it.
        aMeasuredSet(d);
        aSweptSet(d);
        if (string(o.effect) == "rows") {
          // The rows-file shape: one legal row, so the CONTENT gate passes and only the state gate
          // is left to check.  The geometry is the one the smoke A3 leg proves legal, and the
          // verdict below re-checks it against the rule table -- a self-test whose own fixture went
          // illegal would otherwise test nothing while looking green.
          vk_tune_row_spec r {};
          r.path = VK_TILE_PATH_SCALAR;
          r.src0_type = 1;
          r.src1_type = 0;
          r.key_m = 0;
          r.key_k = 0;
          r.geom.BM = 64;
          r.geom.BN = 64;
          r.geom.BK = 0;
          r.geom.WM = 32;
          r.geom.WN = 32;
          r.geom.WMITER = 1;
          r.geom.TM = 4;
          r.geom.TN = 4;
          r.geom.BS = 128;
          r.geom.pad = -1;
          r.geom.reshape = -1;
          r.geom.smaller = -1;
          d.rows.push_back(r);
        }
        else {
          // THE NEGATIVE-ENTRY SHAPE, and it is the THIRD BRANCH own executor (section 2.6(a) asks for an
          // executor per branch): a complete pair set (so branch ③ must NOT fire) with no candidate
          // not-worse on every covered shape (so branch ① must NOT fire) => branch ②, whose durable
          // effect the table declares as `negative:no-gain`.  The verdict below then checks that the
          // decision holds exactly that effect, which is what makes this fixture prove the function
          // rather than the table.
          d.deviceName = "self-test";
          // The fixture stands AFTER a successful measurement -- that is the state the branch selection
          // reads, and forgetting it is what the first version of this fixture did: the loop then reported
          // "measured-no-gain was not refused as declared", because the selection returned early on
          // `!measurement.ok` and the decision never reached the outcome the table declares.
          d.measurement.ok = true;
          d.judgement.verdictsComputed = true;
          d.judgement.winner = -1;
          d.pairsComplete = true;
          d.outcome = "running";
          // No device here, and for THIS fixture that is the honest argument: it exercises branch (2), which
          // never reaches the confirmation.  Branch (1) needs a device and is registered as a window item
          // (the self-test has none); its REFUSAL path is the null-backend guard in the shell.
          concludeBranches(d, nullptr);
        }
      }
      string err;
      bool verdict = false;
      const bool stateOk = (outcomeWritable(d) == o.writable);   // the table agrees with itself
      // (7d) and the decision really HOLDS the durable effect its row declares: the column is read here,
      // which is what keeps it from being a label that drifts from what the writer does.  The effect
      // string also DROVE the fixture above (a negative row is minted by the real branch selection), so
      // this assertion is the positive half of "the self-test does not assert the wrong effect".
      const bool effectOk = effectMatches(d, o.effect);
      // The condition's OTHER direction, per row (G-D20): a row that names an extra condition must
      // be REFUSED when that condition does not hold.  Asserting only the satisfying direction would
      // leave the condition itself untested (and would let it decay into a comment).
      // The conditions' OTHER direction, per NAME (G-D20 for the reasons + W3-P1 for provenance): a row
      // that names extra conditions must be REFUSED whenever any one of them does not hold -- asserting
      // only the satisfying direction would leave the condition itself untested (and would let it decay
      // into a comment).  A name with no breaker is a FAIL, not a skip: see `breakCondition`.
      for (const string & cname : splitConditions(o.condition)) {
        Decision dCond = d;
        if (!breakCondition(cname, dCond)) {
          ok = false;
          if (firstBad.empty()) {
            firstBad = string("outcome ") + o.outcome + " names the condition '" + cname
                       + "', which has no breaker in this self-test (so it could never be tested in "
                         "the refusing direction)";
          }
        }
        else if (outcomeWritable(dCond)) {
          ok = false;
          if (firstBad.empty()) {
            firstBad = string("outcome ") + o.outcome + " is writable while its condition ("
                       + cname + ") does not hold";
          }
        }
      }
      // The `completed` fixture builds one legal row (BM=BN=64, WM=WN=32 => num_warps = 4, BS=128),
      // which is a WAVE32 row: synthesized here for the same reason as above (no device in this
      // function), and the writer of that leg documents the wave it assumes.
      const bool fixtureOk = d.rows.empty()
                          || vk_tile_candidate_reason(d.rows[0].path, d.rows[0].geom,
                                                      vk_tile_wave_for(32)) == nullptr;
      if (!o.writable) {
        // The STATE gate itself must refuse, and the error string must name it -- with a
        // write-ALLOWED policy, so the mode cannot be the thing that refused, and with the string
        // checked so the CONTENT gate cannot be mistaken for the state gate.  🔴 The example used to
        // be `gate-reject`, and R-28 step (2) is exactly why it moved: this batch flips that row, so a
        // comment naming it here would have become a false sentence about which gate refused.  The
        // example is now an outcome that is still unwritable AND has nothing to write.
        const bool wrote = writeCacheIfAllowed(d, policy, err);
        verdict = !wrote && err.find("not in a writable state") != string::npos;
        refused++;
      }
      else {
        // A writable outcome must get PAST the state gate: with a read-only policy the refusal has
        // to come from the permission gate, which its message names.  Written this way the
        // positive direction is checked without ever writing a file that no real verdict produced
        // (the first version of this test called the writer with a write-allowed policy here and
        // dutifully created the probe file -- found by the test itself, which is the point).
        string err2;
        const bool wroteReadOnly = writeCacheIfAllowed(d, readOnly, err2);
        verdict = !wroteReadOnly && err2.find("read-only") != string::npos
               && cacheEntryWritable(d);
        allowed++;
      }
      declared++;
      if (!verdict || !stateOk || !fixtureOk || !effectOk) {
        ok = false;
        if (firstBad.empty()) {
          firstBad = string(o.outcome)
                   + (!fixtureOk ? " has an ILLEGAL self-test fixture" : " was not refused as declared");
        }
      }
    }
    // G-D20 leg (R1b-2a): a `structural-all-reject` whose reasons came from OUR implementation
    // gap must be refused by the state gate even though the declared table says the outcome is
    // writable.  This is the RULE's execution body, and it needs no write side: the plumbing that
    // fills `allRejectReasonsAreDevice` from the install accounting is what waits for B5/B6.
    {
      Decision d;
      d.cachePath = probePath;
      d.nnXLen = 19;
      d.nnYLen = 19;
      d.maxBatchSize = 8;
      d.fingerprint = "self-test";
      d.policyFingerprint = "self-test";
      d.baselineVersion = "1";
      d.modelIdentity = "self-test";
      d.outcome = "structural-all-reject";
      d.negative = true;
      d.negativeReason = "no-injection-slot";
      // W3-P1: provenance is satisfied for BOTH readings, so the only thing they differ in is
      // `device-only` -- which is what makes this leg a test of G-D20 rather than of the table as a
      // whole (with the candidates unmet, both readings would be false and the leg would prove nothing).
      // W4b: the other conditions are satisfied for the same reason -- this leg is about ONE of the four,
      // and a leg that left two others unmet would go green for the wrong cause.
      aMeasuredSet(d);
      aSweptSet(d);
      d.allRejectReasonsAreDevice = false;   // one reason came from our own gap
      const bool refusedOurGap = !outcomeWritable(d);
      d.allRejectReasonsAreDevice = true;    // every reason is a device property
      const bool allowedDeviceOnly = outcomeWritable(d);
      if (!refusedOurGap || !allowedDeviceOnly) {
        ok = false;
        if (firstBad.empty()) {
          firstBad = string("G-D20: our-gap structural-all-reject writable=")
                     + (refusedOurGap ? "no" : "YES (BUG)") + ", device-only writable="
                     + (allowedDeviceOnly ? "yes" : "NO (BUG)");
        }
      }
    }

    // R1b-2b close-out: the two ways a wave-less call can reach the geometry table, asserted here
    // because both need NO device -- the earlier registration said "implemented but unreachable on
    // this machine", which was the wrong diagnosis: vk_tile_wave_for(0) is 8, so the predicate's
    // `wave == 0` clause is unreachable through a caller that uses the helper, and what actually
    // protects us is the consumer guard above.  A self-test can reach both directly.
    {
      vk_tile_geometry g0 = {};
      g0.BM = 128; g0.BN = 128; g0.BK = 0;
      g0.WM = 32;  g0.WN = 64;  g0.WMITER = 2;
      g0.TM = 2;   g0.TN = 2;   g0.BS = 256;
      g0.pad = -1; g0.reshape = -1; g0.smaller = -1;
      const char* r0 = vk_tile_candidate_reason(VK_TILE_PATH_SCALAR, g0, 0);
      if(r0 == nullptr || string(r0).find("wave == 0") == string::npos) {
        ok = false;
        if(firstBad.empty())
          firstBad = string("predicate with wave=0 said ") + (r0 == nullptr ? "LEGAL (BUG)" : r0);
      }
      RowsFile rf0; string err0;
      const bool parsed0 = parseRowsFile("ROW scalar 1 1 0 0 32 32 32 0 0 0 0 0 0 0 0 0\n", 0, rf0, err0);
      if(parsed0 || !rf0.rows.empty()) {
        ok = false;
        if(firstBad.empty())
          firstBad = "parseRowsFile accepted a ROW line with an unknown wave (0): rows="
                     + to_string(rf0.rows.size());
      }
      else if(err0.find("subgroup") == string::npos && err0.find("unknown") == string::npos) {
        ok = false;
        if(firstBad.empty())
          firstBad = "the unknown-wave ROW refusal does not name the cause: " + err0;
      }
    }

    // SECOND probe, deliberately a DIFFERENT path (r6): the one above exists to prove the write
    // GATE refuses to write -- its entire verdict is "still absent", so reusing its path for an
    // encoding test would dismantle that assertion.  This one proves the ENCODING path can really
    // publish: written through the real atomic writer, then read back, then removed.  The name is
    // NON-ASCII on purpose, because that is exactly where the ANSI-open / UTF-8-rename split used
    // to break (write succeeded, rename could not see the file, tmp litter left behind).  The name
    // carries the pid so two NN server threads cannot collide, and nothing here persists: a
    // self-test must not leave state behind or read as a decision.
#if defined(_WIN32)
    const long encProbePid = (long) _getpid();
#else
    const long encProbePid = (long) getpid();
#endif
    const string encProbePath = cacheDir + "/self-test-" + to_string(encProbePid) + "-写入.txt";
    bool encWrote = false, encReadBack = false, encRemoved = false;
    {
      string encErr;
      const string encContent = "VERSION 4\nROWS 0\n";
      // WRITE-PATH: self-test
      // The third disk-write path, and deliberately the only UNGATED one: its subject is the
      // ENCODING of a path, so routing it through the gated cache writer would prove nothing (a
      // refusal there is the FIRST probe's subject).  It is removed again just below, so it cannot
      // leave state behind; the structural half (exactly one tag, one call site, inside this
      // function) is pinned by guardrail R16, the runtime half by the smoke's A33 pair.
      encWrote = writeFileAtomic(encProbePath, encContent, encErr);
      if(encWrote) {
        string back;
        encReadBack = readWholeFile(encProbePath, back, encErr) && back == encContent;
        encRemoved = FileUtils::tryRemoveFile(encProbePath);
      }
    }

    // ── W2 (post-review): the finalizer's rules, asserted as JUDGEMENTS ────────────────────────────────
    //
    // Why these live here and not in a smoke leg: (b) needs an install to land between one context's
    // first eval and a later context's install (timing-dependent), and the from-file retirement needs
    // DECLARED rows on a key that also carries a from-file marker (only the enumerator can produce that
    // today).  Neither is deterministically constructible, so both are registered as "no deterministic
    // trigger" in t3-shell-plan.md section 4 rather than faked with a leg that would pass for the wrong
    // reason.  Here the judgement is driven directly -- AND through the real writer, so a deleted branch
    // shows up as a FAIL in this line instead of as a missing NOTICE in some future leg.
    //
    // The cases are the closed set of "what may happen to the marker": (a) a file-sourced set may not create one,
    // (b) one is not created when our own account refutes it, (c) a from-file marker is retired by a
    // durable row, (d) a device-only all-reject DOES create one, (e) our-gap creates none.  (d) is the
    // CONTROL: without it, a finalizer that simply never set a marker would pass all the others.
    bool verdictOk = true;
    string verdictBad;
    auto verdictName = [](Decision::FinalizeVerdict v) -> const char * {
      switch (v) {
        case Decision::FinalizeVerdict::Nothing: return "Nothing";
        case Decision::FinalizeVerdict::Measured: return "Measured";
          case Decision::FinalizeVerdict::AnyRowInstalled: return "AnyRowInstalled";
        case Decision::FinalizeVerdict::OurGap: return "OurGap";
        case Decision::FinalizeVerdict::DeviceOnlyAllReject: return "DeviceOnlyAllReject";
        case Decision::FinalizeVerdict::StaleFromFileNegativeRetired: return "StaleFromFileNegativeRetired";
      }
      return "?";
    };
    // `wantRefusal == nullptr` means "do not call the writer at all" -- used by the two cases in which a
    // write would really produce a file (a durable row, or a legitimate marker): a self-test must not
    // create state, and the `absent` assertion below is what proves it did not.
    auto verdictCase = [&](const char * what, Decision & d, Decision::FinalizeVerdict want,
                           Decision::FinalizeVerdict wantSecond, bool wantNegative,
                           const char * wantRefusal) {
      const Decision::FinalizeVerdict got = finalizeOutcome(d);
      const bool negFirst = d.negative;
      const Decision::FinalizeVerdict again = finalizeOutcome(d);   // idempotence, one call later
      if (got != want || again != wantSecond || negFirst != wantNegative || d.negative != wantNegative) {
        verdictOk = false;
        if (verdictBad.empty()) {
          verdictBad = string(what) + ": verdict " + verdictName(got) + "/" + verdictName(again)
                     + " (want " + verdictName(want) + "/" + verdictName(wantSecond) + "), negative "
                     + (negFirst ? "true" : "false") + "/" + (d.negative ? "true" : "false")
                     + " (want " + (wantNegative ? "true" : "false") + " both times)";
        }
      }
      if (wantRefusal != nullptr) {
        string err;
        const bool wrote = writeCacheIfAllowed(d, policy, err);
        if (wrote || err.find(wantRefusal) == string::npos) {
          verdictOk = false;
          if (verdictBad.empty()) {
            verdictBad = string(what) + ": the REAL writer "
                       + (wrote ? string("wrote a file") : ("refused with '" + err + "'"))
                       + " (want a refusal naming '" + wantRefusal + "')";
          }
        }
      }
    };
    // The registered marker, READ OUT OF THE TABLE rather than spelled a second time here: the
    // self-test must assert the sentence the writer actually prints, so a rename cannot leave it
    // testing a string nobody prints -- and the .cpp keeps ONE occurrence of the literal, which is
    // also what section 9.4 counts.  🔴 W3b: the marker is now a NAMED CONSTANT, because each condition's
    // sentence is a whole sentence this table owns; "the table lost its sentence" can no longer happen
    // silently (the constant is compile-time non-empty), and the checks that the TABLE still carries the
    // condition and that its sentence equals the one the harness greps are R42's (b) and (e).
    const char * measuredMarker = MEASURED_MARKER;

    { // (b) our own account refutes the marker
      Decision d;
      d.deviceName = "self-test";
      d.cachePath = probePath;
      d.anyRowInstalled = true;
      d.install.declared = 1;
      d.install.installed = 0;
      d.allRejectReasonsAreDevice = true;
      // Same shift as (a): the durable channel is closed by provenance before the self-falsification
      // guard can be the thing the WRITER reports; the guard's own effect is the missing marker.
      verdictCase("(b) self-falsified all-reject", d, Decision::FinalizeVerdict::AnyRowInstalled,
                  Decision::FinalizeVerdict::AnyRowInstalled, false, measuredMarker);
    }
    { // (c) a from-file marker is retired by a durable row (one-way: the second call is `Measured`)
      Decision d;
      d.deviceName = "self-test";
      d.cachePath = probePath;
      d.negative = true;
      d.negativeReason = "no-injection-slot";
      d.negativeOrigin = Decision::NegativeOrigin::FromFile;
      d.install.declared = 1;
      d.install.installed = 1;
      vk_tune_row_spec r {};
      r.path = VK_TILE_PATH_SCALAR;
      r.src0_type = 1;
      r.src1_type = 0;
      r.geom.BM = 64; r.geom.BN = 64; r.geom.BK = 0;
      r.geom.WM = 32; r.geom.WN = 32; r.geom.WMITER = 1;
      r.geom.TM = 4; r.geom.TN = 4; r.geom.BS = 128;
      r.geom.pad = -1; r.geom.reshape = -1; r.geom.smaller = -1;
      d.rows.push_back(r);
      verdictCase("(c) from-file marker retired", d, Decision::FinalizeVerdict::StaleFromFileNegativeRetired,
                  Decision::FinalizeVerdict::Measured, false, nullptr);
      // The retirement has to be FOR a durable row, not for "nothing to write": this is that difference.
      if (!cacheEntryWritable(d)) {
        verdictOk = false;
        if (verdictBad.empty()) {
          verdictBad = "(c) the retirement left nothing to write (want the durable row to remain writable)";
        }
      }
    }
    { // (d) CONTROL: a device-only all-reject really does create the marker -- and keeps it
      Decision d;
      d.deviceName = "self-test";
      d.cachePath = probePath;
      d.install.declared = 1;
      d.install.installed = 0;
      d.allRejectReasonsAreDevice = true;
      // W3-P1: the control must satisfy the OTHER conditions, or it would not be a control -- it would be a
      // second refusal case.  With provenance met, the marker is writable: that is the positive half the
      // table still has to allow, and (i) below is its mirror image.  W4b raised the count from two to
      // four, so all four are set here; the scope is what makes the writer accept the marker at all.
      aMeasuredSet(d);
      aSweptSet(d);
      d.negativeScope.clear();
      d.negativeScope.push_back("mm");
      verdictCase("(d) device-only all-reject", d, Decision::FinalizeVerdict::DeviceOnlyAllReject,
                  Decision::FinalizeVerdict::DeviceOnlyAllReject, true, nullptr);
      if (!outcomeWritable(d) || !cacheEntryWritable(d)) {
        verdictOk = false;
        if (verdictBad.empty()) {
          verdictBad = "(d) a device-only all-reject is not writable (the marker would never be written)";
        }
      }
    }
    { // (e) our own gap: the coarse outcome is recorded, and the TABLE refuses to write it (G-D20)
      Decision d;
      d.deviceName = "self-test";
      d.cachePath = probePath;
      d.install.declared = 1;
      d.install.installed = 0;
      d.install.dropped_unsupported_our_gap = 1;
      d.allRejectReasonsAreDevice = false;
      verdictCase("(e) our-gap all-reject", d, Decision::FinalizeVerdict::OurGap,
                  Decision::FinalizeVerdict::OurGap, false,
                  // W3b: the refusal must be `device-only`'s OWN sentence.  It used to assert the generic
                  // one, which said this outcome "may be remembered" while refusing it -- the same class of
                  // false sentence the provenance condition was fixed for in W3-P1.
                  "came from OUR implementation gap");
    }
    { // (f) THE RACE, on ONE decision: the first call concludes "device-only all-reject" and sets OUR
      // marker; a row then installs (the (b) window: another context's install lands between the install
      // path's finalize and the once point's), and the SECOND call must therefore RETRACT that marker --
      // not merely stop setting it.  This is the only case that executes the from-scratch recompute at
      // the top of finalizeOutcome: with that block turned off, (a)-(e) all still pass, which is exactly
      // why this case exists.  The origin is asserted too, because "the marker is gone" and "we still
      // know where it came from" are two different statements and the second half would be a half-state.
      // (No writer call here on purpose: after the transition the durable set is empty, so there is
      // nothing to refuse that (a)/(b)/(e) do not already refuse.)
      Decision d;
      d.deviceName = "self-test";
      d.cachePath = probePath;
      d.install.declared = 1;
      d.install.installed = 0;
      d.allRejectReasonsAreDevice = true;
      const Decision::FinalizeVerdict first = finalizeOutcome(d);
      const bool markerFirst = d.negative;
      const Decision::NegativeOrigin originFirst = d.negativeOrigin;
      d.install.installed = 1;   // the install that arrives between the two finalize calls
      const Decision::FinalizeVerdict second = finalizeOutcome(d);
      if (first != Decision::FinalizeVerdict::DeviceOnlyAllReject || !markerFirst
          || originFirst != Decision::NegativeOrigin::AllReject
          || second != Decision::FinalizeVerdict::Measured || d.negative
          || d.negativeOrigin != Decision::NegativeOrigin::None) {
        verdictOk = false;
        if (verdictBad.empty()) {
          verdictBad = string("(f) all-reject -> measured: verdict ") + verdictName(first) + "/"
                     + verdictName(second) + " (want DeviceOnlyAllReject/Measured), marker "
                     + (markerFirst ? "true" : "false") + "/" + (d.negative ? "true" : "false")
                     + " (want true/false), origin "
                     + (originFirst == Decision::NegativeOrigin::AllReject ? "AllReject" : "?") + "/"
                     + (d.negativeOrigin == Decision::NegativeOrigin::None ? "None" : "?")
                     + " (want AllReject/None)";
        }
      }
    }

    { // (g)/(h) THE FOLD -- the two rules of the write path that had NO executor anywhere before this
      // case: the durable per-row set is the AND of every context's set, and a size mismatch is
      // fail-closed.  Neither can be built as a smoke leg (see foldRowInstalled's comment), so the
      // judgement is executed here, which is also what makes the registered "flip & to |" mutation
      // deterministic.  `mismatchReverse` pins the DIRECTION of the mismatch too: the new size is this
      // context's, not the accumulated one's.
      const std::vector<uint8_t> m11 = { 1, 1 }, m10 = { 1, 0 }, m01 = { 0, 1 }, one = { 1 };
      string foldBad;
      auto wantMask = [&](const char* what, const RowFold& f, std::vector<uint8_t> want,
                          bool wantMismatch, std::size_t wantAcc) {
        const bool ok = (f.mask == want) && (f.mismatch == wantMismatch)
                     && (!wantMismatch || f.mismatchAcc == wantAcc);
        if (!ok && foldBad.empty()) {
          // 🔴 The message must name WHAT differs.  Reporting only (size, mismatch, acc) was blind to the
          // one failure the AND rule can produce: `&` -> `|` keeps the size, the mismatch flag and the
          // accumulated length IDENTICAL and changes only the BYTES, so the first version printed
          // "got size 2, mismatch false, acc 0 (want size 2, mismatch false, acc 0)" -- a message whose
          // actual and expected halves read the same, which is exactly the "error string that does not
          // name the clause" failure this project refuses.
          string detail;
          if (f.mask.size() != want.size()) {
            detail = "mask size " + to_string(f.mask.size()) + " (want " + to_string(want.size()) + ")";
          }
          else {
            std::size_t k = 0;
            while (k < want.size() && f.mask[k] == want[k]) { k++; }
            detail = (k == want.size())
                   ? "the mask bytes match, so the MISMATCH rule is what differs"
                   : ("first differing byte " + to_string(k) + ": got " + to_string((int) f.mask[k])
                      + ", want " + to_string((int) want[k]));
          }
          foldBad = string(what) + ": " + detail + " (mismatch "
                  + (f.mismatch ? "true" : "false") + ", acc " + to_string(f.mismatchAcc)
                  + "; want mismatch " + (wantMismatch ? "true" : "false") + ", acc "
                  + to_string(wantAcc) + ")";
        }
      };
      wantMask("(g) the first context IS the fold", foldRowInstalled(std::vector<uint8_t>(), false, m10),
               { 1, 0 }, false, 0);
      wantMask("(g) both contexts installed", foldRowInstalled(m11, true, m11), { 1, 1 }, false, 0);
      wantMask("(g) THIS context dropped it", foldRowInstalled(m11, true, m10), { 1, 0 }, false, 0);
      wantMask("(g) an EARLIER context dropped it", foldRowInstalled(m10, true, m11), { 1, 0 }, false, 0);
      wantMask("(g) only the other row survived", foldRowInstalled(m01, true, m11), { 0, 1 }, false, 0);
      wantMask("(h) mismatch clears to the NEW size", foldRowInstalled(m11, true, one), { 0 }, true, 2);
      wantMask("(h) mismatch, other direction", foldRowInstalled(one, true, m11), { 0, 0 }, true, 1);
      if (!foldBad.empty()) {
        verdictOk = false;
        if (verdictBad.empty()) {
          verdictBad = "(g)/(h) fold: " + foldBad;
        }
      }
    }

    { // (i) THE MIRROR OF (d), and the shape the acceptance leg (A36) meets in production: the CONTENT
      // gate says writable -- the marker exists and its reason is in the closed set -- while the STATE
      // gate refuses it because the candidate set came from a file.  Both halves are asserted, in this
      // order, because "the write was refused" alone is satisfied by two very different rules, and the
      // whole point of W3-P1 is that the refusal is for PROVENANCE and not because the file had nothing
      // in it.  Without this case the provenance condition would only ever be exercised in its
      // satisfying direction (by the fixtures in the loop above and by (d)).
      Decision d;
      d.deviceName = "self-test";
      d.cachePath = probePath;
      d.install.declared = 1;
      d.install.installed = 0;
      d.allRejectReasonsAreDevice = true;   // the outcome's own condition holds ...
      d.rowSource = "cache";                // ... and the candidates came from a file
      verdictCase("(i) unmeasured all-reject", d, Decision::FinalizeVerdict::DeviceOnlyAllReject,
                  Decision::FinalizeVerdict::DeviceOnlyAllReject, true, measuredMarker);
      const bool contentWritable = cacheEntryWritable(d);
      const bool stateWritable = outcomeWritable(d);
      if (stateWritable || !contentWritable) {
        verdictOk = false;
        if (verdictBad.empty()) {
          verdictBad = string("(i) provenance is not what refused: state-writable=")
                     + (stateWritable ? "YES (BUG)" : "no") + ", content-writable="
                     + (contentWritable ? "yes"
                                        : "NO (BUG: the marker is set and its reason is in the closed "
                                          "set, so the content gate must allow it -- otherwise the "
                                          "refusal could be 'nothing to write', which is FALSE here)");
        }
      }
    }

    { // (j) D3b: THE WRITE PATH IS EXCEPTION-SAFE, and this case is the only executor that clause has.
      // The shape it builds is one the FORMATTER refuses by throwing (a marker AND a row for the SAME
      // kind), and the point is that the refusal comes back as a REFUSAL rather than as an exception out
      // of the shell.  Every other case above either has no rows or a marker of another kind, so without
      // this one the `try`/`catch` around the cache write path would be a clause nothing could
      // falsify -- which is the "declared but never executed" gap this project refuses to ship.  The
      // criterion is the refusal's OWN prefix: a writer that crashed would not produce a string at all.
      Decision d;
      d.deviceName = "self-test";
      d.cachePath = probePath;
      d.outcome = "structural-all-reject";
      d.negative = true;
      d.negativeReason = "no-injection-slot";
      d.negativeOrigin = Decision::NegativeOrigin::AllReject;
      aMeasuredSet(d);
      aSweptSet(d);
      d.allRejectReasonsAreDevice = true;   // or the FIRST unmet condition refuses, and the formatter is never reached
      d.negativeScope.clear();
      d.negativeScope.push_back("mm");   // the SAME kind the row below carries: the formatter refuses that
      d.install.declared = 1;
      d.install.installed = 1;
      vk_tune_row_spec r {};
      r.path = VK_TILE_PATH_SCALAR;
      r.src0_type = 1;
      r.src1_type = 0;
      r.geom.BM = 64; r.geom.BN = 64; r.geom.BK = 0;
      r.geom.WM = 32; r.geom.WN = 32; r.geom.WMITER = 1;
      r.geom.TM = 4; r.geom.TN = 4; r.geom.BS = 128;
      r.geom.pad = -1; r.geom.reshape = -1; r.geom.smaller = -1;
      d.rows.push_back(r);
      string jErr;
      const bool jWrote = writeCacheIfAllowed(d, policy, jErr);
      if (jWrote || jErr.find("the cache entry was refused") == string::npos) {
        verdictOk = false;
        if (verdictBad.empty()) {
          verdictBad = string("(j) the formatter's refusal did not come back as a refusal: wrote=")
                     + (jWrote ? "YES (BUG)" : "no") + ", err='" + jErr + "'";
        }
      }
    }

    { // (k) THE SPIKE-RATE GATE (R-28 step (2)): the judgement in BOTH directions, the verdict it
      // mints, and -- the half that makes it durable rather than decorative -- that entry's WRITE and
      // READ as ONE pair (the two are indivisible: a reason nothing can emit is prose, and an emitted
      // line nothing can parse is a cache that can never load).
      //
      // BOTH DIRECTIONS, and that is the whole reason this case exists: the ruling's claim is "the UPPER
      // bound decides", and the only way to execute it is to point the SAME judgement at two readings
      // that differ only in the rate -- the calm one must NOT reject, the harsh one MUST.  A one-sided
      // test would pass for a function that always rejects (which would refuse every machine, forever)
      // and for one that never does (which would ship rows from a machine whose spikes the algorithm
      // cannot survive).  The numbers carry margin because a self-test that lives at the boundary is a
      // coin flip, not a test.
      {
        const SpikeGateVerdict calm  = judgeSpikeRate(4u, 100u, 6u);
        const SpikeGateVerdict harsh = judgeSpikeRate(25u, 100u, 6u);
        // `intervalCovers` is the PAIRED machinery's, so the rate interval's own containment is asserted
        // here directly: the gate compares a RATE with the bound, and an interval that did not contain
        // the rate it was built from would be comparing a different quantity (R-3(a)'s "same definition"
        // discipline, one level down).
        const bool calmInside  = calm.interval.ok && calm.interval.low <= calm.interval.p
                              && calm.interval.p <= calm.interval.high;
        const bool harshInside = harsh.interval.ok && harsh.interval.low <= harsh.interval.p
                              && harsh.interval.p <= harsh.interval.high;
        // `blocks == 0`: no denominator is not "a calm machine".  Fail-closed means reject.
        const SpikeGateVerdict noDenominator = judgeSpikeRate(0u, 0u, 6u);
        if (!calmInside || !harshInside || calm.reject || !harsh.reject || !noDenominator.reject
            || calm.interval.high > TuneStats::SPIKE_RATE_GATE
            || harsh.interval.high <= TuneStats::SPIKE_RATE_GATE) {
          ok = false;
          if (firstBad.empty()) {
            firstBad = string("(k) the spike gate is not the fail-closed comparison the ruling "
                              "describes: calm(4/100) reject=") + (calm.reject ? "YES (BUG)" : "no")
                     + " upper=" + to_string(calm.interval.high)
                     + " inside=" + (calmInside ? "yes" : "NO (BUG)")
                     + ", harsh(25/100) reject=" + (harsh.reject ? "yes" : "NO (BUG)")
                     + " upper=" + to_string(harsh.interval.high)
                     + " inside=" + (harshInside ? "yes" : "NO (BUG)")
                     + ", 0-blocks reject=" + (noDenominator.reject ? "yes" : "NO (BUG)")
                     + " (bound " + to_string(TuneStats::SPIKE_RATE_GATE) + ")";
          }
        }
      }
      // The verdict, built by the REAL gate on a decision that HAS a row set, so the "no row set"
      // clause is executed rather than asserted: after the call the rows must be GONE.
      {
        Decision g;
        g.deviceName = "self-test";
        g.cachePath = probePath;
        g.nnXLen = 19;
        g.nnYLen = 19;
        g.maxBatchSize = 8;
        g.fingerprint = "self-test";
        g.policyFingerprint = "self-test";
        g.baselineVersion = "1";
        g.modelIdentity = "self-test";
        aMeasuredSet(g);
        vk_tune_row_spec r {};
        r.path = VK_TILE_PATH_SCALAR;
        r.src0_type = 1;
        r.src1_type = 0;
        r.geom.BM = 64; r.geom.BN = 64; r.geom.BK = 0;
        r.geom.WM = 32; r.geom.WN = 32; r.geom.WMITER = 1;
        r.geom.TM = 4; r.geom.TN = 4; r.geom.BS = 128;
        r.geom.pad = -1; r.geom.reshape = -1; r.geom.smaller = -1;
        g.rows.push_back(r);
        rejectBySpikeGate(g, judgeSpikeRate(25u, 100u, 6u));
        const bool verdictOk0 = g.outcome == "gate-reject" && g.negative
                             && g.negativeReason == "gate-reject"
                             && g.negativeOrigin == Decision::NegativeOrigin::SpikeGate
                             && g.negativeScope.size() == 1 && g.negativeScope[0] == "mm"
                             && g.rows.empty();
        // The entry must be WRITABLE and it must be the only thing the file says (ROWS 0 + the marker,
        // never both for one kind -- the formatter throws on that, which is a different case's subject).
        const bool writableOk0 = outcomeWritable(g) && cacheEntryWritable(g);
        // 🔴 READ THE PRODUCT (section 8 trap 12): "the verdict is right" is not "the file is right".
        // The formatter is the authority on what would reach the disk, so its OUTPUT is what is read.
        string emitted0, emitErr;
        try {
          emitted0 = formatRowsFileImpl(g);
        }
        catch (const std::exception & e) {
          emitErr = e.what();
        }
        const bool emitsOk = emitted0.find("\nROWS 0\n") != string::npos
                          && emitted0.find("\nNEGATIVE gate-reject mm\n") != string::npos
                          && emitted0.find("\nROW ") == string::npos;
        // ... and the READ half of the pair, through the REAL parser: the file this process would write
        // must be one this process can load.  The wave is synthesized (there is no device here), and it
        // cannot matter: the fixture carries no ROW line.
        RowsFile back;
        string backErr;
        const bool readOk = parseRowsFile(emitted0, vk_tile_wave_for(32), back, backErr)
                         && back.negative && back.negativeReason == "gate-reject"
                         && back.negativeScope.size() == 1 && back.negativeScope[0] == "mm"
                         && back.rows.empty();
        if (!verdictOk0 || !writableOk0 || !emitsOk || !readOk) {
          ok = false;
          if (firstBad.empty()) {
            firstBad = string("(k) the gate's durable entry is not the one the format carries: "
                              "verdict=") + (verdictOk0 ? "ok" : "BAD")
                     + " writable=" + (writableOk0 ? "ok" : "BAD")
                     + " emitted=" + (emitsOk ? "ok" : ("BAD (" + emitted0.substr(0, 120) + ")"))
                     + " read-back=" + (readOk ? "ok" : ("BAD (" + backErr + ")"))
                     + (emitErr.empty() ? "" : (" formatter refused: " + emitErr));
          }
        }
        // 🔴 THE EMPTY-INSTALL HALF OF THE ORDERING QUESTION, now EXECUTED instead of registered
        // (R-28 step (3) tightened this note, and the half that has a carrier must not stay a sentence):
        // the finalizer recomputes the outcome from the INSTALL accounting, so on a decision that
        // installed nothing it must leave a `SpikeGate`-origin verdict alone -- retitling it, or dropping
        // the marker, would turn "the measurement was too spiky to trust" into a claim about the device.
        // The OTHER half (a row installing AFTER a gate rejection) still has no carrier and stays
        // registered for ⑤.
        const Decision::FinalizeVerdict vGate = finalizeOutcome(g);
        if (vGate != Decision::FinalizeVerdict::Nothing || g.outcome != "gate-reject" || !g.negative
            || g.negativeOrigin != Decision::NegativeOrigin::SpikeGate) {
          ok = false;
          if (firstBad.empty()) {
            firstBad = string("(k) the finalizer did not leave the gate's verdict alone on an "
                              "empty-install decision: verdict ") + verdictName(vGate) + ", outcome '"
                     + g.outcome + "', marker " + (g.negative ? "set" : "GONE (BUG)");
          }
        }
      }
    }

    { // (l) THE THIRD BRANCH (R-28 step (3)): the drop accounting, and the THREE directions the branch
      // has to get right -- including the one that would be a disaster read the other way (an incomplete
      // pair set must NOT veto a row set).
      //
      // A synthetic result, not a device run: the accounting is a copy plus one branch, and the numbers
      // that matter (which pairs were dropped, how many candidates were enumerated) are the CALLER's
      // facts.
      {
        auto makeResult = [](uint32_t objs, bool complete, uint32_t buildFailed) {
          vk_tune_probe_result r {};
          r.st = VK_TUNE_PROBE_OK;
          r.shapes = 3;
          r.objects = objs;
          r.pairs_legal = 12;
          r.pairs_illegal = 2;
          r.dropped_build_failed = buildFailed;
          r.pairs_complete = complete;
          return r;
        };
        // (A) incomplete + no winner -> the branch FIRES, and its outcome is `no-winner` (declared and
        // unwritable), so nothing can be remembered from a comparison that could not be finished.
        Decision a;
        a.deviceName = "self-test";
        // The measured-candidate count is the CALLER's statement (the object count includes the two
        // arms), which is why it is passed explicitly rather than read off `r.objects`.
        noteMeasurementAccounting(a, makeResult(6u, false, 2u), 9u, 6u);
        const bool accountedOk = !a.pairsComplete && a.droppedBuildFailed == 2u
                              && a.candidatesMeasured == 6u && a.candidatesEnumerated == 9u
                              && a.pairsLegal == 12u && a.shapesCount == 3u;
        const bool firedA = noGainSuppressedOnIncompleteSet(a);
        // ... and the FINALIZER must leave that verdict alone on an empty-install decision, which is the
        // ordering claim this case can execute (the protocol half is registered, not claimed).
        const Decision::FinalizeVerdict vA = finalizeOutcome(a);
        const bool finalizerOk = vA == Decision::FinalizeVerdict::Nothing && a.outcome == "no-winner"
                              && !outcomeWritable(a) && !cacheEntryWritable(a);
        // (B) COMPLETE + no winner -> NOT this branch.  Branch ② (a complete measurement with no winner)
        // is not in this build, so the branch must leave the decision exactly as it found it instead of
        // re-labelling it: an outcome nothing produces would be the "name before producer" drift this
        // project refuses for outcomes (unlike reasons, which may be reserved).
        Decision b;
        b.deviceName = "self-test";
        noteMeasurementAccounting(b, makeResult(6u, true, 0u), 9u, 6u);
        const bool firedB = noGainSuppressedOnIncompleteSet(b);
        const bool untouchedB = (b.outcome == "running");
        // (C) incomplete + A WINNER -> the ROW SET STANDS.  This is the direction section 2.6(a) calls
        // out: what was dropped was an ALTERNATIVE, so one dropped candidate must not void a safe row.
        Decision c;
        c.deviceName = "self-test";
        c.install.declared = 1;
        c.install.installed = 1;
        c.rows.push_back(vk_tune_row_spec {});
        c.rows[0].path = VK_TILE_PATH_SCALAR;
        c.rows[0].src0_type = 1;
        c.rows[0].src1_type = 0;
        c.rows[0].geom.BM = 64; c.rows[0].geom.BN = 64; c.rows[0].geom.BK = 0;
        c.rows[0].geom.WM = 32; c.rows[0].geom.WN = 32; c.rows[0].geom.WMITER = 1;
        c.rows[0].geom.TM = 4;  c.rows[0].geom.TN = 4;  c.rows[0].geom.BS = 128;
        c.rows[0].geom.pad = -1; c.rows[0].geom.reshape = -1; c.rows[0].geom.smaller = -1;
        c.rowInstalledSeen = true;
        c.rowInstalled.push_back(1);
        noteMeasurementAccounting(c, makeResult(6u, false, 0u), 9u, 6u);
        c.outcome = "measured";
        const bool firedC = noGainSuppressedOnIncompleteSet(c);
        const bool rowsStand = (c.outcome == "measured") && !rowsForWrite(c).empty();
        // (D) the DECISION LINE is the constant evidence for the two denominators (section 2.6: the hint
        // only exists when a row set is written, the line always does), so the accounted numbers must be
        // READABLE off the line this process would print.
        const string dl = formatDecisionLine(a, 19, 19, 8);
        const bool lineOk = dl.find(" cands=6/9 pairs=12/3") != string::npos;
        if (!accountedOk || !firedA || !finalizerOk || firedB || !untouchedB || firedC || !rowsStand
            || !lineOk) {
          ok = false;
          if (firstBad.empty()) {
            firstBad = string("(l) the third branch is not the decision section 2.6(a) describes: "
                              "accounting=") + (accountedOk ? "ok" : "BAD")
                     + " (A) fired=" + (firedA ? "yes" : "NO (BUG)")
                     + " finalizer=" + (finalizerOk ? "ok" : "BAD")
                     + " outcome='" + a.outcome + "'"
                     + " (B) fired=" + (firedB ? "YES (BUG: branch 2 is not in this build)" : "no")
                     + " untouched=" + (untouchedB ? "yes" : ("NO (BUG: outcome '" + b.outcome + "')"))
                     + " (C) fired=" + (firedC ? "YES (BUG: an incomplete set vetoed a row set)" : "no")
                     + " rows-stand=" + (rowsStand ? "yes" : "NO (BUG)")
                     + " decision-line=" + (lineOk ? "ok" : ("BAD (" + dl + ")"));
          }
        }
      }
    }


    // ── (n) THE POSITIVE WRITE-PATH LEG (step 4's A10b fixture): a decision that passes every gate
    // LANDS a file, and the file it lands is one this process reads back ─────────────────────────────
    //
    // WHY THIS CASE EXISTS.  Until it, NO leg had ever written through the GATED writer: the (a)-(e)
    // loop stops every state-writable row at the read-only policy ON PURPOSE (so that the test itself
    // would not create the probe file), and r6's encoding probe deliberately goes AROUND the gate
    // because its subject is how a PATH is encoded.  So "the cache write path can write" was a claim
    // about code that nothing executed -- and it is the last unproven link of the chain that ends in a
    // durable row set.
    //
    // WHAT IT IS NOT.  It is evidence about no measurement, no candidate and no device: the row below is
    // SYNTHESIZED (this function has no backend, no pipeline and no primitive to call), and the decision
    // is hand-filled the way (j)/(k) fill theirs.  It proves the PATH -- the gates, the formatter, the
    // atomic write, the parser -- and nothing about whether the numbers that would feed it are any good.
    // That division is why it is worth having: the claims that DO need a machine in the right mood (A3's
    // boundary gate, branch (1)'s confirmation) get a carrier whose reading does not move between
    // machines.
    //
    // ONE VARIABLE.  Every gate is satisfied, so the only thing left that can stop the write is the
    // write path itself.  `aMeasuredSet`/`aSweptSet` are the same two helpers (d) and (j) use.  The
    // install mask is deliberately NOT set, because (g)/(h) own the fold and a fixture that set both
    // would make an install-fold bug read as a write-path bug.  The one precondition it INHERITS rather
    // than restates: a write-ALLOWED policy, which the (a)-(e) loop already requires -- its unwritable
    // direction asserts that the STATE gate, not the mode, is what refused.
    //
    // 🔴 WRITING A REAL FILE BRINGS TWO DUTIES, and both are asserted here rather than promised: it must
    // use its OWN path (reusing the first probe's path would dismantle that probe's "still absent"
    // verdict), and it must REMOVE the file again, because the line below is read by a smoke leg that
    // requires no `self-test*` entry to survive the run.  The name carries the pid for r6's reason (two
    // NN server threads must not collide), and it reuses r6's pid: what has to be unique is the NAME,
    // not the call.
    const string caseNPath = cacheDir + "/self-test-" + to_string(encProbePid) + "-case-n.txt";
    // The verdict, SPLIT BY CLAUSE rather than folded into one bool: a failure has to name which half of
    // "the file is right" broke (its legality, the write, the read, the key, the rows, the cleanup),
    // because those send a reader to six different places.
    bool caseNWasAbsent = false, caseNRowLegal = false, caseNWrote = false;
    bool caseNRead = false, caseNKey = false, caseNRows = false, caseNRemoved = false;
    string caseNDetail;
    {
      Decision d;
      d.deviceName = "self-test";
      d.cachePath = caseNPath;
      d.nnXLen = 19;
      d.nnYLen = 19;
      d.maxBatchSize = 8;
      d.fingerprint = "self-test";
      // The REAL policy string, not a literal: POLICY is what the loader compares against the running
      // policy, so a fixture that invented one would only prove that the write path can land a file no
      // run could ever load.
      d.policyFingerprint = policyFingerprint();
      d.baselineVersion = "1";
      d.modelIdentity = "self-test";
      d.modeString = modeString(enabled_t::True);
      d.outcome = "measured";
      aMeasuredSet(d);
      aSweptSet(d);
      // The (a)-(e) loop's `completed` row, and the wave it is legal under is SYNTHESIZED here for the
      // same reason it is there: there is no device in this function.  Its legality is asserted rather
      // than assumed -- a fixture that wrote a row the catalogue could never produce would prove the
      // file format, not the write path -- and that same synthesized wave is what the read-back must be
      // parsed with, because the parser re-checks the row against it.
      vk_tune_row_spec r {};
      r.path = VK_TILE_PATH_SCALAR;
      r.src0_type = 1;
      r.src1_type = 0;
      r.geom.BM = 64; r.geom.BN = 64; r.geom.BK = 0;
      r.geom.WM = 32; r.geom.WN = 32; r.geom.WMITER = 1;
      r.geom.TM = 4;  r.geom.TN = 4;  r.geom.BS = 128;
      r.geom.pad = -1; r.geom.reshape = -1; r.geom.smaller = -1;
      d.rows.push_back(r);
      caseNRowLegal = vk_tile_candidate_reason(r.path, r.geom, vk_tile_wave_for(32)) == nullptr;
      caseNWasAbsent = !FileUtils::exists(caseNPath);
      if (!caseNRowLegal) {
        caseNDetail = "the fixture's own row is not one the catalogue could produce (illegal geometry)";
      }
      else if (!caseNWasAbsent) {
        // The name carries the pid, so this can only be a file an earlier run of THIS process left
        // behind -- a real finding, not a precondition to skip: a fixture that reused a file would read
        // as "the writer created it" while having proved nothing.
        caseNDetail = "the probe file already existed before the write (" + caseNPath + ")";
      }
      else {
        // THE CACHE WRITE PATH, through the REAL gated writer, and the FIRST call in this function
        // with a policy that allows writing -- which is what makes this case the executor of it.
        string err;
        caseNWrote = writeCacheIfAllowed(d, policy, err);
        if (!caseNWrote) {
          caseNDetail = "the write path refused a decision that satisfies every gate: " + err;
        }
      }
      if (caseNWrote) {
        // ── THE READ-BACK, through the pair the PRODUCTION loader uses (`readWholeFile` +
        // `parseRowsFile`): a file this process can write and cannot read back is a cache that can never
        // load, and that is the half which makes the write durable rather than decorative.
        string text, rdErr;
        RowsFile back;
        caseNRead = readWholeFile(caseNPath, text, rdErr)
                 && parseRowsFile(text, vk_tile_wave_for(32), back, rdErr);
        if (!caseNRead) {
          caseNDetail = "the file just written did not read back: " + rdErr;
        }
        else {
          // (1) THE KEY.  These are the comparisons the loader makes before it will believe a file,
          // quoted in the loader's own order: a fixture that checked fewer of them would go green on a
          // file the loader refuses.  `absentKeys` is the one that catches a header the writer forgot --
          // the loader REFUSES a file carrying any -- so "the writer emits every key it declares" is
          // executed here rather than asserted.
          caseNKey = back.foundVersion && back.version == SCHEMA_VERSION
                  && back.foundTunerVersion && back.tunerVersion == TUNER_VERSION
                  && back.foundFingerprint && back.fingerprint == d.fingerprint
                  && back.foundPolicy && back.policy == d.policyFingerprint
                  && back.foundBaseline && back.baseline == d.baselineVersion
                  && back.foundModel && back.model == d.modelIdentity
                  && back.foundBoard && back.boardX == d.nnXLen && back.boardY == d.nnYLen
                  && back.foundMaxBatch && back.maxBatch == d.maxBatchSize
                  && back.foundMode && back.mode == d.modeString
                  && back.foundCreated
                  && back.absentKeys.empty()
                  && !back.negative && back.negativeScope.empty();
          // (2) THE ROWS, field by field, against the DURABLE set the writer announced -- plus the one
          // link that makes the file describe ITSELF: the declared ROWSET_HASH must equal the hash of
          // the rows that came back, which is exactly the check the loader performs.
          const vector<vk_tune_row_spec> want = rowsForWrite(d);
          caseNRows = back.foundDeclaredRows && back.declaredRows == (long) want.size()
                   && back.rows.size() == want.size()
                   && back.foundRowSetHash && back.declaredRowSetHash == rowSetHash(back.rows);
          for (size_t i = 0; caseNRows && i < want.size(); i++) {
            const vk_tune_row_spec & got = back.rows[i];
            const vk_tune_row_spec & exp = want[i];
            caseNRows = got.path == exp.path && got.src0_type == exp.src0_type
                     && got.src1_type == exp.src1_type && got.key_m == exp.key_m
                     && got.key_k == exp.key_k
                     && got.geom.BM == exp.geom.BM && got.geom.BN == exp.geom.BN
                     && got.geom.BK == exp.geom.BK && got.geom.WM == exp.geom.WM
                     && got.geom.WN == exp.geom.WN && got.geom.WMITER == exp.geom.WMITER
                     && got.geom.TM == exp.geom.TM && got.geom.TN == exp.geom.TN
                     && got.geom.BS == exp.geom.BS && got.geom.pad == exp.geom.pad
                     && got.geom.reshape == exp.geom.reshape && got.geom.smaller == exp.geom.smaller;
          }
          if (!caseNKey) {
            caseNDetail = "the file does not carry the key its decision was written under";
          }
          else if (!caseNRows) {
            caseNDetail = "the file does not carry the row set the writer announced";
          }
        }
        // The cleanup is attempted even when the read-back failed, and LAST: leaving the file behind
        // would make the NEXT run report "the probe file already existed" instead of this failure, i.e.
        // the second run would name the wrong cause.
        caseNRemoved = FileUtils::tryRemoveFile(caseNPath);
        if (!caseNRemoved && caseNDetail.empty()) {
          caseNDetail = "the probe file could not be removed (" + caseNPath + ")";
        }
      }
    }
    const bool caseNOk = caseNRowLegal && caseNWasAbsent && caseNWrote && caseNRead
                      && caseNKey && caseNRows && caseNRemoved;
    if (!caseNOk) {
      if (caseNDetail.empty()) {
        // Fail-closed: a clause that failed with no sentence would leave the line saying "FAIL: " and
        // nothing else, which is the "error string that does not name the clause" failure (see (g)/(h)).
        caseNDetail = "a clause failed without naming itself (the read-back verdict is incomplete)";
      }
      ok = false;
      if (firstBad.empty()) {
        firstBad = string("(n) the write path did not land the row set it was allowed to write: ")
                 + caseNDetail + " [fixture-legal=" + (caseNRowLegal ? "yes" : "NO (BUG)")
                 + " path-absent=" + (caseNWasAbsent ? "yes" : "NO")
                 + " wrote=" + (caseNWrote ? "yes" : "no")
                 + " read=" + (caseNRead ? "yes" : "no")
                 + " key=" + (caseNKey ? "ok" : "BAD")
                 + " rows=" + (caseNRows ? "ok" : "BAD")
                 + " removed=" + (caseNRemoved ? "yes" : "no") + "]";
      }
    }

    // ── R-29 Cut 5: THE MOVED `A1_export` ASSERTION, READ FROM THE REAL FORMATTER ─────────────────
    // The `A1_export` Check voted on the header of the file the EXPORT channel wrote, and the export
    // channel is gone (Cut 1).  Its POLICY half -- "the header the product writes carries all 8 declared
    // constants, in order" -- must keep a carrier: it is the mechanical witness that a gate parameter
    // cannot leave policyFingerprint unnoticed.  The carrier is the SAME function the cache write path
    // calls (formatRowsFileImpl), driven with the REAL policy string and one LEGAL row.  The row is not
    // decoration: the formatter refuses a decision carrying neither rows nor a marker, which is the rule
    // A8c/A10 assert, so a row-less probe would throw instead of proving anything.
    // The emitted `POLICY` row is reported VERBATIM so the smoke can hold the declared constants as an
    // INDEPENDENT spelling -- a check that compares the product with itself cannot fail (R47's comment
    // names that disease), and the second spelling is the same discipline rowsethash.py follows for
    // ROWSET_HASH.
    {
      Decision pd;
      pd.policyFingerprint = policyFingerprint();
      pd.baselineVersion = "1";
      pd.modelIdentity = "self-test";
      aMeasuredSet(pd);
      vk_tune_row_spec pr {};
      pr.path = VK_TILE_PATH_SCALAR;
      pr.src0_type = 1;
      pr.src1_type = 0;
      pr.geom.BM = 64;   pr.geom.BN = 64;      pr.geom.BK = 0;
      pr.geom.WM = 32;   pr.geom.WN = 32;      pr.geom.WMITER = 1;
      pr.geom.TM = 4;    pr.geom.TN = 4;       pr.geom.BS = 128;
      pr.geom.pad = -1;  pr.geom.reshape = -1; pr.geom.smaller = -1;
      pd.rows.push_back(pr);
      // The marker is not a shortcut, it is the shape this file can WRITE: the formatter emits the
      // durable per-row set (rowsForWrite), and a synthetic decision installs no row anywhere, so a
      // row-only probe would come out EMPTY and the formatter refuses "neither rows nor a marker".
      // The (k) case above uses exactly this marker shape and its output is already read back -- but the
      // probe takes the MARKER only: it is not a gate event (it measured nothing), and calling the
      // logging wrapper here emitted a third spike-rate NOTICE into the line whose count the smoke pins
      // at two, with an empty device name, because the synthetic decision has none (measured).
      setGateRejectMarker(pd);
      string emittedPolicy;
      try {
        emittedPolicy = formatRowsFileImpl(pd);
      }
      catch (const std::exception & e) {
        policyRowText = string("FAIL: the formatter threw: ") + e.what();
      }
      // "Not decided yet" is EMPTY, not "does not start with FAIL:" -- the sentinel used to
      // satisfy its own guard, which is the defect this cut removes.
      if (policyRowText.empty()) {
        const size_t at = emittedPolicy.find("\nPOLICY ");
        if (at == string::npos) {
          policyRowText = "FAIL: the emitted file carries no POLICY row";
        } else {
          const size_t eol = emittedPolicy.find('\n', at + 1);
          policyRowText = emittedPolicy.substr(at + 1, (eol == string::npos ? emittedPolicy.size() : eol) - at - 1);
          policyRowOk = policyRowText.compare(0, 7 + pd.policyFingerprint.size(),
                                              "POLICY " + pd.policyFingerprint) == 0;
          if (!policyRowOk) {
            policyRowText = "FAIL: the emitted POLICY row is not the string policyFingerprint() built: " + policyRowText;
          }
        }
      }
    }
    if (!policyRowOk) {
      ok = false;
      // A probe that neither threw nor produced a row never ran: saying so is the honest
      // reading, and it is said HERE because this is where the verdict is built.
      if (policyRowText.empty()) { policyRowText = "FAIL: the POLICY-row probe did not run"; }
      if (firstBad.empty()) { firstBad = string("(m) the POLICY row the formatter emits: ") + policyRowText; }
    }

    const bool absent = !FileUtils::exists(probePath);    logLine("cache write self-test: " + to_string(declared) + " outcome(s) declared, "
            + to_string(refused) + " refused by the state gate, " + to_string(allowed)
            + " state-writable; probe file " + (absent ? "absent" : "PRESENT")
            + "; encoding probe (non-ASCII name) written+read-back+removed ["
            + ((encWrote && encReadBack && encRemoved) ? "ok" : "FAIL") + "]"
            + "; (a)(b) no marker / (c) from-file retirement / (d) measured control / (e) our-gap / (f) all-reject->measured retracts / (g)(h) fold AND+mismatch / (i) unmeasured marker refused by name / (j) the formatter's throw is contained / (k) spike gate: both directions + no-row-set + the durable entry's write/read pair / (l) third branch: drop accounting + the three directions + the finalizer leaves it alone ["
            + (verdictOk ? "ok" : "FAIL: " + verdictBad) + "]"
            + "; (n) the positive write-path leg (the real writer lands the row set it was allowed to write, the real reader reads it back, and the probe is removed) ["
            + (caseNOk ? "ok" : ("FAIL: " + caseNDetail)) + "]"
            + "; POLICY row [" + policyRowText + "]"
            + " [" + ((ok && caseNOk && absent && encWrote && encReadBack && encRemoved && verdictOk && policyRowOk) ? "ok" : "FAIL") + "]"
            + (ok ? "" : " first problem: " + firstBad));
  }

  // -----------------------------------------------------------------------------------------------
  // W4e 2c-c: the MEASUREMENT primitive's self-test leg (w4e-preview.md R-14 / R-15 / R-16).
  //
  // CARRIER (R-14): bit 16 of the EXISTING KATAGO_VK_T2_PROBE -- no new env, because R14's
  // `MANAGED_WANT = 41` and the smoke's `$managed` (41 names) are a COUNT EQUALITY and a new name would
  // have to move both.  It cannot ride bit 8 with the other self-tests, and that is a property of the
  // object under test rather than a preference: the primitive REFUSES to measure while bit 4 or bit 8
  // is on (its own shapes would be mixed into the user's tables), so a leg that asks it to measure must
  // use a bit that is neither.  `all` therefore stays 0xF and does NOT include this one: a full-probe
  // run would otherwise have every measurement refused by the primitive's own interference gate.
  //
  // WHAT THIS LEG VERIFIES -- written out because a leg that only says "ok" is read as "verified" for
  // every neighbouring claim (R-15), and because 2c-c is the cut that made the second half reachable:
  //
  //   1. THE TWO SHARED DEFINITIONS, on the CPU, with no device.
  //      (a) The rotation: balanced inside a cycle (each index occupies each position exactly once),
  //          reproducible from its seed, and actually dependent on it.
  //      (b) The spike detector, driven over a synthetic arm series with a KNOWN injected rate through
  //          the very function the probe calls, returns an interval that CONTAINS that rate.  Without
  //          this, "the rate gate was measured with the detector's band" would be a claim about two
  //          definitions that merely look alike (R-3(a), R-4(i)).
  //   2. THE REFUSALS 2c-c made reachable: shots == 0, reps < 2, warmupBlocks < 2, stallBlocks == 0, a
  //      band that is not an interval, and a sample buffer too small for the grid.  Every one of them
  //      must be refused BEFORE any device work, so the whole battery costs nothing.
  //   3. THE DEVICE RUN, on a SYNTHETIC single-family fixture (R-16: a row this file builds, whose
  //      family and accumulator match the shape's -- NOT an enumerated candidate, because a real one
  //      would make R-13 a hard prerequisite of this batch): the pipeline override is CONSUMED (the
  //      candidate blocks report hits > 0), the ANCHOR blocks report 0, the guard RESTORES the field,
  //      blocks and samples are counted, the grid's paired samples are all finite while the anchor
  //      slots stay NaN, and the summary reports pairs_complete.
  //
  //   4. WHAT IT DOES NOT VERIFY.  The gate comparison itself (interval upper bound vs 0.15), the
  //      interval-vs-tolerance decisions, `coverage_complete`, and whether the numbers are any GOOD --
  //      that last one needs the calibration window, not a self-test.  The gate
  //      is the shell's decision path and the numbers are the calibration batch's business (R-10).
  //      The fixture is synthetic and the line says so: it exercises the (rowIndex, aligned) memo and
  //      the A/A sentinel exactly as a real candidate would, and it is evidence about NO catalogue row.
  void probeSelfTest(ggml_backend_t backend) {
    int checks = 0;
    int failed = 0;
    string firstBad;
    const auto want = [&](bool cond, const char * what) {
      checks++;
      if (!cond) {
        failed++;
        if (firstBad.empty()) {
          firstBad = what;
        }
      }
    };

    // ---- 1(a) the rotation ----------------------------------------------------------------------
    bool balanced = true;
    bool reproducible = true;
    bool seedMatters = false;
    {
      const uint32_t n = 5;
      std::vector<uint32_t> perm(n, 0);
      for (uint32_t cycle = 0; cycle < 3; cycle++) {
        vk_tune_rotation_permutation(TuneStats::PROBE_ROTATION_SEED, n, cycle, perm.data());
        std::vector<uint32_t> again(n, 0);
        vk_tune_rotation_permutation(TuneStats::PROBE_ROTATION_SEED, n, cycle, again.data());
        if (again != perm) {
          reproducible = false;
        }
        for (uint32_t phase = 0; phase < n; phase++) {
          std::vector<int> seen(n, 0);
          for (uint32_t k = 0; k < n; k++) {
            seen[vk_tune_rotation_at(perm.data(), n, phase, k)]++;
          }
          for (uint32_t i = 0; i < n; i++) {
            if (seen[i] != 1) {
              balanced = false;
            }
          }
        }
      }
      std::vector<uint32_t> base0(n, 0);
      std::vector<uint32_t> other(n, 0);
      vk_tune_rotation_permutation(TuneStats::PROBE_ROTATION_SEED, n, 0, base0.data());
      vk_tune_rotation_permutation(TuneStats::PROBE_ROTATION_SEED ^ 0x9E3779B97F4A7C15ULL, n, 0, other.data());
      seedMatters = (base0 != other);
    }
    want(balanced, "the rotation is not balanced inside a cycle");
    want(reproducible, "the rotation is not reproducible from its seed");
    want(seedMatters, "two different seeds produced the same rotation");

    // ---- 1(b) the detector, over a series whose rate we INJECTED ---------------------------------
    const long detTrials = 4000;
    const double detInjected = 0.106;   // the measured regime whose band the 0.15 gate came from
    uint32_t detMarks = 0;
    double detMeasured = 0.0;
    double detLow = 0.0;
    double detHigh = 0.0;
    bool detContains = false;
    {
      // The generator's OWN arm (one object's block times), so the series this detector is asked about
      // is the series the envelope is built from -- not a re-invention of it with the same constants.
      const std::vector<double> arm = TuneStats::spikeSyntheticArm(detTrials, 1.0, 0.001, detInjected,
                                                                   TuneStats::SPIKE_BAND_LO, TuneStats::SPIKE_BAND_HI,
                                                                   TuneStats::PROBE_ROTATION_SEED);
      std::vector<double> sorted = arm;
      std::sort(sorted.begin(), sorted.end());
      const double med = sorted.empty() ? 0.0 : sorted[sorted.size() / 2];
      for (size_t i = 0; i < arm.size(); i++) {
        if (vk_tune_spike_band_marked(arm[i], med, TuneStats::SPIKE_BAND_LO, TuneStats::SPIKE_BAND_HI)) {
          detMarks++;
        }
      }
      const TuneStats::RateInterval ri = TuneStats::rateInterval((long) detMarks, detTrials, TuneStats::ALPHA);
      detMeasured = ri.p;
      detLow = ri.low;
      detHigh = ri.high;
      detContains = ri.ok && ri.low <= detInjected && ri.high >= detInjected;
    }
    want(detContains, "the detector's interval does not contain the injected spike rate: the runtime "
                      "detector and the envelope generator's band are not the same quantity");

    // ---- 2 the refusals, none of which may touch the device ---------------------------------------
    vk_tune_probe_shape shape {};
    shape.m = 192;
    // n must clear `mul_mat_vec_max_cols` (8): a mul_mat with n <= 8 is routed to the dmmv/mat-vec path
    // (ggml-vulkan.cpp:564/2538), which never asks the matmul pipeline resolver at all -- so no tuned row
    // and no probe override can reach it, and the primitive's own "override was not consumed" refusal is
    // the fail-closed answer rather than a bug.  MEASURED (the first run of this leg, n = 8): the
    // candidate's blocks reported override_hits = 0 and the probe refused exactly there.  That is a fact
    // about which shapes a row can serve, not about this fixture: the covered set cannot contain a shape
    // whose dispatch never consults the row table.
    shape.n = 64;
    shape.k = 512;
    shape.src0_type = VK_TUNE_SRC_F16;
    shape.src1_type = VK_TUNE_SRC_F16;
    vk_tune_probe_object refObjects[2];
    for (int i = 0; i < 2; i++) {
      refObjects[i].stock = true;      // two entries pointing at the same pipeline: the A/A sentinel
      refObjects[i].rowIndex = 0;
      refObjects[i].aligned = false;
    }
    vk_tune_row_spec refCand {};
    refCand.path = VK_TILE_PATH_SCALAR;
    refCand.src0_type = 1;             // f16, the same family as the shape
    refCand.src1_type = 1;
    refCand.key_m = 192;
    refCand.key_k = 512;
    vk_tune_probe_params base {};
    base.shapes = &shape;
    base.nShapes = 1;
    base.objects = refObjects;
    base.nObjects = 2;
    base.shots = 4;
    base.warmupBlocks = 2;
    base.reps = 3;
    base.stallBlocks = 8;
    // The caller owns this fact (its enumerator built the set); the primitive only carries it.
    base.coverageComplete = true;
    // Per-shape shots, exercised END TO END with ONE shape carrying the same value: the device
    // fixture's grid (blocks / samples / pairs_legal) is pinned, and a second shape would move all
    // three -- so the multi-shape DIFFERENT-value path is REGISTERED (see not_verified) rather than
    // faked with a fixture whose numbers would then mean something else.
    static const uint32_t baseShotsOne[1] = { 4 };
    base.shotsPerShape = baseShotsOne;
    base.nShotsPerShape = 1;
    base.seed = TuneStats::PROBE_ROTATION_SEED;
    base.spikeBandLo = TuneStats::SPIKE_BAND_LO;
    base.spikeBandHi = TuneStats::SPIKE_BAND_HI;

    int refused = 0;
    int refusalCases = 0;
    bool refusalFlag = false;   // did any REFUSED call report coverage_complete set?
    string refusalBad;
    {
      std::vector<double> dbuf(64, 0.0);
      char rerr[512];
      // Each case names the SENTENCE it must produce, not just the status: every precondition failure in
      // this function returns the same status, so "it was refused" alone would go green while a case
      // tested a different door than it claims.  (Measured: the first version of this battery did exactly
      // that -- the buffer case was answered by the candidate-geometry check.)
      const auto refusedBy = [&](const vk_tune_probe_params & q, size_t cap, const char * sentence) {
        refusalCases++;
        vk_tune_probe_result r {};
        rerr[0] = '\0';
        const bool ok = ::ggml_backend_vk_tune_probe_candidate(backend, &refCand, 1, &q, dbuf.data(), cap,
                                                               &r, rerr, sizeof(rerr));
        const bool good = !ok && r.st == VK_TUNE_PROBE_REFUSED_PRECONDITION && rerr[0] != '\0'
                          && string(rerr).find(sentence) != string::npos;
        if (r.coverage_complete) {
          refusalFlag = true;
        }
        if (good) {
          refused++;
        } else if (refusalBad.empty()) {
          refusalBad = string(sentence) + " -> " + (string(rerr).empty() ? string("(no sentence)") : string(rerr));
        }
        return good;
      };
      vk_tune_probe_params q = base;
      q.shots = 0;         refusedBy(q, dbuf.size(), "shots == 0");
      q = base;
      q.nShotsPerShape = 2;   refusedBy(q, dbuf.size(), "one entry per shape");
      static const uint32_t zeroShot[1] = { 0 };
      q = base;
      q.shotsPerShape = zeroShot; refusedBy(q, dbuf.size(), "is 0: a zero-length");
      q = base;
      q.reps = 1;          refusedBy(q, dbuf.size(), "reps < 2");
      q = base;
      q.warmupBlocks = 1;  refusedBy(q, dbuf.size(), "warmupBlocks < 2");
      q = base;
      q.stallBlocks = 0;   refusedBy(q, dbuf.size(), "stallBlocks is 0");
      q = base;
      q.spikeBandLo = 0.0; refusedBy(q, dbuf.size(), "not a usable amplitude band");
      q = base;

      q = base;
      refusedBy(q, 4, "the sample buffer holds");   // the grid needs 1*2*3 = 6 slots
    }
    if (refused != refusalCases) {
      firstBad = firstBad.empty() ? ("refusal battery: " + refusalBad) : firstBad;
    }
    want(refused == refusalCases, "a refusal that 2c-c made reachable did not refuse with its own name");
    want(!refusalFlag, "a refused probe call reported coverage_complete set: the input was minted into "
                       "an answer where no measurement exists");

    // ---- 3 the device run, on a synthetic single-family fixture -----------------------------------
    bool deviceRan = false;
    bool deviceCoverageEcho = false;
    uint32_t deviceShotsMin = 0;
    uint32_t deviceShotsMax = 0;
    uint32_t deviceBlocks = 0;
    uint32_t deviceSamples = 0;
    uint32_t deviceSpikes = 0;
    uint32_t devicePairsLegal = 0;
    double deviceHot = 0.0;
    bool devicePairsComplete = false;

    bool deviceAnchorNaN = false;
    bool deviceSamplesFinite = true;
    {
      // Two wave variants of ONE synthetic geometry, both f16 x f16 (the shape's family) and both with
      // the routing key the shape carries.  Which of them the device accepts is not this leg's business
      // to re-derive: the rules live in the shared header and the PROBE is their authority here, so the
      // leg tries the wave32 fixture and, if the probe refuses the geometry, the wave64 mirror.  Asking
      // the predicate directly would make this leg a second writer of the wave rule (and a new wave
      // source in the census) for no gain.
      vk_tune_row_spec rows[2];
      for (int i = 0; i < 2; i++) {
        rows[i] = refCand;
      }
      rows[0].geom.BM = 128; rows[0].geom.BN = 128; rows[0].geom.BK = 32;
      rows[0].geom.WM = 32;  rows[0].geom.WN = 64;  rows[0].geom.WMITER = 2;
      rows[0].geom.TM = 2;   rows[0].geom.TN = 2;   rows[0].geom.BS = 256;
      rows[0].geom.pad = 1;  rows[0].geom.reshape = 0; rows[0].geom.smaller = 0;
      rows[1] = rows[0];
      rows[1].geom.WM = 64;  rows[1].geom.WN = 64;  rows[1].geom.WMITER = 4;
      rows[1].geom.TM = 4;   rows[1].geom.TN = 4;

      const uint32_t nObjects = 4;
      vk_tune_probe_object objects[nObjects];
      objects[0].stock = true;  objects[0].rowIndex = 0; objects[0].aligned = false;   // the anchor
      objects[1].stock = false; objects[1].rowIndex = 0; objects[1].aligned = false;   // the candidate
      objects[2].stock = false; objects[2].rowIndex = 0; objects[2].aligned = false;   // its memo twin
      objects[3].stock = true;  objects[3].rowIndex = 0; objects[3].aligned = false;   // A/A sentinel

      vk_tune_probe_params q = base;
      q.objects = objects;
      q.nObjects = nObjects;

      string note;
      // Which geometry built, so the R-13 case below reuses the SAME wave choice instead of becoming
      // a second writer of the wave rule (the two-attempt loop exists to avoid exactly that).
      int workedRow = -1;
      for (int attempt = 0; attempt < 2 && !deviceRan; attempt++) {
        std::vector<double> dsamp((size_t) 1 * nObjects * base.reps, 0.0);
        vk_tune_probe_result r {};
        char derr[1024];
        derr[0] = '\0';
        const bool ok = ::ggml_backend_vk_tune_probe_candidate(backend, &rows[attempt], 1, &q, dsamp.data(),
                                                               dsamp.size(), &r, derr, sizeof(derr));
        if (ok && r.st == VK_TUNE_PROBE_OK) {
          deviceRan = true;
          workedRow = attempt;
          deviceCoverageEcho = r.coverage_complete;
          deviceShotsMin = r.shots_min;
          deviceShotsMax = r.shots_max;
          deviceBlocks = r.blocks;
          deviceSamples = r.samples;
          deviceSpikes = r.spikes;
          devicePairsLegal = r.pairs_legal;
          deviceHot = r.t_block_hot_us;
          devicePairsComplete = r.pairs_complete;
          // The NaN discipline, checked rather than assumed: the anchor's three slots were never
          // written (there is no difference to take against itself), and every other slot IS finite --
          // a NaN among them would poison an interval the caller believes it measured.
          deviceAnchorNaN = std::isnan(dsamp[0]) && std::isnan(dsamp[1]) && std::isnan(dsamp[2]);
          for (size_t i = 3; i < dsamp.size(); i++) {
            if (!std::isfinite(dsamp[i])) {
              deviceSamplesFinite = false;
            }
          }
        } else {
          note = string(derr);
        }
      }

      want(deviceRan, "the probe did not complete a synthetic run on this device (see the note in the line)");
      if (deviceRan) {
        // 1 shape x 4 objects x 3 measured passes, and 3 reps x 3 non-anchor objects.
        want(deviceBlocks == 12, "the probe timed a different number of blocks than the measured grid");
        want(deviceSamples == 9, "the probe wrote a different number of paired samples than 3 x 3");
        want(devicePairsLegal == 4, "the legal-pair count of the synthetic grid changed");
        // With no drop rule left, every legal pair of this fixture is measured: the flag is the plain
        // count equality (`pairs_measured == pairs_legal`), so it must HOLD here (all four pairs are
        // measured, blocks=12 samples=9).
        want(devicePairsComplete,
             "pairs_complete is false on a fixture where every legal pair was measured: the flag no "
             "longer follows the measured/legal count equality");
        want(deviceHot > 0.0, "t_block_hot_us is not a positive median over hot blocks");
        want(deviceSpikes <= deviceBlocks, "more spike marks than measured blocks");
        want(deviceAnchorNaN, "an anchor slot was written instead of staying NaN");
        want(deviceSamplesFinite, "a measured paired sample is not finite");
        want(deviceShotsMin == 4 && deviceShotsMax == 4,
             "per-shape shots were not honoured: the result reports a range the run did not use");
        want(deviceCoverageEcho == base.coverageComplete,
             "the probe did not echo the caller's coverage_complete: the primitive is deriving a fact "
             "that lives in the enumerator");
      }

      // ---- R-13: a candidate that does not target the SHAPE'S OBJECT must refuse the whole call ---
      //
      // The pair is the plan's own example: the shape's tensor says F32 while the candidate declares the
      // (F16,F32) family.  On a coopmat2 device the SHAPE is dispatched as f16 -- that is the
      // `coopmat2 && src1->type == GGML_TYPE_F32` term inside the resolver -- so the shape lives in the
      // (F16,F16) object while the candidate targets the (F16,F32) one.  Without coopmat2 the shape
      // stays in (F16,F32) and the SAME pair is legal, so the outcome is DEVICE-CONDITIONAL: this leg
      // asserts the half this machine can reach and REGISTERS the other (see not_verified) rather than
      // fabricating a device to reach it.
      //
      // The refusal must carry its OWN status (NOT_MEASURED, not a precondition) and its own sentence:
      // a wiring error is not a malformed argument, and "it was refused" alone would also be
      // satisfied by the argument checks above -- the failure mode this battery was built to end.
      if (deviceRan) {
        // The device half comes from the fingerprint the backend already publishes (R1's own interface):
        // a test must not invent an interface to ask a test question.
        char fp[512];
        fp[0] = '\0';
        uint32_t fpsg = 0;
        const bool fpOk = backendFingerprint(backend, fp, sizeof(fp), &fpsg) > 0;
        const bool coopmat2 = fpOk && string(fp).find("cm2=1") != string::npos;

        vk_tune_row_spec r13Cand = rows[workedRow];
        r13Cand.src0_type = VK_TUNE_SRC_F16;
        r13Cand.src1_type = VK_TUNE_SRC_F32;
        vk_tune_probe_shape r13Shape = shape;
        r13Shape.src1_type = VK_TUNE_SRC_F32;
        vk_tune_probe_params q13 = q;
        q13.shapes = &r13Shape;
        vector<double> s13((size_t) 1 * nObjects * base.reps, 0.0);
        vk_tune_probe_result r13res {};
        char r13err[1024];
        r13err[0] = '\0';
        const bool r13ok = ::ggml_backend_vk_tune_probe_candidate(backend, &r13Cand, 1, &q13, s13.data(),
                                                                  s13.size(), &r13res, r13err, sizeof(r13err));
        if (coopmat2) {
          bool allNaN = true;
          for (size_t i = 0; i < s13.size(); i++) {
            if (!std::isnan(s13[i])) {
              allNaN = false;
            }
          }
          want(!r13ok && r13res.st == VK_TUNE_PROBE_NOT_MEASURED
                   && string(r13err).find("does not target the shape's object (R-13)") != string::npos
                   && allNaN,
               "R-13: a candidate whose object is not the shape's object did not refuse the whole call "
               "by its own name (and leave the samples NaN)");
        }
      }
      if (!deviceRan && !note.empty()) {
        firstBad = firstBad.empty() ? ("device run: " + note) : firstBad;
      }
    }

    // ---- the ONE line ----------------------------------------------------------------------------
    // House style, and the boundary clause is part of the LINE rather than only of this comment: the
    // smoke reads the line, so a reader of the evidence sees what was and was not verified without
    // having to open this file.  The per-block evidence (the `probe block:` lines with their
    // override_hits) is emitted by the primitive itself under this bit, and the smoke asserts it there.
    const bool ok = (failed == 0);
    ostringstream os;
    // `fixed`: a default stream prints 71.0 as 7e+01, and the smoke matches on these numbers.
    os << fixed << "probe self-test: rotation[balance=" << (balanced ? "ok" : "FAIL")
       << " reproduce=" << (reproducible ? "ok" : "FAIL")
       << " seed_dependent=" << (seedMatters ? "ok" : "FAIL") << "]"
       << " detector[injected=" << setprecision(4) << detInjected << " marks=" << detMarks
       << " trials=" << detTrials << " rate=" << detMeasured << " ci=[" << detLow << "," << detHigh
       << "] contains_p=" << (detContains ? 1 : 0) << "]"
       << " refusals=" << refused << "/" << refusalCases
       << " device[synthetic=1 shapes=1 objects=4 blocks=" << deviceBlocks << " samples=" << deviceSamples
       << " pairs_legal=" << devicePairsLegal << " pairs_complete=" << (devicePairsComplete ? 1 : 0)
       << " spikes=" << deviceSpikes << " t_block_hot_us=" << setprecision(1) << deviceHot
       << " verifies[override_consumed anchor_hits=0 raii_restore samples+pairs detector_identity "
       << "coverage_complete_echo shots_shape_honoured]"
       << " not_verified[gate_0.15 vs_tolerance coverage_complete number_quality "
       << "r13_without_coopmat2 shots_shape_multi probe_call_window]"
       << " checks=" << checks << " [" << (ok ? "ok" : "FAIL") << "]"
       << (ok ? "" : " first problem: " + firstBad);
    logLine(os.str());
  }

  shared_ptr<Decision> makeDecision(
    ggml_backend_t backend,
    int deviceIdx,
    const string& homeDataDirOverride,
    int nnXLen,
    int nnYLen,
    int maxBatchSize,
    enabled_t useFP16Mode,
    const ModelIdentity& model,
    const TunePolicy& policy,
    const std::vector<vk_tune_probe_shape>& graphShapes,
    const std::vector<uint32_t>& graphShapeCounts
  ) {
    // KATAGO_VK_TUNE_RETUNE=1: ignore whatever is on disk (rows AND the negative entry) and
    // decide again.  It is the operator's "I do not trust this file" switch, and it is
    // deliberately about the whole file rather than about the rows or the negative marker:
    // one file = one key, so a partial override would leave a half-trusted state.
    const bool retune = getenv("KATAGO_VK_TUNE_RETUNE") != nullptr
                     && getenv("KATAGO_VK_TUNE_RETUNE")[0] != '\0'
                     && !(getenv("KATAGO_VK_TUNE_RETUNE")[0] == '0'
                          && getenv("KATAGO_VK_TUNE_RETUNE")[1] == '\0');
    const bool enabled = tuneEnabled();

    if (!enabled) {
      reportIgnoredRequests(true, false);
      return nullptr;
    }
    // cfg vulkanTune=off: read-only.  Still LOUD about the requests that cannot be honoured
    // (RETUNE, PRIME): "I set RETUNE and nothing happened" is the same trap as under the master
    // switch, and this mode is the one a user is most likely to combine with a request.
    reportIgnoredRequests(false, readOnlyMode(policy));

    const auto t0 = chrono::steady_clock::now();
    shared_ptr<Decision> d = make_shared<Decision>();
    d->deviceIdx = deviceIdx;
    d->nnXLen = nnXLen;
    d->nnYLen = nnYLen;
    d->maxBatchSize = maxBatchSize;
    d->modelIdentity = model.headerValue();
    d->policyFingerprint = policyFingerprint();
    // The mode the backend actually received, turned into identity by its ONE producer.  Set HERE,
    // next to the policy, because both are "what was asked for" and both are read by the load
    // chain: from this point on, d->modeString is the only spelling of the mode in this process.
    d->modeString = modeString(useFP16Mode);
    d->baselineVersion = std::to_string(TUNER_POLICY_VERSION);
    // The fail-closed default, and deliberately NOT the answer: two later writers speak about this same
    // fact -- the catalogue ("measured", when no file supplies a set) and the loader below ("cache" when
    // one does, and "measured" again in the three branches that leave this key with no row set).  This
    // line exists so that a decision which reaches NEITHER writer can never claim a source (W3-P1).
    d->rowSource = "none";
    d->tuneModeText = policy.modeText;
    // W2: the resolved policy travels WITH the decision so the once hook can capture it BY VALUE.
    // The hook is assembled on the INSTALL path (that is what makes "callback is empty" mean "this
    // context's install has not run yet") and called from the backend's eval path, which knows
    // nothing about policies.
    d->tunePolicy = policy;
    // (7b) step 4: the coverage set travels WITH the decision, because everything downstream (the
    // pre-check's heaviest shape, the per-shape shots, the candidate rows, `coverageComplete`) is a
    // statement about THIS set.
    d->graphShapes = graphShapes;
    d->graphShapeCounts = graphShapeCounts;
    // ONE producer for the sum: the decision line's `graph_nodes=` and the coverage rule's denominator
    // are then the same arithmetic over the same vector, instead of two numbers that could disagree.
    d->graphNodes = 0;
    for (size_t i = 0; i < graphShapeCounts.size(); i++) {
      d->graphNodes += graphShapeCounts[i];
    }

    try {
      char fp[512];
      // The backend hands back BOTH the identity string and the raw subgroup size in one call, so
      // "the sg inside the string" and "the sg the row path uses" cannot drift (G-C3: the shell has
      // no device handle, so the wave can only be INJECTED, and this call is the seam that already
      // existed).  `sg` stays RAW here: vk_tile_wave_for() is the single conversion and it is called
      // where the wave is used, never here.
      uint32_t sg = 0;
      const size_t fpLen = backendFingerprint(backend, fp, sizeof(fp), &sg);
      if (fpLen == 0) {
        logLine("WARNING: the backend could not describe the device; the tuner is disabled for device "
                + to_string(deviceIdx) + " (fail-open: no tuned rows)");
        return nullptr;
      }
      if (sg == 0) {
        // Fail-closed, in the same direction as the predicate's `wave == 0` clause: a device whose
        // wave we do not know cannot have its geometry judged, and assuming 32 would silently refuse
        // every legal wave64 row (or accept illegal wave32 ones).  Loud, and fail-open (no rows).
        logLine("WARNING: the backend reported subgroup size 0 for device " + to_string(deviceIdx)
                + "; the tuner is disabled for it (a geometry cannot be judged without the wave; "
                  "fail-open: no tuned rows)");
        return nullptr;
      }
      d->deviceSubgroupSize = sg;
      d->fingerprint = string(fp, fpLen);

      // The device name comes out of the fingerprint rather than from a second query: one
      // source, so the log line can never disagree with the identity used for the file.
      {
        const size_t at = d->fingerprint.find("|name=");
        const size_t end = d->fingerprint.find('|', at == string::npos ? 0 : at + 6);
        d->deviceName = at == string::npos
          ? string("unknown")
          : d->fingerprint.substr(at + 6, (end == string::npos ? d->fingerprint.size() : end) - (at + 6));
      }

      // ONE FILE = ONE KEY, IN TWO LEVELS: the PATH is a pure function of the key.  Never a
      // directory listing and never a trial read -- that would trade routing for guessing, and it
      // could pick up another key's file.  The split follows LIFETIMES: the directory holds what
      // IDENTIFIES the cache (device, model version+sha, mode), the file holds the two segments a
      // sweep varies (S and mb), so `benchmark -threads 1,2,4` keeps its three buckets side by
      // side instead of scattering them, and "re-tune" is still a whole-file overwrite rather than
      // a read-modify-write.  Version and policy are deliberately NOT in the path: they are the
      // header's business, and putting them here would orphan every file on every bump (and on
      // every POLICY change, which today is every rebuild).
      // The model NAME is deliberately absent -- it can contain characters a filesystem dislikes,
      // and it is redundant with the sha for routing.
      // The MODE segment is a hash (not the readable string) for the same reason the model segment
      // is a sha: the path keeps a FIXED shape however many mode dimensions are added later, while
      // the header keeps the readable value that can refute it.
      const string cacheRoot = HomeData::getHomeDataDir(true, homeDataDirOverride) + "/vulkan-tune";
      // The ROOT is created here -- it is the STORE, not one cache entry, and the self-test below
      // plants its probe in it.  The PER-KEY directory is created by the write path instead, so a
      // run that concluded nothing leaves no new path behind at all.
      MakeDir::make(cacheRoot);
      d->cacheDir = cacheRoot + "/" + shortHash(d->fingerprint) + "-" + model.fileTag()
                  + "-m" + shortHash(d->modeString);
      d->deviceFile = std::to_string(nnXLen) + "x" + std::to_string(nnYLen)
                    + "-mb" + std::to_string(maxBatchSize) + ".txt";
      d->cachePath = d->cacheDir + "/" + d->deviceFile;
      d->cacheFileExists = FileUtils::exists(d->cachePath);
      d->hasPipelineCacheBytes = pipelineCacheBytesBefore(d->pipelineCacheBytesBefore);

      // Step 3-4's self-check, once per process, under the backend's probe bit 8.  It runs HERE
      // because it needs a REAL directory to prove nothing was written into it and the resolved
      // policy; it is deliberately before the load decides anything, so its line always precedes
      // the decision line in the log (order is what makes a two-line story readable).
      // The directory it gets is the STORE ROOT, not the per-key one: the per-key directory is a
      // write-path side effect now, and a self-test that created it would defeat exactly that
      // (it would also make its own "nothing appeared" assertion partly self-fulfilling).
      if (t2ProbeBit8()) {
        static std::atomic<bool> selfTested {false};
        bool expected = false;
        if (selfTested.compare_exchange_strong(expected, true)) {
          cacheWriteSelfTest(policy, cacheRoot);
          // 2026-09-21 (project-max-principle.md section 9: "the algorithm IS the core of tuning"):
          // the comparison algorithm's own self-test.  It injects KNOWN GROUND TRUTH on the CPU --
          // no device, no files -- and prints coverage / power / spike-robustness / sentinel numbers,
          // because "anti-jitter + accurate" has to be MEASURED rather than claimed.  Its own line, so
          // the three self-tests stay distinguishable in the evidence, and it writes nothing.
          logLine("tune stats self-test: " + TuneStats::selfTestSummary());
        }
      }

      // 🔴 (7a) / W4b Stage 1: THE CANDIDATE CATALOGUE IS A PERSISTENT LINE NOW.  It used to be the third
      // thing inside the bit-8 block above -- a once-per-process diagnostic.  It runs on EVERY decision
      // instead, outside the probe gate, and BEFORE the load chain, because it is not a self-test: it is
      // the step of the decision that establishes WHAT WAS SEARCHED (and the loader's no-row-set branches
      // below record the source it establishes -- see the `rowSource` contract in vulkantuner.h).
      //   * It is the ONE setter of `candidateSearched`, `rowSource = "measured"`, `candidateCoverage`
      //     and `coverageComplete` (R42(d) and R45 census exactly that).
      //   * It writes nothing, builds no pipeline and selects no candidate: it counts and it fills the
      //     decision's fields.  That is why making it persistent cannot change what lands on disk -- the
      //     only outcome (7a) makes reachable is `gate-reject`, whose producer has no production caller yet.
      //   * The two lines stay two lines and keep their order: the probe reports the DECLARED domain, the
      //     catalogue reports the SEARCH, and one line answers one question.
      //   * ⚠️ THE DECISION LINE'S DENOMINATOR: `candidatesEnumerated` is filled from the set the
      //     catalogue actually returned, so `cands=` reports the search's own number.  0/0 would read as
      //     "a complete measurement of nothing" -- the reading this pair of fields exists to prevent.
      {
        logLine("candidate enumeration: " + enumerateCandidateCounts(d->deviceSubgroupSize));
        const vector<Candidate> cands = runCandidateCatalog(d.get());
        d->candidatesEnumerated = (uint32_t) cands.size();
        // (7b) step 5: the SET ITSELF travels with the decision, not just its size.  The measurement
        // runs at the end of `installForHandle`, long after this block, and re-deriving the geometries
        // there would be a second answer to "what did the search return".
        d->candidateGeoms.clear();
        for (size_t ci = 0; ci < cands.size(); ci++) {
          d->candidateGeoms.push_back(cands[ci].geom);
        }
        logLine(formatCandidateCatalog(*d, cands));
      }

      // W4e 2c-c: the measurement primitive's own leg, on its OWN bit (R-14).  It cannot share bit 8
      // with the self-tests above: the primitive refuses to measure while bit 4 or bit 8 is on, because
      // its shapes would be mixed into the user's tables -- so the carrier is bit 16 and `all` still
      // means 0xF.  Same once-per-process shape as the block above, and it is placed AFTER it so the
      // load/decision lines keep their order.
      if (t2ProbeBit16()) {
        static std::atomic<bool> probeSelfTested {false};
        bool probeExpected = false;
        if (probeSelfTested.compare_exchange_strong(probeExpected, true)) {
          probeSelfTest(backend);
        }
      }

      // One loader, one channel (R-29 left exactly one).  The file is read from the cache path
      // and keyed by the device: a file a developer drops there IS the input, by design, and
      // "the file I hand-wrote behaves exactly like the file the tuner wrote" stays true.
      const char* loadPath = nullptr;
      const char* loadKind = nullptr;
      if (d->cacheFileExists) {
        loadPath = d->cachePath.c_str();
        loadKind = "cache";
      }

      if (loadPath == nullptr) {
        d->cacheState = "absent";
        // 🔴 (7a): "measured", not "none".  `rowSource` is the FINAL source, and the catalogue has already
        // run -- it is persistent and it is placed above this chain on purpose -- so with no file supplying
        // a set, the set this decision uses IS this process's own enumeration.  Writing "none" here would be
        // a second and later claim about the same fact, and the false one.  (Note the asymmetry: this
        // branch may write `rowSource`, and must not write `candidateSearched` -- see vulkantuner.h.)
        d->rowSource = "measured";
        d->note = "no cache file for this (device, model, S, mb)";
      }
      else if (retune) {
        // Loudly, because "I set RETUNE and nothing happened" must never be the outcome, and
        // because the reader has to be able to tell this leg apart from a normal load.
        d->cacheState = "retune-ignored";
        // (7a): the file is ignored ENTIRELY, so the decision's set is the one this process enumerated --
        // hence `source=measured`, and the RETUNE leg's reading flips with it in the same batch (A19).
        d->rowSource = "measured";
        d->note = string("KATAGO_VK_TUNE_RETUNE is set: ") + loadKind
                + " ignored entirely (rows and negative entry), deciding again";
        logLine("WARNING: " + string("KATAGO_VK_TUNE_RETUNE is set, so ") + loadKind + " ("
                + loadPath + ") is IGNORED: neither its rows nor its negative entry are used"
                " (delete the env to use it again)");
      }
      else {
        RowsFile file;
        string err;
        string text;
        if (!readWholeFile(loadPath, text, err) || !parseRowsFile(text, d->deviceSubgroupSize, file, err)) {
          d->cacheState = "rejected";
          refuseLoad(loadKind, err);
          d->note = string(loadKind) + " unreadable/unparsable, using the stock heuristic";
        }
        else if (!file.foundFingerprint) {
          d->cacheState = "rejected";
          refuseLoad(loadKind, "no FINGERPRINT header, so the device identity is unknown");
          d->note = string(loadKind) + " has no device identity, ignored";
        }
        else if (file.fingerprint != d->fingerprint) {
          // Section 2.1: a row set is accepted only for the SAME device.  Merging across
          // devices is how a tuned row for a 5080 ends up on a 1660.
          d->cacheState = "rejected";
          refuseLoad(loadKind, "fingerprint mismatch (file=" + file.fingerprint
                              + " current=" + d->fingerprint + ")");
          d->note = string(loadKind) + " belongs to another device";
        }
        else if ((file.foundVersion && file.version != SCHEMA_VERSION)
                 || (file.foundTunerVersion && file.tunerVersion != TUNER_VERSION)) {
          // A matching fingerprint is not enough: the same device read by a different
          // tuner may mean something different by the same bytes.
          d->cacheState = "rejected";
          refuseLoad(loadKind, "file VERSION=" + to_string(file.foundVersion ? file.version : -1)
                              + "/TUNER_VERSION=" + to_string(file.foundTunerVersion ? file.tunerVersion : -1)
                              + " but this tuner speaks VERSION=" + to_string(SCHEMA_VERSION)
                              + "/TUNER_VERSION=" + to_string(TUNER_VERSION));
          d->note = string(loadKind) + " was written by another schema version";
        }
        else if (!file.foundPolicy || !file.foundBaseline) {
          // REQUIRED_ANY_CHANNEL: these carry "the policy changed, so this row set is void".
          // Accepting a file without them would make that guarantee vanish silently -- the file
          // would load, the policy would change, and nothing would invalidate anything -- which
          // is why they are refused on BOTH channels rather than tolerated like MODEL/BOARD.
          d->cacheState = "rejected";
          refuseLoad(loadKind, string("missing header key(s) ")
                              + (file.foundPolicy ? "" : "POLICY")
                              + (file.foundPolicy || file.foundBaseline ? "" : ",")
                              + (file.foundBaseline ? "" : "BASELINE")
                              + " (a missing policy key would let a policy change go unnoticed)");
          d->note = string(loadKind) + " does not say which policy its rows were measured under";
        }
        else if (file.policy != d->policyFingerprint) {
          // The row set belongs to a different policy: void, not "possibly still fine".
          d->cacheState = "rejected";
          refuseLoad(loadKind, "policy fingerprint mismatch (file=" + file.policy
                              + " current=" + d->policyFingerprint + ")");
          d->note = string(loadKind) + " was measured under another policy";
        }
        else if (file.baseline != d->baselineVersion) {
          d->cacheState = "rejected";
          refuseLoad(loadKind, "baseline version mismatch (file=" + file.baseline
                              + " current=" + d->baselineVersion + ")");
          d->note = string(loadKind) + " was measured against another policy baseline";
        }
        else if (!file.absentKeys.empty()) {
          // A machine-written cache has NO legitimate reason to be missing a header key, and
          // "the name routes, the header verifies" only holds if the header actually carries
          // the key: without MODEL, a file copied from another model and renamed to this key's
          // name would pass every check left.  So a missing key is a REFUSAL on the one channel
          // that is left (the import channel, whose fixtures are hand-written, is retired).
          string missing;
          for (size_t i = 0; i < file.absentKeys.size(); i++) {
            missing += (i ? "," : "") + file.absentKeys[i];
          }
          d->cacheState = "rejected";
          refuseLoad(loadKind, "missing header key(s) " + missing
                              + " (a machine-written cache always carries them)");
          d->note = string(loadKind) + " is missing " + missing + ", so the model dimension would not be verified";
        }
        else if (file.foundMode && file.mode != d->modeString) {
          // The user-facing mode is part of the key even though the NAME already routes by it:
          // this check is what turns "a file renamed to look like another mode's file" into a
          // REFUSAL instead of an install of rows measured for a different precision.  It is
          // guarded by foundMode for the same reason the version comparisons keep theirs: MODE is
          // in REQUIRED_FOR_CACHE, so an absent one is refused above, and the guard keeps "is it the
          // RIGHT mode" from being answered by "did it say at all".
          d->cacheState = "rejected";
          refuseLoad(loadKind, "mode mismatch (file=" + file.mode + " current=" + d->modeString + ")");
          d->note = string(loadKind) + " was measured with another precision mode";
        }
        else {
          // Outer key, one item at a time so a refusal names WHICH item did not match.
          string keyMismatch;
          if (file.foundModel && file.model != d->modelIdentity) {
            keyMismatch = "MODEL (file=" + file.model + " current=" + d->modelIdentity + ")";
          }
          else if (file.foundBoard && (file.boardX != nnXLen || file.boardY != nnYLen)) {
            keyMismatch = "BOARD (file=" + to_string(file.boardX) + "x" + to_string(file.boardY)
                        + " current=" + to_string(nnXLen) + "x" + to_string(nnYLen) + ")";
          }
          else if (file.foundMaxBatch && file.maxBatch != maxBatchSize) {
            keyMismatch = "MAXBATCH (file=" + to_string(file.maxBatch)
                        + " current=" + to_string(maxBatchSize) + ")";
          }
          if (!keyMismatch.empty()) {
            d->cacheState = "rejected";
            refuseLoad(loadKind, "outer key mismatch on " + keyMismatch);
            d->note = string(loadKind) + " is for a different model/board/batch";
          }
          else if (file.foundRowSetHash
                   && file.declaredRowSetHash != rowSetHash(file.rows)) {
            // The cache channel: the file must describe the rows it contains.  A mismatch means
            // it was edited or truncated after it was written.
            d->cacheState = "rejected";
            refuseLoad(loadKind, "ROWSET_HASH mismatch (file=" + file.declaredRowSetHash
                                + " computed=" + rowSetHash(file.rows)
                                + "): the file does not describe the rows it contains");
            d->note = string(loadKind) + " was edited after it was written";
          }
          else {
            // W4b: THE SCOPE CHECK, and it is DEEPENED DEFENCE rather than a production guard -- stated
            // here so nobody strengthens the claim later by accident.  Two INDEPENDENT reasons make a
            // mismatching file unreachable from our own writer:
            //   (i) the kind SET is a function of the IDENTITY (KATAGO_VK_FA* is a POLICY knob, and POLICY
            //       is a header key), so one identity never alternates between "mm only" and "mm+fa" --
            //       two different identities are two different files;
            //   (ii) the completeness gate (allKindsConcluded) requires a file to cover EVERY registered
            //       kind, so the writer cannot produce a subset-scoped one.
            // It is still checked, because the file is not always ours: a hand-edited or reflowed file can
            // say anything, and a marker that does not cover a kind this run can select must not suppress
            // that kind's measurement.  A marker about FEWER kinds than this build has is therefore
            // refused; one about MORE kinds cannot exist today, since the parser's closed set IS
            // TUNE_KINDS (the `tuneKindKnown` call above), and the day a kind is removed from the registry
            // this comparison has to be revisited with it.
            if (file.negative) {
              string uncovered;
              for (size_t k = 0; k < sizeof(TUNE_KINDS) / sizeof(TUNE_KINDS[0]); k++) {
                bool covered = false;
                for (size_t i = 0; i < file.negativeScope.size(); i++) {
                  if (file.negativeScope[i] == TUNE_KINDS[k].name) { covered = true; break; }
                }
                if (!covered) {
                  uncovered = TUNE_KINDS[k].name;
                  break;
                }
              }
              if (!uncovered.empty()) {
                string declared;
                for (size_t i = 0; i < file.negativeScope.size(); i++) {
                  declared += (i ? "," : "") + file.negativeScope[i];
                }
                // The refusal takes the same shape as every other rejection above: NOTE is set, and
                // nothing about the file reaches the Decision.  In particular the ROWS are dropped too
                // -- a file whose own marker does not speak for this build is not a file to half-trust,
                // and "rows win, loudly" is reserved for the ROW+NEGATIVE-SAME-KIND case below, which is
                // a statement the format itself forbids rather than one this build cannot interpret.
                refuseLoad(loadKind, "NEGATIVE scope [" + declared + "] does not cover the kind '"
                                    + uncovered + "' this build can select (the marker does not speak "
                                      "for it, so it must not suppress it)");
                d->cacheState = "rejected";
                d->note = string(loadKind) + " carries a marker that does not speak for this run";
                d->rows.clear();
                d->rowCount = 0;
                d->negative = false;
                d->negativeScope.clear();
                d->negativeReason.clear();
                // (7a): the file's marker is dropped together with its rows, and what is left is this
                // process's own enumeration -- the same rule as the two branches above (the catalogue ran,
                // and it ran first).
                d->rowSource = "measured";
              }
            }
            if (d->cacheState != "rejected") {
            d->rows = file.rows;
            d->rowCount = d->rows.size();
            d->rowSource = "cache";
            // 🔴 ONE fact, ONE spelling (W3b): `cacheState` and `negative` used to be two independent
            // expressions of the same question, so limiting only one of them would have printed
            // `cache_state=negative` next to `d->negative == false` -- two readings of ONE run that
            // contradict each other.  Both derive from the named predicate now.
            const bool adoptedNegative = negativeAdoptable(file);
            d->cacheState = adoptedNegative ? "negative" : "loaded";
            d->negative = adoptedNegative;
            if (adoptedNegative) {
              // W4b: the SCOPE travels with the marker.  The cache writer refuses an unscoped marker,
              // so dropping it here would make a file that loaded fine become unwritable on the next
              // write -- the very field this batch added would be lost on the way through.
              d->negativeScope = file.negativeScope;
            }
            if (d->negative) {
              // Carried onto the Decision, not just into the note: the cache writer checks the
              // reason against the closed set and writes the marker back out, so it has to be
              // here.  In the "both ROW and NEGATIVE" case
              // below the reason is deliberately dropped with the stale entry.
              d->negativeReason = file.negativeReason;
              // W2 (post-review): record WHO wrote this marker.  The read-side legs assert that
              // "nothing to do" means "nothing written", so the finalizer must be able
              // to tell a marker it owns from one the loader handed it -- see Decision::NegativeOrigin.
              // (`d->negative` is false on the other branch, where NegativeOrigin::None is already the
              // value, so this one assignment is the only place a from-file origin can be born.)
              d->negativeOrigin = Decision::NegativeOrigin::FromFile;
            }
            if (file.negative && !file.rows.empty()) {
              // Reachable only from a hand-edited or merged file: the writer refuses to emit
              // both (one file = one key makes them exclusive statements about that key).
              // The rows are the useful thing, so they win -- but saying nothing would turn
              // "we wrote NEGATIVE and it still installed rows" into a mystery (lesson #46).
              d->cacheState = "loaded";
              d->negativeScope.clear();
              logLine("WARNING: " + string(loadKind) + " has both ROW and NEGATIVE for this key"
                      " (the writer never emits that); using the " + to_string(d->rowCount)
                      + " row(s) and treating the negative entry as stale");
            }
            if (d->negative) {
              // A negative entry means "this (device, model, S, mb) was DETERMINISTICALLY
              // all-rejected": there is nothing to install and nothing to try.  It is never
              // written on a timeout (t1-schema.md section 1), because "we ran out of time"
              // is not a property of the device.
              d->note = string(loadKind) + ": negative entry (" + file.negativeReason
                       + "), no tuning attempted";
            }
            else {
              d->note = string(loadKind) + " loaded: " + to_string(d->rowCount) + " row(s)";
            }
            }
          }
        }
      }


      d->tuneMs = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();
      logLine(formatDecisionLine(*d, nnXLen, nnYLen, maxBatchSize));
      return d;
    }
    catch (const std::exception& e) {
      // Section 3: fail open and loud.  A tuner that can take down a search is worse than
      // no tuner at all.
      logLine("WARNING: the tuner failed for device " + to_string(deviceIdx) + ": " + e.what()
              + " (fail-open: no tuned rows)");
      return nullptr;
    }
  }

  string formatRowsFile(const Decision& decision) {
    return formatRowsFileImpl(decision);
  }

  // W2 (decision A): pick this decision's OUTCOME from the install accounting, using ONLY names the
  // declared table already carries -- this batch invents no new outcome.
  //
  // It runs at the END OF EACH INSTALL, not at the end of makeDecision: at makeDecision's return the
  // accounting is still EMPTY (no ComputeHandle exists yet, and the installs happen inside its
  // constructor), so finalizing there would leave every decision in "running" for ever.  The last
  // install before the first eval is therefore what decides, and the once point re-finalizes so the
  // body is self-contained.
  //
  //   installed > 0                      -> "measured"
  //   declared > 0, nothing installed    -> "structural-all-reject"  (coarse, see below)
  //   otherwise                          -> stay in "running"        (nothing to conclude: the empty
  //                                         set is the 1b-2b production form, not a verdict)
  //
  // 🔴 The coarse name is recorded EVEN WHEN allRejectReasonsAreDevice is false, and then this prints
  // flag=false + the counts (G-D20).  The write is refused by the table's own `device-only` condition,
  // so no file appears -- and "the file was not written" must never be the only thing the log says.
  // The NEGATIVE ENTRY (negative + negativeReason) is set ONLY on the device-only branch: that marker
  // outlives the process, so our own implementation gap must never leave one behind.
  //
  // 🔴 POST-REVIEW (2026-09-20): this function COMPUTES and does not PRINT, and it only ever removes a
  // marker it created itself.  Both halves are load-bearing:
  //   * SILENT => the loud lines live at ONE reporting site (reportVerdict below), so a second call for
  //     the same event cannot print the same conclusion twice.  The first smoke run with the write live
  //     printed every (a)/(b) NOTICE exactly twice -- once from the install path, once from the once
  //     point -- because this function both computed AND printed.  A finalizer that is not idempotent is
  //     what made that possible; making it a pure function removes the whole class.
  //   * NegativeOrigin::AllReject only => a marker READ FROM A FILE belongs to the loader (the
  //     read-side legs assert "nothing to do means nothing written"), so all this
  //     function may do to it is RETIRE it on the one transition that makes it false by our own account
  //     -- a durable row was installed for this key -- and say so as StaleFromFileNegativeRetired.
  // Together those two rules make `formatRowsFileImpl`'s "both ROW and NEGATIVE" throw (:1078)
  // UNREACHABLE from the write path instead of merely caught: rows are emitted only when a row is
  // durable in EVERY context (`rowsForWrite` is the AND mask), which implies install.installed > 0,
  // which is exactly the branch that retires a from-file marker before the write.
  Decision::FinalizeVerdict finalizeOutcome(Decision& d) {
    // Recompute OUR marker from scratch: the verdict below is a pure function of the accounting, so the
    // same call on the same accounting must produce the same fields (and no leftover marker).
    if (d.negativeOrigin == Decision::NegativeOrigin::AllReject) {
      d.negative = false;
      d.negativeReason.clear();
      d.negativeOrigin = Decision::NegativeOrigin::None;
    }
    if (d.install.installed > 0) {
      d.outcome = "measured";
      if (d.negativeOrigin == Decision::NegativeOrigin::FromFile) {
        d.negative = false;
        d.negativeReason.clear();
        d.negativeOrigin = Decision::NegativeOrigin::None;
        return Decision::FinalizeVerdict::StaleFromFileNegativeRetired;
      }
      return Decision::FinalizeVerdict::Measured;
    }
    if (d.install.declared > 0) {
      d.outcome = "structural-all-reject";
      if (d.anyRowInstalled) {
        // (b) Self-falsification guard.  Some context of this device installed a row for this very key,
        // so "this key has no injection slot" is false BY THIS PROCESS'S OWN ACCOUNT.  The trigger is
        // candidate-scoped (that particular row's path, its shared-memory demand) while the negative
        // entry is key-scoped and STICKY, so it must not be written.
        return Decision::FinalizeVerdict::AnyRowInstalled;
      }
      if (d.allRejectReasonsAreDevice) {
        d.negative = true;
        d.negativeReason = "no-injection-slot";
        d.negativeOrigin = Decision::NegativeOrigin::AllReject;
        return Decision::FinalizeVerdict::DeviceOnlyAllReject;
      }
      return Decision::FinalizeVerdict::OurGap;
    }
    return Decision::FinalizeVerdict::Nothing;
  }

  // THE ONE reporting site for the finalizer's verdict.  Called from BOTH finalize sites (the install
  // path and the once point, both under the owning ComputeContext's tuneMutex) and it prints only when
  // the verdict CHANGES: a repeat for the same event is silent (one event = one line), while a
  // conclusion that really changed afterwards is still said out loud.
  // `Nothing`, `Measured` and `DeviceOnlyAllReject` print nothing BY DESIGN -- their evidence is the
  // decision line and the file itself -- and they are listed explicitly, with no `default:`, so that
  // adding a verdict cannot compile without deciding what it says.
  void reportVerdict(Decision& d, Decision::FinalizeVerdict v) {
    if (v == d.reportedVerdict) {
      return;
    }
    d.reportedVerdict = v;
    switch (v) {
      case Decision::FinalizeVerdict::Nothing:
      case Decision::FinalizeVerdict::Measured:
      case Decision::FinalizeVerdict::DeviceOnlyAllReject:
        break;
      case Decision::FinalizeVerdict::AnyRowInstalled:
        logLine("NOTICE: " + d.deviceName + ": rows were rejected, but another context of this device "
                "INSTALLED at least one row for this SAME key, so 'no-injection-slot' would be false: NO "
                "negative entry is created (a candidate-scoped reason must not become a key-scoped, "
                "sticky verdict)");
        break;
      case Decision::FinalizeVerdict::OurGap:
        logLine("NOTICE: " + d.deviceName + ": every declared row was rejected, but NOT every reason "
                "is a device property (flag=false; our-gap=" + to_string(d.install.dropped_unsupported_our_gap)
                + " illegal=" + to_string(d.install.dropped_illegal)
                + " variant-failed=" + to_string(d.install.variant_failed) + "): the coarse outcome "
                "'structural-all-reject' is recorded WITHOUT a negative entry, because our own "
                "implementation gap is not a property of this device (G-D20)");
        break;
      case Decision::FinalizeVerdict::StaleFromFileNegativeRetired:
        logLine("NOTICE: " + d.deviceName + ": a row was installed for this key, so the NEGATIVE entry "
                "this run READ FROM A FILE is retired before writing (a measurement supersedes a loaded "
                "verdict: one file = one statement, so the marker is dropped instead of being written "
                "next to rows)");
        break;
    }
  }

  void installForHandle(const shared_ptr<Decision>& decision, ggml_backend_t backend) {
    if (decision == nullptr) {
      return;
    }
    // The backend prints the per-context install line (it is the only side that can name
    // the context), so this side reports a failure, and -- B1 -- the from-file face: a row
    // set that came from a FILE is USER DATA that decides which production pipelines get built, and
    // the one thing it must never do is change behaviour without saying so (lesson #46).
    const vk_tune_row_spec* rows = decision->rows.empty() ? nullptr : decision->rows.data();
    vk_tune_install_result installed {};
    // W2 (decision C): the backend fills a SCRATCH buffer, never the fold itself -- it zeroes what it
    // is handed (that is what makes "not touched" distinguishable from "installed"), so handing it
    // `decision->rowInstalled` would erase the fold on every new context.  One byte per declared row,
    // in the same order, sized exactly (the backend refuses a short one instead of truncating).
    vector<uint8_t> thisCtx(decision->rows.size(), 0);
    string err;
    if (!backendInstallRows(backend, rows, decision->rows.size(),
                            thisCtx.empty() ? nullptr : thisCtx.data(), (unsigned) thisCtx.size(),
                            err, installed)) {
      // A FAILED install owns no per-row truth: the buffer may be half-filled, and folding it would
      // claim this context contributed rows when it did not.
      logLine("WARNING: install failed for " + decision->deviceName + ": " + err
              + " (fail-open: this context uses the stock heuristic)");
      return;
    }
    // Conservative accumulation over this device's contexts: a row belongs to the durable
    // set only if EVERY context installed it, so keep the smallest `installed` and the
    // largest drop counts (see the field comment in the header).
    if (!decision->installSeen) {
      decision->install = installed;
      decision->installSeen = true;
    } else {
      decision->install.installed = std::min(decision->install.installed, installed.installed);
      decision->install.dropped_illegal = std::max(decision->install.dropped_illegal, installed.dropped_illegal);
      decision->install.dropped_unsupported = std::max(decision->install.dropped_unsupported, installed.dropped_unsupported);
      // Same direction as every other drop count: one context whose refusal came from our gap
      // disqualifies the device-only claim for the whole key.
      decision->install.dropped_unsupported_our_gap = std::max(decision->install.dropped_unsupported_our_gap,
                                                               installed.dropped_unsupported_our_gap);
      decision->install.dropped_build_failed = std::max(decision->install.dropped_build_failed, installed.dropped_build_failed);
      // The two per-target fields are inside no equality (they describe one row twice), so
      // the conservative direction for them is the maximum: a partial install or a failed
      // variant anywhere disqualifies the stronger claim.
      decision->install.partial = std::max(decision->install.partial, installed.partial);
      decision->install.variant_failed = std::max(decision->install.variant_failed, installed.variant_failed);
    }
    // W2 (decision C): the durable PER-ROW set is the AND of every context's set -- the aggregates
    // above can only bound it, so the write side reads THIS.  The first successful context IS the
    // fold; every later one may only confirm a row (a row installed here and dropped there is not
    // durable).  A size mismatch cannot happen (the row set is fixed per decision); it is handled
    // FAIL-CLOSED -- an empty set writes nothing -- rather than by folding mismatched vectors.
    // 🔴 W2 closure: the COMPUTATION moved into foldRowInstalled() (a pure function, defined next to the
    // self-test that executes it).  The reason is the same one that made finalizeOutcome pure: the fold's
    // two rules -- per-row AND, and "a mismatch clears the set" -- had NO executor anywhere, because the
    // mask write sits after every `continue` and before the pipeline build in the backend
    // (ggml-vulkan.cpp:21400-21401), so two contexts of one device normally agree and a smoke leg cannot
    // be built for either rule.  A judgement can be, in-process.  The MESSAGE stays HERE, at the single
    // call site: if it moved into the pure function's return value and a criterion asserted it, that
    // WARNING would become a counted line by section 9.4's own definition (the smoke parses the
    // self-test's summary string, so the text would be parsed transitively).
    const RowFold folded = foldRowInstalled(decision->rowInstalled, decision->rowInstalledSeen, thisCtx);
    decision->rowInstalled = folded.mask;
    decision->rowInstalledSeen = true;
    if (folded.mismatch) {
      logLine("WARNING: the durable per-row install set has " + to_string(folded.mismatchAcc)
              + " byte(s) but this context declared " + to_string(thisCtx.size())
              + " row(s): the per-row set is cleared (fail-closed: nothing of it will be written)");
    }
    // W2 (decision B): the contexts that contributed, by the backend's own ordinal -- an IDENTITY
    // list.  It is what the decision log can print as "written by ctx 0,1" and what a LATE context is
    // told about below.
    decision->ctxOrdinals.push_back(installed.ctx_ordinal);
    // W2 (post-review): the OR fold.  `install.installed` is the AND fold -- a row is durable only if
    // EVERY context installed it -- but "this key HAS an injection slot" is the OPPOSITE question, and
    // one context that installed a row answers it for good.  Hence a separate flag that only goes ON.
    if (installed.installed > 0) {
      decision->anyRowInstalled = true;
    }
    // G-D20 (t3-shell-plan.md 6.9): "deterministically all-rejected" is a claim about the DEVICE only
    // when no refusal came from OUR implementation gap.  The counts above are MAX-folded, so one
    // context is enough to disqualify the claim for the whole key -- the safe direction, the same one
    // `installed` takes with its min.  `dropped_build_failed` is deliberately NOT part of this gate
    // (the recorded formula is the three below); if a site ever starts counting row-level build
    // failures there, this gate has to be revisited in the same batch.
    decision->allRejectReasonsAreDevice = decision->install.declared > 0
                                       && decision->install.dropped_unsupported_our_gap == 0
                                       && decision->install.dropped_illegal == 0
                                       && decision->install.variant_failed == 0;
    // W2 (decision B/E): a context that arrives AFTER the once point has already fired is told the
    // truth, and the truth is the tri-state the once point recorded -- never a guess:
    //   Wrote         -> the file exists and does NOT contain this context's rows;
    //   RefusedByGate -> no file was written at all (nothing was silently lost);
    //   NotAttempted  -> nothing to say: the hook has not run yet, which is the normal order.
    // "Trigger and consume": this REPORTS.  There is no retry, by design.
    // --- W2 (decision A): this context's accounting is in, so the outcome can be chosen NOW. -------
    // (Not at the end of makeDecision: there the accounting is still empty -- no ComputeHandle exists
    // yet -- so every decision would stay in "running" for ever.)  The last install before the first
    // eval is what decides; the once body below re-finalizes so it stands on its own.
    reportVerdict(*decision, finalizeOutcome(*decision));
    // --- W2 (decision E, 2'): assemble the once hook.  Built HERE, on the install path, and that is
    // the point: it is what makes an empty callback mean "this context's install has not run yet",
    // which the backend's eval point MONITORS instead of assuming.  It is assembled only while the hook
    // has NOT been taken -- re-arming a consumed hook would let a second eval write the same file again.
    //
    // The policy is captured BY VALUE (the body runs on the eval path, which knows nothing about
    // policies), and NO shared_ptr is captured: the closure lives INSIDE the Decision, so capturing one
    // would be a reference cycle, and the Decision arrives as the argument instead.
    //
    // The refusal chain, in one place: the backend zeroes the per-row buffer and REFUSES (rather than
    // truncating) a buffer that cannot hold one byte per declared row; that refusal is an install
    // failure, so this side folds nothing and fails open to the stock heuristic -- loudly, above.
    if (!decision->onceHookTaken) {
      const TunePolicy hookPolicy = decision->tunePolicy;
      decision->onFirstEval = [hookPolicy](Decision& d) {
        // (a) Finalize from the accounting, which the lock has frozen for this whole call, and REPORT
        //     the verdict (the finalizer itself is silent -- see reportVerdict).
        reportVerdict(d, finalizeOutcome(d));
        // (b) The four write gates and the write itself.  writeCacheIfAllowed refuses a read-only mode,
        //     a non-writable outcome (fail-closed through the declared table -- which is also what makes
        //     the flag=false all-reject unwritable), an entry with nothing in it, and an unconcluded
        //     kind; it emits exactly the durable per-row set (rowsForWrite).
        // THE PRODUCTION CALL.  R16's second half pins the CALLERS of writeCacheIfAllowed, not just the
        // writeFileAtomic inside it: exactly two roles may call it -- `self-test` (both calls inside
        // cacheWriteSelfTest) and `production` (here) -- and the production call's return value must
        // feed the tri-state below.  A new, undeclared caller (for instance "write on every eval")
        // therefore cannot appear silently.  Registered in t3-shell-plan.md section 9.1 under R16.
        // WRITE-PATH: production
        string err;
        const bool wrote = writeCacheIfAllowed(d, hookPolicy, err);
        // (c) The TRI-STATE, written back BEFORE returning (2'): it is what a later install reads to say
        //     the right thing.  "Trigger and consume": a refusal is FINAL and is never retried.
        d.writeTarget = d.cachePath;
        if (wrote) {
          d.writeState = Decision::WriteState::Wrote;
          d.writeRefusal.clear();
          // The write-success line.  A COUNTED line (a smoke leg greps it), emitted in ONE statement
          // through the shell's single-line writer, so nothing can interleave inside it.  The count is
          // the rows ACTUALLY WRITTEN -- the durable per-row set -- not the declared count.
          logLine("wrote " + to_string(rowsForWrite(d).size()) + " row(s) to " + d.cachePath);
        }
        else {
          d.writeState = Decision::WriteState::RefusedByGate;
          d.writeRefusal = err;
          logLine("NOTICE: the cache write was refused for " + d.deviceName + ": " + err
                  + " (final: this decision is consumed and is NOT retried)");
        }
      };
    }
    if (decision->writeState != Decision::WriteState::NotAttempted) {
      logLine(string("WARNING: ctx=") + to_string(installed.ctx_ordinal)
              + " installed after the cache write for " + decision->deviceName + ": "
              + (decision->writeState == Decision::WriteState::Wrote
                   ? ("the written file " + decision->writeTarget
                      + " does NOT contain this context's rows")
                   : ("no file was written (the write gate refused: " + decision->writeRefusal + ")")));
    }
  

      // ---- (7b) step 4, first half: THE PRE-CHECK (tune-measurement.md section 2.6's sequence) ---------
      // "预检：只给首个（最重）形状的 STOCK 热身 warmup 块 + K 个热块 ⇒ t_block_hot = K 个热块的中位数 ⇒
      //  逐形状 shots ⇒ 调原语".  It runs HERE because the primitive refuses once an eval has happened or a
      // row is installed (its own assertion, section 2.4), and this is the last point before the return
      // where neither is true.
      //   * WHAT IT PRODUCES: `shots = ceil(MIN_BLOCK_TIME_MS / per_dispatch)`, where per_dispatch comes
      //     from THIS machine's own hot block (section 2.2: the protocol declares `min_block_time`, not
      //     `shots`), and `reps` from `allocateReps` over the A/A sentinel's own scatter -- "how noisy is a
      //     paired block difference here" is exactly what the sentinel measures, and section 2.5's
      //     normalization correction says the scale must be BLOCK-relative.
      //   * THE SHAPE: `m*n*k` is the declared PROXY for "the heaviest shape by per-dispatch time"
      //     (section 2.6), because nothing has been measured yet; the plan's own rule is that the shape
      //     dimension takes the UPPER bound.  A wrong proxy cannot produce a false verdict: `t_block_hot`
      //     feeds only the cost REPORT (the ceiling was cancelled by R-1), never a criterion.
      //   * NOTHING IS BUILT for a candidate: the two objects are the A/A sentinel, i.e. two entries
      //     pointing at STOCK's pipeline (`不建候选管线`).
      //   * FAIL-CLOSED: a refusal leaves `precheckMeasured == false`; the decision line then reads
      //     `precheck=none` and no shots value is fabricated.
      // 🔴 THE GATE IS THE ORDERING RULE, and the smoke measured what happens without it: the primitive's
      // own blocks query the row table, so measuring here FREEZES it before `installForHandle` runs and the
      // rows the load produced are refused ("the row set is frozen: a shape lookup already happened").
      // Three independent statements say the same thing: section 2.5 ("a cache hit does not re-measure"),
      // the primitive's precondition (it REFUSES once a row is installed), and section 2.6's sequence
      // (the measurement is how a key with nothing on disk decides).  So the measurement's window is
      // "nothing to install", and freezing the table there is harmless.
      // 🔴 PARKED (handoff section 7.11): the smoke MEASURED that this placement cannot work.  The
      // primitive's blocks query the row table, and `installForHandle` runs AFTER `makeDecision`, so
      // measuring here freezes the table and the (possibly EMPTY) install is refused -- and that
      // refusal removes the `install (ctx=...): N declared, N installed` line that three legs assert
      // as the evidence that the install ran at all.  R-30 cell (v) assumed the opposite order; the
      // measurement therefore belongs AFTER the install, and its readings cannot ride the decision
      // line (which is printed before the install) -- they need a counted line of their own.
      // The flag is explicit rather than commented out so that the block stays compiled and reviewed:
      // what is parked is the CALL, not the derivation.
      // 🔴 B1 (2026-09-30, ONE cache form): a file that was ACCEPTED is this key's durable answer, and
      // an EMPTY row set is one of its two shapes ("use the stock configuration everywhere").  A loaded
      // answer must therefore NOT be re-measured: `cacheState == "loaded"` is exactly "the file was taken
      // as the answer" (a negative marker spells it "negative", a refused file "rejected", no file
      // "absent"), so the pre-check -- and with it the measurement -- is skipped for a hit and still runs
      // otherwise.  Without this the second run paid for the measurement it had already written down
      // (measured: the two-run acceptance showed `cache_state=loaded` AND a `measure:` line).
      if (!decision->precheckAttempted && !decision->graphShapes.empty() && decision->rows.empty()
          && !decision->negative && decision->cacheState != "loaded") {
        decision->precheckAttempted = true;
        const vk_tune_probe_shape* heaviest = &decision->graphShapes[0];
        unsigned long long bestVolume = 0;
        for (size_t i = 0; i < decision->graphShapes.size(); i++) {
          const unsigned long long volume = (unsigned long long) decision->graphShapes[i].m
                                          * (unsigned long long) decision->graphShapes[i].n
                                          * (unsigned long long) decision->graphShapes[i].k;
          if (volume > bestVolume) {
            bestVolume = volume;
            heaviest = &decision->graphShapes[i];
          }
        }
        vk_tune_probe_shape preShape = *heaviest;
        vk_tune_probe_object preObjects[2];
        for (int i = 0; i < 2; i++) {
          preObjects[i].stock = true;
          preObjects[i].rowIndex = 0;
          preObjects[i].aligned = false;
        }
        // The SEED of the inversion, not a protocol constant: it only sets the first block's length, and
        // the inversion below is what decides `shots`.  It is deliberately kept out of POLICY_CONSTANTS.
        const uint32_t preShots = 1;
        const uint32_t preReps = (uint32_t) TuneStats::HOT_BLOCKS_K;
        vk_tune_probe_params pp {};
        pp.shapes = &preShape;
        pp.nShapes = 1;
        pp.objects = preObjects;
        pp.nObjects = 2;
        pp.shots = preShots;
        pp.warmupBlocks = (uint32_t) TuneStats::WARMUP_BLOCKS;
        pp.reps = preReps;
        pp.stallBlocks = (uint32_t) TuneStats::STALL_BLOCKS;
        // ONE shape is not the slot's whole shape set: saying so is the CALLER's job (R-25), and this
        // pre-check must not claim coverage it does not have.
        pp.coverageComplete = false;
        pp.shotsPerShape = nullptr;
        pp.nShotsPerShape = 0;
        pp.seed = TuneStats::PROBE_ROTATION_SEED;
        pp.spikeBandLo = TuneStats::SPIKE_BAND_LO;
        pp.spikeBandHi = TuneStats::SPIKE_BAND_HI;
        std::vector<double> preD((size_t) 2 * (size_t) preReps, 0.0);
        vk_tune_probe_result preRes {};
        char preErr[512];
        preErr[0] = '\0';
        const bool preOk = ::ggml_backend_vk_tune_probe_candidate(backend, nullptr, 0, &pp, preD.data(),
                                                                 preD.size(), &preRes, preErr, sizeof(preErr));
        if (preOk && preRes.st == VK_TUNE_PROBE_OK && preRes.t_block_hot_us > 0.0) {
          decision->precheckMeasured = true;
          decision->precheckHotUs = preRes.t_block_hot_us;
          // 🔴 THE SECOND POINT (2026-09-29, B-0).  A block is NOT `shots x per_dispatch`: a 1-shot block
          // measures `fixed + marginal`, where `fixed` is the per-block submission + fence cost that no
          // number of shots removes.  The first version divided the 2 ms target by that 1-shot number, so
          // it asked for 33 shots and produced ~350 us -- 6x short of the declared block length, with a
          // quantisation floor of 1/350 = 0.29% against the 1% tolerance (and it is why 14 of 18 candidate
          // pairs read as exactly zero differences: the ruler was coarser than the thing it measured).
          // The model `block(s) = fixed + s * marginal` needs two points, so the probe is run twice at
          // DECLARED shot counts, and the same shape/objects/seed are reused so the two points differ in
          // `shots` and nothing else.
          const uint32_t preProbeShots = 8;
          pp.shots = preProbeShots;
          std::vector<double> preD2((size_t) 2 * (size_t) preReps, 0.0);
          vk_tune_probe_result preRes2 {};
          char preProbeErr[512];
          preProbeErr[0] = '\0';
          const bool preProbeOk = ::ggml_backend_vk_tune_probe_candidate(backend, nullptr, 0, &pp, preD2.data(),
                                                                        preD2.size(), &preRes2,
                                                                        preProbeErr, sizeof(preProbeErr));
          pp.shots = preShots;   // restore: the params object is shared with anything that follows
          const double blkAtPreShots = preRes.t_block_hot_us;
          const double marginalUs = (preProbeOk && preRes2.st == VK_TUNE_PROBE_OK)
            ? (preRes2.t_block_hot_us - blkAtPreShots) / (double) (preProbeShots - preShots) : 0.0;
          const double fixedUs = blkAtPreShots - marginalUs * (double) preShots;
          decision->precheckFixedUs = fixedUs;
          decision->precheckMarginalUs = marginalUs;
          const double targetBlockUs = TuneStats::MIN_BLOCK_TIME_MS * 1000.0;
          double wantShots = 1.0;
          if (marginalUs > 0.0) {
            wantShots = (targetBlockUs - fixedUs) / marginalUs;
          }
          else {
            // A non-positive marginal means the two points disagree with the model (a spike, a clock
            // hiccup, a device whose block time does not scale with shots).  Fail closed by KEEPING the
            // measured 1-shot block and saying so -- inventing a shots value here would be the very
            // extrapolation this cut removes.
            logLine("WARNING: the two pre-check points (shots=" + to_string((long) preShots) + " -> "
                    + to_string((long) blkAtPreShots) + " us, shots=" + to_string((long) preProbeShots)
                    + " -> " + to_string(preProbeOk ? preRes2.t_block_hot_us : 0.0) + " us) give a "
                      "non-positive marginal cost, so `shots` stays at " + to_string((long) preShots)
                    + " and the block will be shorter than the declared minimum");
          }
          if (wantShots < 1.0) {
            wantShots = 1.0;
          }
          decision->precheckShots = (uint32_t) (wantShots + 0.999);
          // The block-relative paired scale, from the sentinel's own pairs: 1.4826 x MAD(d).  `d` NOW
          // ARRIVES BLOCK-RELATIVE -- the primitive divides by THAT SHAPE own anchor median (its `d_out`
          // write says so, and names why) -- which is the very unit `TOLERANCE` lives in, so this line is
          // the whole conversion and there is no second division here.
          // Before (7e-1) this divided by `t_block_hot_us`: that is the HEAVIEST shape's anchor median, so
          // on the heaviest shape the two agreed and on a light shape they did not.  The measured
          // correction removed the coincidence instead of depending on it.
          // THE PAIRED DIFFERENCE IS TAKEN BY THE PRIMITIVE, and the ANCHOR's slots are NaN BY DESIGN
          // (`ggml-vulkan.cpp` seeds every slot with NaN, names the anchor as the FIRST stock object, and
          // leaves the anchor's own slot carrying its MEDIAN rather than a difference).  So the differences
          // live at the NON-anchor object's slots -- object 1 here.  (Measured: the first version
          // differenced the two slots itself, so every pair was NaN and `reps` read 0 -- handoff 7.11,
          // closed in 7.12.)
          std::vector<double> preDiffs;
          for (uint32_t r = 0; r < preReps; r++) {
            const double dv = preD[(size_t) (preReps + r)];
            if (std::isfinite(dv)) {
              preDiffs.push_back(dv);
            }
          }
          if (!preDiffs.empty()) {
            const double mad = TuneStats::madScale(preDiffs);
            const double sigmaRel = mad;   // already block-relative: the primitive normalises
            decision->precheckSigmaRel = sigmaRel;
            const TuneStats::Budget preBudget =
              TuneStats::allocateReps(std::vector<double>(1, sigmaRel), TuneStats::TOLERANCE,
                                      TuneStats::ALPHA, TuneStats::PAIRS_FLOOR, TuneStats::REPS_MAX);
            if (!preBudget.repsRequired.empty()) {
              decision->precheckReps = (uint32_t) preBudget.repsRequired[0];
              // (P0, 2026-09-30) the same budget's other two readings, carried to the printed line.  See
              // the field comments in vulkantuner.h: the clamp is a price, not a refusal, and a formula
              // whose own comment calls it "systematically optimistic" must not out-vote the declared floor
              // silently (that is the 512-vs-25 question P0 exists to make readable).
              if (!preBudget.repsFormula.empty()) {
                decision->precheckRepsFormula = (uint32_t) preBudget.repsFormula[0];
              }
              decision->precheckClampedAtMax = preBudget.anyClampedAtMax;
            }
          }
        }
      }
      // (7b) step 4 / section 7.11: THE PRE-CHECK'S OWN LINE.  It cannot ride the decision line, which is
      // printed BEFORE the install (and the measurement must come after it, or its own blocks freeze the
      // row table and the install is refused).  One line, born with its writer and registered in
      // SHELL_INVENTORY / section 9.4 / the smoke in this same batch.  `measured=0` is the fail-closed
      // state -- a refused pre-check (the primitive refuses under probe bit 4/8 BY DESIGN) reports no
      // reading rather than an invented `shots`.
      {
        std::ostringstream preOs;
        preOs << "precheck: measured=" << (decision->precheckMeasured ? 1 : 0)
              << " hot_us=" << decision->precheckHotUs
            << " fixed_us=" << decision->precheckFixedUs
            << " marginal_us=" << decision->precheckMarginalUs
              << " shots=" << decision->precheckShots
              << " reps=" << decision->precheckReps
              << " reps_formula=" << decision->precheckRepsFormula
              << " clamped=" << (decision->precheckClampedAtMax ? 1 : 0)
              << " sigma=" << decision->precheckSigmaRel
              << " (K=" << TuneStats::HOT_BLOCKS_K << " warmup=" << TuneStats::WARMUP_BLOCKS << ")";
        logLine(preOs.str());
      }

      // ---- (7b) step 5: THE MEASUREMENT (tune-measurement.md section 2.6's main step) ----------------
      // WHAT IT IS: one call to the primitive over the COVERAGE SET -- the shapes that carry 95% of the
      // row-table-side MUL_MAT dispatches (section 2.2 step 2) -- each measured against the catalogue's
      // candidates, the stock anchor and the A/A sentinel.  The pre-check above produces the two numbers
      // this call cannot produce for itself (`shots` from this machine's hot block, `reps` from the
      // sentinel's own scatter), so "pre-check, then measurement" is not decoration: with no reading there
      // is no block length, and the primitive refuses `shots == 0` rather than inventing one.
      //
      // 🔴 WHY IT RUNS HERE AND NOT IN `makeDecision`: MEASURED (handoff section 7.11).  The primitive's
      // blocks query the row table, so measuring before the install FREEZES the table and the (possibly
      // EMPTY) install is refused -- "the row set is frozen: a shape lookup already happened" -- and that
      // refusal removes the `install (ctx=...): N declared, N installed` line three legs read as evidence
      // that the install ran at all.  `makeDecision` -> install -> measure is the order the smoke settled.
      //
      // THE CANDIDATE ROWS: one per (covered shape, enumerated candidate).  Their `key_m`/`key_k` are the
      // SHAPE's (m,k) -- section 3.1's minimal WRITABLE form, since `coverageComplete` is false today --
      // and their family is the SHAPE's own type pair, so the pipeline the primitive builds is the one
      // that shape's dispatcher would look up: R-13's "the candidate must target the shape's object" holds
      // by construction instead of by a comparison that could disagree with the resolver.
      //
      // THE OBJECT ARRAY: 1 stock anchor + one object per candidate row + the A/A SENTINEL, which is a
      // SECOND STOCK entry -- "two entries pointing at the same pipeline" (section 2.6's `n_sentinel = 2`)
      // and that pipeline is stock's.  Three independent statements say the sentinel is a stock entry, and
      // they are why a CANDIDATE-row sentinel would be wrong here rather than merely different:
      //   * the plan's own cost model: the sentinel "与 stock 共管线、不建管线" and is therefore left out
      //     of `N_once` (section 2.6's estimate, lines on `C_once`);
      //   * the primitive's own words at the exclusion site: "only a CANDIDATE is dropped: a stock entry is
      //     an arm (the A/A sentinel too)" (ggml-vulkan.cpp);
      //   * the shell's own device fixture, whose sentinel object is `stock = true` (probeSelfTest).
      // A candidate-row sentinel would also be ILLEGAL on every shape but its own -- the routing key
      // decides legality -- i.e. it would produce no A/A pair for the rest of the covered set.
      //
      // FAIL-CLOSED: a refusal leaves `measurement.ok == false`, its line prints `ok=0`, and NOTHING is
      // accounted (`cands=0/0` there means "no measurement was accounted for", never "a complete
      // measurement of nothing").  The primitive prints its own refusal sentence with its `[T3]` marker, so
      // it is not repeated here.  The gate names the doors the primitive itself refuses on, so the call is
      // not made only to be refused: no pre-check reading, no per-shape weights, no candidates, rows
      // installed, or a negative entry.
      if (!decision->measurement.attempted
          && decision->precheckMeasured
          && decision->precheckReps >= 2 && decision->precheckShots >= 1
          && decision->graphShapeCounts.size() == decision->graphShapes.size()
          && !decision->graphShapes.empty()
          && !decision->candidateGeoms.empty()
          && decision->rows.empty() && !decision->negative
          // B1: an ACCEPTED answer (the empty row set included) is not re-measured -- see the pre-check gate.
          && decision->cacheState != "loaded") {
        decision->measurement.attempted = true;
        // (a) THE COVERAGE SET: counts descending, accumulate until 95% of the total weight is covered.
        //     The tie-break is the graph's own order (a STABLE sort), so two runs over one graph choose the
        //     same set: "which shapes were measured" has to be reproducible from the log alone.
        std::vector<uint32_t> order(decision->graphShapes.size(), 0);
        for (uint32_t i = 0; i < (uint32_t) order.size(); i++) {
          order[i] = i;
        }
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
          return decision->graphShapeCounts[a] > decision->graphShapeCounts[b];
        });
        uint64_t totalCount = 0;
        for (size_t i = 0; i < decision->graphShapeCounts.size(); i++) {
          totalCount += decision->graphShapeCounts[i];
        }
        std::vector<vk_tune_probe_shape> coverage;
        uint64_t coveredCount = 0;
        for (size_t i = 0; i < order.size(); i++) {
          if (i > 0 && coveredCount * 100 >= totalCount * 95) {
            break;   // the 95% is already covered; one more shape would only cost blocks
          }
          coverage.push_back(decision->graphShapes[order[i]]);
          coveredCount += decision->graphShapeCounts[order[i]];
        }
        // (b) THE ROWS, in an order the identity count below depends on: object `2 + c*nCovered + s` is
        //     candidate `c` measured on covered shape `s`.
        const uint32_t armCount = 2;
        std::vector<vk_tune_row_spec> rows;
        rows.reserve(coverage.size() * decision->candidateGeoms.size());
        for (size_t s = 0; s < coverage.size(); s++) {
          for (size_t c = 0; c < decision->candidateGeoms.size(); c++) {
            vk_tune_row_spec row {};
            // The ROW's family comes from the catalogue (its path's own pair), NOT from the shape: the
            // shape's declared pair can be f16 x f32 while the row that would serve it must be f16 x f16
            // (measured -- see the declaration's comment in the header).
            row.src0_type = decision->candidateRowSrc0;
            row.src1_type = decision->candidateRowSrc1;
            row.key_m = coverage[s].m;
            row.key_k = coverage[s].k;
            row.path = decision->candidatePath;
            row.geom = decision->candidateGeoms[c];
            rows.push_back(row);
          }
        }
        std::vector<vk_tune_probe_object> mObjects(armCount + rows.size());
        for (uint32_t a = 0; a < armCount; a++) {
          // The FIRST stock object is the anchor every difference is taken against; the second is the A/A
          // sentinel ("a second entry pointing at the same pipeline", ggml-vulkan.cpp's own words at the
          // anchor's definition).  Both run the dispatcher's own choice for the shape, i.e. the same
          // pipeline, which is exactly what an A/A pair is.
          mObjects[a].stock = true;
          mObjects[a].rowIndex = 0;
          mObjects[a].aligned = false;
        }
        for (size_t r = 0; r < rows.size(); r++) {
          // K1 (same-source align): `aligned` comes from the SHARED header's own function -- the one the
          // row builder calls for `pipe->align` -- so "the variant measured" and "the variant the
          // dispatcher picks" cannot come from two spellings of the rule.  A zero divisor cannot happen
          // (the rule returns a positive alignment for a legal geometry); if it ever did, "not aligned" is
          // the fail-closed answer rather than a division by zero.
          const uint32_t alignFor = vk_tile_row_align_for(rows[r].path, rows[r].geom);
          mObjects[armCount + r].stock = false;
          mObjects[armCount + r].rowIndex = (uint32_t) r;
          mObjects[armCount + r].aligned = alignFor > 0 && (rows[r].key_k % alignFor) == 0;
        }
        // (c) THE CALL.  `shots`/`reps` are the pre-check's own answers.  One block length for every shape
        //     on purpose: the pre-check derived it from the HEAVIEST shape, which is the upper bound
        //     section 2.6 asks for (per-shape shortening is the VRAM-driven refinement, and it needs the
        //     try-allocate path, not a number invented here).  `coverageComplete` is the ENUMERATOR's value
        //     passed through unchanged -- the primitive echoes it and must never derive it.
        vk_tune_probe_params mp {};
        mp.shapes = coverage.data();
        mp.nShapes = (uint32_t) coverage.size();
        mp.objects = mObjects.data();
        mp.nObjects = (uint32_t) mObjects.size();
        mp.shots = decision->precheckShots;
        mp.warmupBlocks = (uint32_t) TuneStats::WARMUP_BLOCKS;
        mp.reps = decision->precheckReps;
        mp.stallBlocks = (uint32_t) TuneStats::STALL_BLOCKS;
        mp.coverageComplete = decision->coverageComplete;
        mp.shotsPerShape = nullptr;
        mp.nShotsPerShape = 0;
        mp.seed = TuneStats::PROBE_ROTATION_SEED;
        mp.spikeBandLo = TuneStats::SPIKE_BAND_LO;
        mp.spikeBandHi = TuneStats::SPIKE_BAND_HI;
        const size_t mSlots = coverage.size() * mObjects.size();
        std::vector<double> mSamples(mSlots * (size_t) mp.reps, 0.0);
        vk_tune_probe_result mRes {};
        char mErr[512];
        mErr[0] = '\0';
        const bool mOk = ::ggml_backend_vk_tune_probe_candidate(backend, rows.data(), rows.size(), &mp,
                                                               mSamples.data(), mSamples.size(), &mRes,
                                                               mErr, sizeof(mErr));
        // 🔴 (B-1) ONE CORRECTION STEP, FROM THE BLOCK THIS RUN ACTUALLY MEASURED.  The derivation above
        // is a FIRST GUESS: it fits `block(s) = fixed + s*marginal` from two short points, and that fit
        // overestimates the marginal cost at large `s` (measured: it asked for shots=142 and the run
        // achieved 1202 us against the declared 2000).  So the achieved block is the feedback -- the same
        // quantity the declaration is about -- and one step lands it.  The re-call reuses the same params,
        // objects and buffer (the buffer is sized by slots x reps, which `shots` does not change, and the
        // primitive NaN-fills it on entry, so a second call is safe by construction).
        const double targetBlockUs = TuneStats::MIN_BLOCK_TIME_MS * 1000.0;
        // 🔴 (B-2) THE CORRECTION IS PER SHAPE, BECAUSE THE BLOCK LENGTH IS.  One shared `shots` gave
        // the heaviest shape ~2000 us and left the light ones at 1582-1999 (measured), while the floor is a
        // per-shape criterion -- so each shape gets ITS OWN ratio, computed from ITS OWN achieved block
        // (`shape_anchor_us[s]`, the same median the primitive divides by).  Handing the heaviest shape's
        // cost to the light ones would repeat the "one ruler for six shapes" mistake one layer down.
        std::vector<uint32_t> shotsPerShape(coverage.size(), mp.shots);
        bool anyShapeShort = false;
        for (size_t sIdx = 0; sIdx < coverage.size() && sIdx < (size_t) mRes.n_shape_anchor_us; sIdx++) {
          const double aUs = mRes.shape_anchor_us[sIdx];
          if (aUs > 0.0 && aUs < targetBlockUs) {
            anyShapeShort = true;
            // 🔴 DAMPED, AND HERE IS WHY (measured, B-2).  The first pass computes every ratio while
            // ALL shapes share one `shots`; once the shapes carry different block lengths their per-dispatch
            // costs move (differently sized blocks interfere through cache and thermals), so a full step
            // overshoots -- the heaviest shape came out at 3596 us against a 2000 us target and the run cost
            // ~40% more for the same blocks, with NO correctness benefit: the floor is a MINIMUM.
            // THE BLOCK LENGTH IS A FUNCTION OF DEVICE STATE, not a constant of the protocol, which is why
            // this is damped rather than solved once -- a device whose state shifts will show it in
            // `anchor_us=` on the measurement line, which is the reading that exists for exactly this.
            // Damping (not a second feedback round): a second round would double the measurement budget to
            // chase a quantity that cannot change any verdict.
            const double rawRatio = targetBlockUs / aUs;
            const double dampedRatio = 1.0 + 0.5 * (rawRatio - 1.0);
            const double want = (double) mp.shots * dampedRatio;
            const uint32_t wantShots = (uint32_t) (want + 0.999);
            if (wantShots > shotsPerShape[sIdx]) {
              shotsPerShape[sIdx] = wantShots;
            }
          }
        }
        if (anyShapeShort && mRes.st == VK_TUNE_PROBE_OK && mp.shots > 0) {
          mp.shotsPerShape = shotsPerShape.data();
          mp.nShotsPerShape = (uint32_t) shotsPerShape.size();
          vk_tune_probe_result mRes2 {};
          char corrErr[512];
          corrErr[0] = '\0';
          const bool corrOk = ::ggml_backend_vk_tune_probe_candidate(backend, rows.data(), rows.size(), &mp,
                                                                    mSamples.data(), mSamples.size(), &mRes2,
                                                                    corrErr, sizeof(corrErr));
          // The params object is shared with whatever follows: leave it exactly as the caller found it.
          mp.shotsPerShape = nullptr;
          mp.nShotsPerShape = 0;
          if (corrOk && mRes2.st == VK_TUNE_PROBE_OK) {
            mRes = mRes2;   // the corrected run IS the measurement: the short one only calibrated the shots
          }
        }
        decision->measurement.shapes = coverage;
        decision->measurement.objects = mp.nObjects;
        decision->measurement.reps = mp.reps;
        decision->measurement.ok = mOk && mRes.st == VK_TUNE_PROBE_OK;
        if (decision->measurement.ok) {
          // (d) HOW MANY OF THE ENUMERATED CANDIDATES WERE REALLY MEASURED.  Read from the primitive's own
          //     NaN discipline -- "which samples exist is answered by the NaN fill and by nothing else" --
          //     rather than inferred: a candidate counts when at least one of its rows produced a FINITE
          //     paired difference.  The anchor's own slots stay NaN by design (there is no difference to
          //     take against itself), which is why the walk starts at the candidate objects, and a slot
          //     that was never written stays NaN too, so a build failure cannot be counted as a measurement.
          uint32_t measuredCandidates = 0;
          for (size_t c = 0; c < decision->candidateGeoms.size(); c++) {
            bool anyCell = false;
            for (size_t s = 0; s < coverage.size() && !anyCell; s++) {
              const size_t o = (size_t) armCount + c * coverage.size() + s;
              const size_t base = (s * mObjects.size() + o) * (size_t) mp.reps;
              for (uint32_t r = 0; r < mp.reps; r++) {
                if (std::isfinite(mSamples[base + r])) {
                  anyCell = true;
                  break;
                }
              }
            }
            if (anyCell) {
              measuredCandidates++;
            }
          }
          decision->measurement.candidatesMeasured = measuredCandidates;
          // THE ONE ACCOUNTING ENTRY: this is where a probe result enters a Decision, and it is the only
          // place that writes the pair's fields.
          noteMeasurementAccounting(*decision, mRes, decision->candidatesEnumerated, measuredCandidates);
          decision->measurement.blocks = mRes.blocks;
          // R50's carrier: the block length this run ACTUALLY achieved on the heaviest shape, so "the
          // block is defined by time" can be checked instead of believed.
          decision->measurement.hotUs = mRes.t_block_hot_us;
          // (7e-3) the per-shape rulers, copied out of the primitive's reading: the judgement gates on the
          // shape's OWN block length, and `ci_missing` is split by which cause produced it.
          decision->measurement.anchorUs.assign(mRes.shape_anchor_us,
                                                mRes.shape_anchor_us + mRes.n_shape_anchor_us);
          decision->measurement.samplesWritten = mRes.samples;
          decision->measurement.pairsIllegal = mRes.pairs_illegal;
          decision->measurement.spikes = mRes.spikes;
          decision->measurement.droppedBuildFailed = mRes.dropped_build_failed;
          decision->measurement.pairsComplete = mRes.pairs_complete;
          decision->measurement.coverageEcho = mRes.coverage_complete;
          decision->measurement.shotsMin = mRes.shots_min;
          decision->measurement.shotsMax = mRes.shots_max;
          decision->measurement.elapsedUs = mRes.elapsed_us;
          // The samples are KEPT rather than dropped with the stack frame: the judgement layer computes its
          // intervals from this very buffer, and re-measuring for it would pay section 2.6's cost twice.
          decision->measurement.samples = std::move(mSamples);
        }
      }
      // (7c) THE JUDGEMENT, and it runs HERE for a reason the gate itself states: the gate judges the
      // MEASUREMENT (upstream of every count the branches read), so it must see THIS run's counts, and
      // the intervals must see THIS run's samples.  `judgement.attempted` is the once flag, like the
      // pre-check's and the measurement's.
      if (decision->measurement.ok && !decision->judgement.attempted) {
        judgeMeasurement(*decision);
      }
      // (7d) THE THREE BRANCHES, ONCE PER DECISION (tune-measurement.md section 2.6(a)).  The order is
      // the ruling order: the gate judged the measurement, the judgement produced the verdicts, and
      // this turns them into an outcome.  It runs at the END of the install path because that is where
      // every input exists -- and BEFORE the once hook, which is the only thing that may write.
      if (!decision->branchesConcluded) {
        decision->branchesConcluded = true;
        concludeBranches(*decision, backend);
      }
      // (7c): THE JUDGEMENT'S OWN LINE.  Same two reasons as the pre-check's and the measurement's: a
      // reading nobody prints is a reading nobody can falsify, and the decision line (printed before
      // the install) cannot carry a verdict that only exists after the measurement.  `winner=-1` means
      // "no candidate passed on every covered shape", which is a READING -- whether that becomes
      // `no-gain` or `no-winner` is the branches' question, decided from `pairsComplete`.
      {
        std::ostringstream jOs;
        jOs << "judge: attempted=" << (decision->judgement.attempted ? 1 : 0)
            << " gate_reject=" << (decision->judgement.gateRejected ? 1 : 0)
            << " gate_high=" << decision->judgement.gateHigh
            << " gate_low=" << decision->judgement.gateLow
            << " verdicts=" << (decision->judgement.verdictsComputed ? 1 : 0)
            << " sentinel_ok=" << (decision->judgement.sentinelEquivalent ? 1 : 0)
            << " ties_bounded=" << decision->judgement.tiesBounded
            << " shapes_below_floor=" << decision->judgement.shapesBelowFloor
            << " min_anchor_us=" << decision->judgement.minAnchorUs
            << " resolution_rel=" << (decision->measurement.hotUs > 0.0
                                         ? 1.0 / decision->measurement.hotUs : 0.0)
            << " sentinel_pairs=" << decision->judgement.sentinelPairs
            << " cand_pairs=" << decision->judgement.candidatePairs
            << " not_worse=" << decision->judgement.notWorsePairs
            << " ci_missing=" << decision->judgement.ciMissing
            << " ci_never=" << decision->judgement.ciNever
            << " pairs_measured=" << decision->judgement.pairsMeasured
            << "/" << (decision->pairsLegal > decision->measurement.shapes.size()
                        ? decision->pairsLegal - (uint32_t) decision->measurement.shapes.size() : 0u)
            << " miss_slot=" << decision->judgement.firstMissingShape
            << ":" << decision->judgement.firstMissingObject
            << " ci_coarse=" << decision->judgement.ciMissingCoarse
            << " ci_wide=" << decision->judgement.ciMissingWide
            << " winner=" << decision->judgement.winner
            << " winner_unstable=" << (decision->judgement.winnerUnstable ? 1 : 0)
            << " winner_margin=" << decision->judgement.winnerMargin
            << " spikes=" << decision->judgement.gateSpikes
            << " blocks=" << decision->judgement.gateBlocks
            << " objects=" << decision->judgement.gateObjects;
        // 🔴 (R52) THE PAIRING INVARIANT, as a LOUD rather than a silent counter: the read set and the
      // measured set are computed independently, and a mismatch means this layer judged slots the primitive
      // never wrote -- which is exactly how a "no candidate is not-worse" claim was minted from 14 pairs of
      // nothing.  The address is printed so the offender is findable from the log alone.
      {
        const uint32_t measuredByPrimitive = decision->pairsLegal > decision->measurement.shapes.size()
          ? decision->pairsLegal - (uint32_t) decision->measurement.shapes.size() : 0u;
        if (decision->judgement.pairsMeasured != measuredByPrimitive) {
          logLine("WARNING: the judgement found samples for " + to_string(decision->judgement.pairsMeasured) + " slot(s) while "
                  "the measurement reported " + to_string(measuredByPrimitive) + " measured slot(s) (the "
                  "first slot without samples was shape " + to_string(decision->judgement.firstMissingShape) + " / object "
                  + to_string(decision->judgement.firstMissingObject) + "): THE PAIRING IS NOT THE MEASUREMENT, so the counts "
                  "this line reports are about slots that were never written");
        }
      }
      logLine(jOs.str());
      }
      // (7b) step 5: THE MEASUREMENT'S OWN LINE, for the same reason the pre-check has one: the decision
      // line is printed BEFORE the install (`makeDecision` returns, then the caller installs), so the
      // accounting this step writes can never ride it -- and a reading nobody prints is a reading nobody
      // can falsify.  `attempted` separates "the gate did not fire" from "it ran and refused": `ok=0` with
      // `attempted=1` is the fail-closed state, and the primitive's refusal sentence is already loud on
      // stderr with its own marker.
      {
        std::ostringstream mOs;
        mOs << "measure: attempted=" << (decision->measurement.attempted ? 1 : 0)
            << " ok=" << (decision->measurement.ok ? 1 : 0)
            << " shapes=" << decision->measurement.shapes.size()
            << " objects=" << decision->measurement.objects
            << " pairs=" << decision->pairsLegal << "/" << decision->shapesCount
            << " cands=" << decision->candidatesMeasured << "/" << decision->candidatesEnumerated
            << " blocks=" << decision->measurement.blocks
            << " hot_us=" << decision->measurement.hotUs
            << " per_dispatch_us=" << (decision->precheckShots > 0
                                          ? decision->measurement.hotUs / (double) decision->precheckShots
                                          : 0.0)
            << " reps=" << decision->measurement.reps
            << " samples=" << decision->measurement.samplesWritten
            << " pairs_illegal=" << decision->measurement.pairsIllegal
            << " spikes=" << decision->measurement.spikes
            << " pairs_complete=" << (decision->measurement.pairsComplete ? 1 : 0)
            << " coverage=" << (decision->measurement.coverageEcho ? 1 : 0)
            << " dropped_build_failed=" << decision->measurement.droppedBuildFailed
            << " shots=[" << decision->measurement.shotsMin << "," << decision->measurement.shotsMax << "]"
            << " elapsed_us=" << decision->measurement.elapsedUs;
        // (7e-3) THE RAW PER-SHAPE RULERS, because `min_anchor_us=` and `shapes_below_floor=` disagreed in
        // the first run and a list is what settles which of the two means what.  Microseconds, in the
        // primitive's shape order, as many as it exported.
        {
          string anchorList = "[";
          for (size_t ai = 0; ai < decision->measurement.anchorUs.size(); ai++) {
            if (ai != 0) {
              anchorList += ",";
            }
            anchorList += to_string((long) (decision->measurement.anchorUs[ai] + 0.5));
          }
          anchorList += "]";
          mOs << " anchor_us=" << anchorList << " n_anchor_us="
              << decision->measurement.anchorUs.size();
        }
        logLine(mOs.str());
      }
}

}  // namespace VulkanTuner

#endif  // USE_VULKAN_BACKEND
