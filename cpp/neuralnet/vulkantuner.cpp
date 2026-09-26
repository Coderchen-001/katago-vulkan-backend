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
    const int TUNER_POLICY_VERSION = 1;

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
    //     explanation; tolerated on IMPORT (hand-written fixtures) with a "[not key-checked]"
    //     note.  For ROWSET_HASH the tolerance goes further on IMPORT: the parsed rows are
    //     hashed live and the declared value is only a hint, since a human cannot compute it.
    const char* const REQUIRED_ANY_CHANNEL[] = { "POLICY", "BASELINE" };
    // VERSION / TUNER_VERSION are here to CLOSE the gap that NegativeReasonSpec used to
    // self-document: a file omitting them was neither refused nor reported.  MODE is the
    // user-facing precision request (Decision::modeString).  All three are name-routed exactly
    // like MODEL/BOARD -- the file NAME already carries the key, so the header only has to
    // confirm it: absence is refused on the cache channel and loudly NOTED on IMPORT.
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
    //   ignoredUnderOff   is it ALSO ignored when cfg vulkanTune=off?  IMPORT is a read, so it
    //                     is honoured; EXPORT writes a file the operator named (not the cache),
    //                     so it is honoured too; RETUNE and PRIME ask for TUNING, which off
    //                     forbids.
    //   consequence       what the operator needs to hear.
    struct RequestSwitch {
      const char* env;
      const char* vacuousWhen;
      bool ignoredUnderOff;
      const char* consequence;
    };
    static const RequestSwitch REQUESTS[] = {
      { "KATAGO_VK_TUNE_IMPORT", nullptr, false, "nothing was imported and no tuned row will be used" },
      { "KATAGO_VK_TUNE_EXPORT", nullptr, false, "nothing was exported" },
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
    // is the fix this note recommended -- absence is refused on the cache channel and reported as
    // "[not key-checked: VERSION,TUNER_VERSION]" on IMPORT.  The comparisons below still keep their
    // `found*` guard, and that is not redundant: they answer "is it the RIGHT version", while
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
    // from a candidate set this process produced ITSELF -- a set read from a file (cache or IMPORT)
    // says what SOME OTHER run saw, and a hand-made or dev-reflowed file says nothing at all.  This is
    // the ONE predicate that answers that question, and it is named and printable because the write
    // path has to cite it (R42 pins the citation).
    //
    // Fail-closed by construction: the tested value is the string "measured", and nothing but the
    // enumerator (step 5) ever writes it.  So today the durable negative entry is UNREACHABLE, which
    // is the intended state until that writer exists -- no writer, no verdict.  The two older guards
    // (`imported` / `anyRowInstalled`) stay in place: they still decide what the FILE would carry on
    // the ungated EXPORT channel, which is a different question from what may be remembered.
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
      { "no-prune-in-effect", [](const Decision & x) { return x.pruneExcluded.empty(); },
        [](const Decision & x) {
          // W4b.  🔴 It NAMES the rules that bit, one by one.  A refusal that only said "a prune was
          // in effect" would send the next reader to the rule table to guess which one; the W3b
          // lesson is that a refusal has to point at the clause that owns it.  The distinction the
          // sentence carries is the whole reason this condition exists: a MUST rule rejects a
          // configuration already proven unusable, while a PRUNE rule only says "not worth trying",
          // so only the latter can make "nothing here works" false.
          string names;
          for (size_t i = 0; i < x.pruneExcluded.size(); i++) {
            names += (i ? ", " : "") + x.pruneExcluded[i];
          }
          return "a PRUNE rule (not a MUST rule) excluded " + to_string(x.pruneExcluded.size())
               + " candidate(s) from the set this run searched [" + names
               + "]: a prune is 'legal but not worth trying', so 'nothing in the domain works' is "
                 "NOT established -- the coarse outcome 'structural-all-reject' is recorded WITHOUT a "
                 "durable verdict (w4b-preview section 1.7)";
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
    // so the CACHE path must refuse to create one while EXPORT may still write it (that is how
    // the format is demonstrated in tests).
    // W2: THE effective row set this process may write down -- ONE spelling, used by BOTH the write
    // gate below and the formatter.  Why one spelling and not two: the gate answers "is there anything
    // to write?" and the formatter emits it, so if they disagreed about a set whose rows were all
    // dropped, the gate would say yes and the file would carry `ROWS 0` with no NEGATIVE -- which the
    // reader treats as "this key is already decided", i.e. tuning suppressed for ever.
    //
    // 🔴 THREE states, and the third one is why this is not a two-line predicate:
    //   Absent       (no install recorded)         -> the DECLARED rows.  The EXPORT demonstration path,
    //                                                 an IMPORTed set that was never installed, the
    //                                                 self-test's synthetic decisions: an ABSENT mask
    //                                                 must never silently turn an export into an empty
    //                                                 file.
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

    bool cacheEntryWritable(const Decision & d) {
      // W2: the gate now asks about the rows this process may actually WRITE (the durable per-row set
      // when an install happened, the declared set otherwise) -- the SAME helper the formatter uses, so
      // "there is something to write" and "here is what gets written" cannot disagree.
      if (!rowsForWrite(d).empty()) {
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
    // Class (b) (every candidate rejected by the validation gate) was declared NOT writable until
    // the gate / tau / soft-gate parameters entered policyFingerprint, because changing a gate
    // parameter would otherwise leave the key permanently skipped (section 6.4 items 1/5 -- the same
    // argument R17 makes for the file's reason).  THAT CONDITION IS NOW MET: the gate/statistics
    // constants are rows of POLICY_CONSTANTS and policyFingerprint() appends them, so a gate change
    // changes the key by construction instead of by trusting that the revision moved (R-28 step (1),
    // 2026-09-26).  The writable flag itself is flipped in the next step of the same batch, with the
    // same-spelling NEGATIVE reason.
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
      // "swept-here" = it must cover the declared domain rather than a sampling surface (W4b),
      // "no-prune-in-effect" = no PRUNE rule may have excluded a candidate from what was searched
      // (W4b).  They live in the table on purpose -- a condition expressed as a branch on `klass`, or as
      // a clause inside one of the gates, would let a rename or a refactor silently change what may be
      // written, which is the "implicit knowledge in a branch" the first design rule bans.  The
      // writer reports the FIRST named condition that does not hold, so a refusal names its clause.
      const char* condition;
    };
    const DecisionOutcomeSpec DECISION_OUTCOMES[] = {
      // A structural all-reject is a KIND-LEVEL, STICKY claim ("this key has no injectable slot"), so
      // it needs all four: the reasons must be about the DEVICE (device-only), the candidate set must
      // be OURS (measured-here), it must be a SWEEP rather than a sample (swept-here), and no PRUNE
      // may have narrowed what we searched (no-prune-in-effect).  The last two are ORTHOGONAL to the
      // first two -- who produced the set, and how completely it was searched, are different
      // questions -- and W4b's measured answer is that this build can satisfy neither tonight: the
      // catalogue is a sampling surface, so no durable all-reject verdict is reachable yet.
      { "structural-all-reject", "structural",  true,  "device-only,measured-here,swept-here,no-prune-in-effect" },
      // A completed measurement writes ROWS, not a verdict about what does not exist.  The two W4b
      // conditions are named here TOO, deliberately: the coverage question is not about the outcome
      // name but about the candidate set, and leaving them off this row would mean an unmarked or
      // pruned run could write rows as if the set had been swept -- the same silent drift, one row
      // over.  Today this row cannot fire either (nothing installs a row yet); it is what keeps the
      // rule attached to the SET rather than to one outcome's spelling.
      { "measured",              "completed",   true,  "measured-here,swept-here,no-prune-in-effect" },
      { "gate-reject",           "gate",        false, ""            },
      { "running",               "measurement", false, ""            },
      { "timeout",               "measurement", false, ""            },
      { "io-error",              "measurement", false, ""            },
      { "device-lost",           "measurement", false, ""            },
      { "no-winner",             "measurement", false, ""            },
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
      { "unstable",              "measurement", false, ""            },
    };

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
    const uint32_t CM2_BK_VALUES[]      = { 64, 128 };
    const uint32_t CM2_SMALLER_VALUES[] = { 0, 1 };
    const AxisDomain CM2_DOMAIN[] = {
      { "BS",      VK_TILE_AXIS_BS,      CM2_BS_VALUES,      2 },
      { "BM",      VK_TILE_AXIS_BM,      CM2_BM_VALUES,      3 },
      { "BN",      VK_TILE_AXIS_BN,      CM2_BN_VALUES,      3 },
      { "BK",      VK_TILE_AXIS_BK,      CM2_BK_VALUES,      2 },
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
    // THE TWO DIRECTIONS ARE NOT THE SAME CLAIM, and conflating them is what makes a negative entry
    // false.  A MUST rule names a configuration ALREADY PROVEN unusable (it does not build, or it
    // computes the wrong answer) -- excluding it cannot make "nothing in this domain works" false.  A
    // PRUNE rule means "legal, but not worth trying", so every PRUNE that really BITES narrows what we
    // searched and weakens the verdict; that is what `pruneExcluded` records and what the
    // `no-prune-in-effect` condition reads.
    //
    //   C4 (t1-predicate.md:105, measured in census-t05-shapes.md section 4.4): the BK axis's useful
    //     values are {64} on cm2's stock geometry.  BK=128 is LEGAL (the predicate accepts it, and stock
    //     itself builds `s_warptile_mmq` with BK=128) but it measured 33 ms pathological vs 1579.8 us at
    //     BK=64 -- a PERFORMANCE verdict, therefore PRUNE, and its exclusion is the ledger's first real
    //     entry.  "Unlock it" needs one experiment: a BK sweep on THIS device with this geometry, i.e.
    //     re-run census-t05-shapes.md section 4.4 the day somebody suspects BK matters here.
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
    // firewall, which must refuse a row that came from the cache or from KATAGO_VK_TUNE_IMPORT and
    // therefore never passed through enumeration.
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
      const char* axis;    // the axis fieldA names; the exclusion sentence says which value was dropped
      AxisValueRule values;      // shape == AxisValue
      ForbiddenCombo combo;      // shape == ForbiddenCombo
      bool prune;          // true = "legal but not worth trying"; false = MUST
      const char* evidence;  // where the verdict comes from
      const char* unlock;    // what experiment would lift it (PRUNE entries must name one)
    };
    const int C4_BK_ALLOWED[] = { 64 };
    const CandidateRule CANDIDATE_RULES[] = {
      { "C4-BK", "mm", RuleShape::AxisValue, "BK", { C4_BK_ALLOWED, 1 }, { nullptr }, true,
        "census-t05-shapes.md section 4.4: mm total 64=1579.8us vs 128=2792.2us with 33 ms pathological "
        "samples (BK=128 legal, just not worth trying)",
        "re-run the BK sweep of census-t05-shapes.md section 4.4 on this device" },
      { "E1-blacklist", "mm", RuleShape::ForbiddenCombo, nullptr, { nullptr, 0 },
        { vk_tile_e1_blacklist_hit }, false,
        "t1-predicate.md:136 (E1): WM==16 && WN==16 && BS>=256 hung the driver, and the custom: "
        "validation lets it through; the boundary lives in vulkan-tile-validity.h so that this table "
        "and the backend's install firewall (ggml_vk_tuned_row_capability_reason) cannot drift",
        nullptr },
    };
    const size_t CANDIDATE_RULE_COUNT = sizeof(CANDIDATE_RULES) / sizeof(CANDIDATE_RULES[0]);

    // The geometry + the rule outcome of ONE candidate.  `ok` is exactly `negativeProvable`'s input: a
    // candidate the rules rejected may not be measured, so it is neither legal nor excluded.
    struct Candidate {
      vk_tile_geometry geom;
      bool legal;
      string exclusion;   // empty = no PRUNE rule bit; otherwise the candidate-level exclusion sentence
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

    // The candidate-table verdicts for ONE geometry, into the out-parameters the catalogue carries.
    //   `pruneName`  non-empty <=> a PRUNE rule rejected it.  The LEDGER entry (the rule's own name:
    //                "which rule bit" is the question the `no-prune-in-effect` sentence answers).
    //   `pruneDetail` the candidate-level sentence ("... excludes BK=128"): which VALUE it dropped, which
    //                is what a reader needs in the log.
    //   `mustName`   non-empty <=> a MUST rule rejected it.  An unusable configuration, not a preference.
    // 🔴 `resolution` says whether the geometry is ADMISSIBLE, and it is a separate out-parameter for a
    // reason this batch paid for: the first version returned `mustName.empty()`, so a PRUNE rule set
    // `pruneName` and the function still said "fine".  The ledger filled up while 36 pre-defined
    // candidates sailed through the gate they had just been excluded by -- and NOTHING in the line could
    // tell that apart from "no rule bit".  Both directions now come out of ONE function, so the two
    // readers (the ledger and the skip) cannot disagree.
    //   "admissible" = admitted by the MUST rules AND not excluded by a PRUNE rule.  A MUST violation is
    //   NOT an exclusion: it is unusable, so it belongs in neither the ledger nor the candidate set.
    enum class RuleResolution { Admissible, ExcludedByPrune, RejectedByMust };

    RuleResolution candidateRuleVerdict(int rulePath, const vk_tile_geometry& g, string& pruneName,
                                        string& pruneDetail, string& mustName) {
      pruneName.clear();
      pruneDetail.clear();
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
          if (r.prune) {
            pruneName = r.name;
            pruneDetail = string(r.name) + " excludes " + r.axis + "=" + to_string(v);
          }
          else {
            mustName = r.name;
          }
        }
        else if (r.shape == RuleShape::ForbiddenCombo) {
          // The verdict comes from the SHARED predicate, and so does the sentence (the firewall
          // returns the same one).  What stays local is only the VOCABULARY the ledger and the log
          // speak: rule names and axis values, which are this table's business.  `ruleAxisValue` is
          // still used for that -- an axis this build does not know returns 0, which is why the
          // readout is spelled as the raw numbers rather than as a claim about why they are wrong.
          const int a = ruleAxisValue(g, "WM");
          const int b = ruleAxisValue(g, "WN");
          const int c = ruleAxisValue(g, "BS");
          // The path is not in the geometry: the catalogue is a per-family walk, and the family's
          // own path is what the shared predicate needs to ask the axis table whether these are
          // row-settable axes at all.  Passed in rather than assumed, so a cm2 catalogue walk cannot
          // inherit a scalar-only boundary.
          if (r.combo.hit == nullptr || !r.combo.hit(rulePath, g)) {
            continue;
          }
          if (r.prune) {
            pruneName = r.name;
            pruneDetail = string(r.name) + " excludes WM=" + to_string(a) + " WN=" + to_string(b)
                        + " BS=" + to_string(c);
          }
          else {
            mustName = r.name;
          }
        }
      }
      // The MUST verdict wins: an unusable geometry is not "excluded by a preference", so a rule table
      // that somehow hit both must not file it in the ledger.
      if (!mustName.empty()) {
        return RuleResolution::RejectedByMust;
      }
      if (!pruneName.empty()) {
        return RuleResolution::ExcludedByPrune;
      }
      return RuleResolution::Admissible;
    }

    // The catalogue for ONE geometry family: it walks the DECLARED stock vector set (not the Cartesian
    // product -- that is the probe's counting surface) and puts every preset through the REAL predicate and
    // then through the candidate table.  It builds nothing: no pipeline, no file, no measurement.
    //
    // `d` receives the two facts the write side needs: `candidateCoverage` (HOW MUCH was searched) and
    // `pruneExcluded` (which PRUNE rules really bit).  Both are set HERE and nowhere else, which is what
    // makes them a property of the enumeration rather than of whoever calls it.
    vector<Candidate> runCandidateCatalog(Decision* d) {
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
        // 🔴 THE LEDGER IS FILLED BY THE WALK BELOW, AND ONLY BY IT.  The first version of this looked at
        // the DECLARED axis values as well ("does a rule reject a value this family declares") and that was
        // wrong in a way worth recording: it made the ledger report an exclusion that had not narrowed the
        // SEARCH, which is the one thing `no-prune-in-effect` exists to measure.  A rule "bites" exactly
        // when it removes a candidate from the set this run considered -- and then that candidate is
        // skipped, which is what the `continue` below encodes.
        for (size_t pk = 0; pk < fam.presetCount; pk++) {
          // 🔴 THE SEARCH DOMAIN IS THIS FAMILY'S OWN STOCK VECTORS, NOT THE CARTESIAN PRODUCT AND NOT the
          // whole vector list.  An earlier draft walked the product here and reported `presets=72` (the
          // two-answers failure this batch is about: the product is what the count-only probe reports as
          // `declared`), and the draft after that filtered the WHOLE list by the family's axis values,
          // which silently pulled the quant family's `s_warptile_mmq` into the non-quant search.  The
          // membership is declared data now (see CM2_PRESETS_*), so neither mistake can recur.
          const uint32_t* pv = CM2_PRESET_VECTORS[fam.presetIndices[pk]];
          vk_tile_geometry g = cm2BaseGeometry();
          g.BS      = pv[0];
          g.BM      = pv[1];
          g.BN      = pv[2];
          g.BK      = pv[3];
          g.smaller = (int) pv[4];
          Candidate c;
          c.geom = g;
          c.legal = false;
          string pruneName, pruneDetail, mustName;
          const RuleResolution verdict = candidateRuleVerdict(path, g, pruneName, pruneDetail, mustName);
          if (verdict == RuleResolution::RejectedByMust) {
            continue;   // unusable: it belongs in neither the ledger nor the set this run searched
          }
          if (verdict == RuleResolution::ExcludedByPrune) {
            // 🔴 THE LEDGER, and the candidate is SKIPPED in the same breath.  A rule "bites" exactly when
            // it removes a candidate from the set this run considered -- so recording it and still
            // considering the candidate would make `no-prune-in-effect` a statement about the table rather
            // than about the search, which is the one thing it exists to measure.
            if (d != nullptr) {
              bool seen = false;
              for (size_t k = 0; k < d->pruneExcluded.size(); k++) {
                if (d->pruneExcluded[k] == pruneName) { seen = true; break; }
              }
              if (!seen) {
                d->pruneExcluded.push_back(pruneName);
              }
            }
            continue;
          }
          if (vk_tile_candidate_reason(path, g, vk_tile_wave_for(32)) != nullptr) {
            continue;
          }
          c.legal = true;
          c.exclusion = pruneDetail;
          out.push_back(c);
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
          d->rowSource = "measured";
          d->candidateCoverage = "sampled";
          if (d->negativeScope.empty()) {
            d->negativeScope = scopeHere;
          }
        }
        break;   // ONE live family today; a second one joins with its own kind and its own coverage rule
      }
      return out;
    }

    // The catalogue's own line (the probe above reports the DECLARED domain; this reports the SEARCH).
    // Full lines only, never truncated: a dropped entry would read as "not excluded", which is the one
    // direction this line must not fail in.
    string formatCandidateCatalog(const Decision& d, const vector<Candidate>& cands) {
      // The declared / post-rule pair PER FAMILY, recomputed from the declaration and the rule table
      // rather than stored, so the two numbers can never disagree with what they describe.  Per family
      // and not in one running variable: with three families in the table, a single pair would report
      // the LAST family's numbers against every family's name.
      size_t declaredCount[GEOMETRY_FAMILY_COUNT] = {};
      size_t postRuleCount[GEOMETRY_FAMILY_COUNT] = {};
      size_t calls = 0, ruleRefused = 0, bkSurvived = 0, inconsistent = 0;
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
                  string pruneName, pruneDetail, mustName;
                  calls++;
                  // The path is the one this catalogue walks, spelled ONCE above it (both call sites
                  // pass it now): the shared ForbiddenCombo predicate needs it to ask the axis table
                  // whether WM/WN/BS are row-settable here, and a defaulted path would be a second
                  // answer to a question the caller already knows.
                  const RuleResolution v = candidateRuleVerdict(VK_TILE_PATH_CM2, g, pruneName,
                                                               pruneDetail, mustName);
                  // 🔴 A PERMANENT cross-check, and it stays: this line's whole subject is "how much did
                  // the rules take away", and the failure this batch hit was exactly a disagreement between
                  // "a rule named a candidate" and "the candidate was skipped" (the first version's
                  // `return mustName.empty()` admitted everything a PRUNE had just excluded, so the ledger
                  // filled while the count did not move).  The two halves are asserted against EACH OTHER
                  // here, offline, forever -- a rule table whose `prune` flag stops matching the skip is a
                  // red line rather than a plausible-looking number.
                  if (v == RuleResolution::ExcludedByPrune) {
                    if (pruneName.empty()) { inconsistent++; }
                    ruleRefused++;
                    continue;
                  }
                  if (v == RuleResolution::RejectedByMust) {
                    if (mustName.empty()) { inconsistent++; }
                    ruleRefused++;
                    continue;
                  }
                  if (!pruneName.empty() || !mustName.empty()) { inconsistent++; }
                  if (g.BK == 128) { bkSurvived++; }
                  // `postRule` counts what survives the CANDIDATE TABLE, MUST and PRUNE alike -- it is the
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
      os << " presets=" << cands.size() << " legal=" << legal
         << " excluded=" << d.pruneExcluded.size() << " by-rule:";
      if (d.pruneExcluded.empty()) {
        os << " none";
      }
      else {
        for (size_t i = 0; i < d.pruneExcluded.size(); i++) {
          os << " " << d.pruneExcluded[i] << ";";
        }
      }
      os << " coverage=" << d.candidateCoverage
         << " remaining=unknown (the format carries no coverage field while sampling cannot be remembered)";
      // The offline cross-check, on the SAME line so the two readings cannot be compared from different
      // runs: `calls` = geometries the candidate table judged, `ruleRefused` = how many it took away, and
      // `inconsistent` = 0 means the rule function's verdict and its out-parameters agreed every time.
      os << " (postRule-check calls=" << calls << " ruleRefused=" << ruleRefused
         << " bk128Admitted=" << bkSurvived << " inconsistent=" << inconsistent << ")";
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
    // A1 export clause (it checks that the header the product actually writes carries them).  `text` is
    // computed FROM the constant, so a value and its spelling cannot drift apart, and removing the loop
    // or a row cannot land green: the exported file stops carrying the constant.
    //
    // WHY THESE ARE IN THE POLICY AT ALL, and it is the plan's own requirement rather than this batch's
    // idea: tune-gaps.md says "代码 / 谓词 / 规则表 / 枚举器 / 判据 / 门参数 -> b<rev16>; **门参数进
    // policyFingerprint 才能写 class-(b)**" (tune-gaps.md:208, restated at :288).  "class (b)" is "every
    // candidate was rejected by the validation gate", i.e. exactly what `gate-reject` is about to become,
    // and the outcome table's own comment refuses to make it writable "until the gate / tau / soft-gate
    // parameters enter policyFingerprint".
    //
    // WHY NOT LEAN ON `b<rev16>` ALONE: the revision is a COARSE automatic identity -- it moves only if a
    // commit moves AND the build actually picks the new revision up, and this repo has produced the
    // opposite (a silently failed build, a regenerated gitinfo: source changed, revision unchanged).
    // In that state a file written under the OLD gate constant would MATCH the stale-revision binary and
    // be adopted -- the one hole the revision chain cannot see.  A value in this string cannot be stale.
    // t1-schema.md's hand-bump list for TUNER_POLICY_VERSION stays for every OTHER code-level semantic
    // change; these rows are simply no longer on it, because here the invalidation is automatic.
    //
    // WHAT BELONGS HERE: a value that changes what a DURABLE VERDICT means -- the spike gate and its
    // band, the drop ratio, and the statistics the interval and the block definition are computed with.
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
      { "too_slow_w",      std::to_string(TuneStats::TOO_SLOW_W)      },
      { "alpha",           std::to_string(TuneStats::ALPHA)           },
      { "tolerance",       std::to_string(TuneStats::TOLERANCE)       },
      { "min_block_ms",    std::to_string(TuneStats::MIN_BLOCK_TIME_MS) },
      { "pairs_floor",     std::to_string(TuneStats::PAIRS_FLOOR)     },
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
      // live, and the loop's absence is visible in the exported file (the smoke asserts it).
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
    // The cache file, KATAGO_VK_TUNE_IMPORT and KATAGO_VK_TUNE_EXPORT all go through these
    // functions, so "the file the tuner writes" and "the file a human can hand-write" are
    // the same thing -- there is no second parser to drift.
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
      // Human-readable keys that were ABSENT; the caller logs them as "not key-checked"
      // rather than pretending the check happened.
      vector<string> absentKeys;
    };

    // ── W3b: MAY THE LOADER BELIEVE A FILE'S NEGATIVE ENTRY? ─────────────────────────────────────────
    // The read side of the same rule W3-P1 closed on the write side.  Two channels share this loader and
    // they are NOT the same kind of statement:
    //   cache  -- this device's own MEMORY.  It is key-checked (fingerprint, version, policy, baseline,
    //             the outer keys, ROWSET_HASH), it is the file the writer maintains, and the operator's
    //             "I do not trust this" switch is KATAGO_VK_TUNE_RETUNE.  Believing it is the DESIGN.
    //   IMPORT -- a REQUEST (a reflow/development channel): the outer keys and ROWSET_HASH are skipped
    //             for it, so it is not even key-checked.  The write side already refuses to mint a
    //             verdict from one ("an IMPORT is not evidence that we measured this key"); the same
    //             sentence has to hold on the read side, or a hand-passed file would decide for this
    //             device by simply asserting it.
    // Named and printable for the same reason `negativeProvable` is: R43 pins that the ONE adoption site
    // CITES this predicate instead of re-deriving the rule from a boolean expression in place.
    bool negativeAdoptable(const RowsFile& file, bool fromImport) {
      return !fromImport && file.negative && file.rows.empty();
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
      // case -- neither -- is only reachable through EXPORT of an empty row set, which is
      // how the format is demonstrated in tests; the CACHE write path cannot create one
      // because cacheEntryWritable() refuses it before this function is called.
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
        // decision that was (or had loaded) a negative entry exported as `ROWS 0` and nothing
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
      err = "nothing to write: no rows and no negative entry (a file that says neither would "
            "load as 'this key is already decided')";
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
    // W4b: the other two states the four-condition row needs, minted the same way.  They must be set
    // EXPLICITLY even though their satisfying values are "swept" and "empty": `candidateCoverage` starts
    // at the fail-closed "unknown" and `pruneExcluded` starts empty, so a fixture that only relied on the
    // defaults would satisfy `no-prune-in-effect` for the wrong reason -- and the breaker for that
    // condition (below) exists precisely because a condition whose default is "satisfied" is the one that
    // can decay into a comment.
    auto aSweptSet = [](Decision & x) { x.candidateCoverage = "swept"; };

    // One breaker per condition NAME in the closed set: the refusing direction of a condition can only
    // be tested by making THAT condition false.  A name with no breaker returns false, and the caller
    // turns that into a FAIL rather than a skip -- an untestable condition is precisely what this loop
    // exists to prevent (a declared condition that nothing can falsify is a comment with a name).
    // 🔴 W4b: `no-prune-in-effect` is the one condition whose SATISFYING direction is the DEFAULT (an
    // empty ledger), so its breaker has to CONSTRUCT a non-empty one.  A breaker that merely cleared the
    // vector would be a no-op on a fixture whose vector is already empty -- it would "test" the condition
    // without ever falsifying it, which is the failure mode condition (b) of the mutation list is aimed at.
    auto breakCondition = [](const string & name, Decision & x) {
      if (name == "device-only")   { x.allRejectReasonsAreDevice = false; return true; }
      if (name == "measured-here") { x.rowSource = "none";                 return true; }
      if (name == "swept-here")    { x.candidateCoverage = "sampled";      return true; }
      if (name == "no-prune-in-effect") {
        x.pruneExcluded.push_back("C4-BK excludes BK=128");
        return true;
      }
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
        // W4b: the row now names FOUR, and the scope is mandatory for the writer, so the fixture carries
        // it.  Each new condition gets its state set HERE rather than left to a default: the positive
        // direction of this loop is what makes forgetting one of them a FAIL instead of a silent pass
        // (the same position W3-P1 landed in, one condition ago).
        aSweptSet(d);
        d.negativeScope.clear();
        d.negativeScope.push_back("mm");
      }
      else if (o.writable && string(o.klass) == "completed") {
        // The rows-file shape: one legal row, so the CONTENT gate passes and only the state gate
        // is left to check.  The geometry is the one the smoke's A3 leg proves legal, and the
        // verdict below re-checks it against the rule table -- a self-test whose own fixture went
        // illegal would otherwise test nothing while looking green.
        aMeasuredSet(d);
        aSweptSet(d);
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
      string err;
      bool verdict = false;
      const bool stateOk = (outcomeWritable(d) == o.writable);   // the table agrees with itself
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
        // checked so the CONTENT gate (which would also refuse `gate-reject`) cannot be mistaken
        // for the state gate.
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
      if (!verdict || !stateOk || !fixtureOk) {
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
    // The cases are the closed set of "what may happen to the marker": (a) an import may not create one,
    // (b) one is not created when our own account refutes it, (c) a from-file marker is retired by a
    // durable row, (d) a device-only all-reject DOES create one, (e) our-gap creates none.  (d) is the
    // CONTROL: without it, a finalizer that simply never set a marker would pass all the others.
    bool verdictOk = true;
    string verdictBad;
    auto verdictName = [](Decision::FinalizeVerdict v) -> const char * {
      switch (v) {
        case Decision::FinalizeVerdict::Nothing: return "Nothing";
        case Decision::FinalizeVerdict::Measured: return "Measured";
        case Decision::FinalizeVerdict::ImportedRejected: return "ImportedRejected";
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

    { // (a) an IMPORTed set may not create a durable verdict
      Decision d;
      d.deviceName = "self-test";
      d.cachePath = probePath;
      d.imported = true;
      d.install.declared = 1;
      d.install.installed = 0;
      d.allRejectReasonsAreDevice = true;   // so no OTHER condition can be the reason it is refused
      // 🔴 W3-P1 moved the durable refusal one step earlier for every structural all-reject this process
      // did not measure, so this call no longer reaches the CONTENT refusal ("nothing to write") it used
      // to reach: what it pins now is that the refusal NAMES provenance.  The guard's own effect -- no
      // marker AT ALL -- is the `wantNegative == false` half, and that is the half that keeps an
      // IMPORTed set from carrying a marker onto the ungated EXPORT channel.
      verdictCase("(a) imported all-reject", d, Decision::FinalizeVerdict::ImportedRejected,
                  Decision::FinalizeVerdict::ImportedRejected, false, measuredMarker);
    }
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
      // this one the `try`/`catch` around the export and cache paths would be a clause nothing could
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

    const bool absent = !FileUtils::exists(probePath);    logLine("cache write self-test: " + to_string(declared) + " outcome(s) declared, "
            + to_string(refused) + " refused by the state gate, " + to_string(allowed)
            + " state-writable; probe file " + (absent ? "absent" : "PRESENT")
            + "; encoding probe (non-ASCII name) written+read-back+removed ["
            + ((encWrote && encReadBack && encRemoved) ? "ok" : "FAIL") + "]"
            + "; (a)(b) no marker / (c) from-file retirement / (d) measured control / (e) our-gap / (f) all-reject->measured retracts / (g)(h) fold AND+mismatch / (i) unmeasured marker refused by name / (j) the formatter's throw is contained ["
            + (verdictOk ? "ok" : "FAIL: " + verdictBad) + "]"
            + " [" + ((ok && absent && encWrote && encReadBack && encRemoved && verdictOk) ? "ok" : "FAIL") + "]"
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
  //      interval-vs-tolerance decisions, the `too-slow` drop, `coverage_complete`, and whether the
  //      numbers are any GOOD -- that last one needs the calibration window, not a self-test.  The gate
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
    // The `too-slow` drop ratio, declared in the judgement layer and HANDED IN (the backend holds no copy
    // of it).  The primary device run below inherits exactly this value, and the control run repeats the
    // same grid with a W that cannot fire -- one run with one W could not tell a criterion that reads its
    // input from one that carries the number internally.
    base.tooSlowW = TuneStats::TOO_SLOW_W;

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
      q.tooSlowW = 1.0;    refusedBy(q, dbuf.size(), "not a usable drop ratio");
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
    uint32_t deviceDroppedTooSlow = 0;
    // The positive control: the same grid with a W that cannot fire (see the block after R-13).
    bool controlRan = false;
    bool controlComplete = false;
    uint32_t controlDroppedTooSlow = 0;
    uint32_t controlBlocks = 0;
    uint32_t controlSamples = 0;
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
          deviceDroppedTooSlow = r.dropped_too_slow;
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
        // The declared W must exclude EXACTLY the two candidate entries: objects 1 and 2 are the same row
        // (the memo twin) and on this fixture both measure several times the anchor's median, while the
        // third non-anchor object is the stock A/A sentinel -- an arm, never a candidate.
        want(!devicePairsComplete && deviceDroppedTooSlow == 2,
             "too-slow: the declared W did not exclude exactly the two candidate pairs, or pairs_complete "
             "did not follow the exclusion (judgeable = measured - dropped)");
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
      // ---- the POSITIVE CONTROL: the SAME grid, only the W changed --------------------------------
      //
      // W is an input (the declared copy lives in cpp/neuralnet/tunestats.h), and this pair of runs is
      // what pins that mechanically: the grid must NOT move (blocks and samples identical) while only the
      // judgeable set does (dropped 2 -> 0, pairs_complete false -> true).  A criterion that ignores its
      // input would leave the first run's numbers on the second; one that SHORTENS the grid instead of
      // excluding after the fact would move the block count.
      if (deviceRan) {
        vk_tune_row_spec ctrlCand = rows[workedRow];
        vk_tune_probe_params qc = q;
        qc.tooSlowW = 10.0;   // no fixture candidate is 10x the anchor, so nothing may be excluded
        std::vector<double> csamp((size_t) 1 * nObjects * base.reps, 0.0);
        vk_tune_probe_result rc {};
        // NOT named after the standard stream: R10e-shell greps for that NAME across the shell file, and a
        // local buffer called `cerr` would read as a second write site (the neighbours are `derr`/`rerr`).
        char qerr[1024];
        qerr[0] = '\0';
        controlRan = ::ggml_backend_vk_tune_probe_candidate(backend, &ctrlCand, 1, &qc, csamp.data(),
                                                           csamp.size(), &rc, qerr, sizeof(qerr))
                     && rc.st == VK_TUNE_PROBE_OK;
        controlDroppedTooSlow = rc.dropped_too_slow;
        controlComplete = rc.pairs_complete;
        controlBlocks = rc.blocks;
        controlSamples = rc.samples;
        want(controlRan, "the too-slow control run (W = 10) did not complete on this device");
        want(controlRan && controlDroppedTooSlow == 0 && controlComplete && controlBlocks == deviceBlocks
             && controlSamples == deviceSamples,
             "the too-slow control: with W = 10 nothing may be excluded, pairs_complete must hold again, "
             "and the grid must be the one the declared W measured (a criterion that ignores its input, or "
             "one that shortens the grid, would move one of these numbers)");
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
       << " dropped_too_slow=" << deviceDroppedTooSlow << " too_slow_w=" << setprecision(2)
       << TuneStats::TOO_SLOW_W
       << " spikes=" << deviceSpikes << " t_block_hot_us=" << setprecision(1) << deviceHot
       << setprecision(4) << " nan_anchor=" << (deviceAnchorNaN ? "ok" : "FAIL")
       << " samples_finite=" << (deviceSamplesFinite ? "ok" : "FAIL") << "]"
       // The control is its own bracketed section rather than extra fields inside `device[...]`: the two
       // runs are two readings, and a reader must not have to guess which run a `pairs_complete` belongs
       // to.  `blocks`/`samples` are repeated on purpose -- "only W changed" is the claim being pinned.
       << " too_slow_control[too_slow_w=" << setprecision(2) << 10.0 << " dropped_too_slow="
       << controlDroppedTooSlow << " pairs_complete=" << (controlComplete ? 1 : 0) << " blocks="
       << controlBlocks << " samples=" << controlSamples << "]"
       << " verifies[override_consumed anchor_hits=0 raii_restore samples+pairs detector_identity "
       << "coverage_complete_echo shots_shape_honoured too_slow_drop]"
       << " not_verified[gate_0.15 vs_tolerance coverage_complete number_quality "
       << "r13_without_coopmat2 shots_shape_multi too_slow_magnitude]"
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
    const TunePolicy& policy
  ) {
    // Path-valued env vars are read through the ONE reader (r6, FileUtils::readPathEnv) so that a
    // non-ASCII path is UTF-8 here and means the same file to every layer below.  The callers test
    // `.empty()`, which is what `!= nullptr && [0] != '\0'` used to express: unset and empty both
    // mean "not given".
    const string importEnv = FileUtils::readPathEnv("KATAGO_VK_TUNE_IMPORT");
    const string exportEnv = FileUtils::readPathEnv("KATAGO_VK_TUNE_EXPORT");
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
    d->rowSource = "none";   // stays "none" unless a load actually produced rows
    d->tuneModeText = policy.modeText;
    // W2: the resolved policy travels WITH the decision so the once hook can capture it BY VALUE.
    // The hook is assembled on the INSTALL path (that is what makes "callback is empty" mean "this
    // context's install has not run yet") and called from the backend's eval path, which knows
    // nothing about policies.
    d->tunePolicy = policy;

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
          // W4b (⑤ first cut): the count-only enumeration probe.  It shares the bit because it is the same
          // subject (the tuned-row surface) and it is the same "once per process, under a probe" shape; it
          // gets its OWN line so the two self-tests stay distinguishable in the evidence.  It writes
          // nothing, so it cannot be mistaken for a decision.
          logLine("candidate enumeration: " + enumerateCandidateCounts(d->deviceSubgroupSize));
          // W4b commit 2: the SEARCH itself, on a second line -- the probe above says how big the declared
          // domain is, this says what was actually looked at and which rules narrowed it.  Kept separate on
          // purpose: one line, one question (the probe is the declaration's business, the catalogue is the
          // search's), and the smoke asserts the grammar of each independently.
          logLine(formatCandidateCatalog(*d, runCandidateCatalog(d.get())));
        }
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

      // One loader for both channels: an import is just a cache file that lives somewhere
      // else and is keyed by the operator instead of by the device.  Sharing it is what
      // makes "the file I hand-wrote behaves exactly like the file the tuner wrote" true.
      const char* loadPath = nullptr;
      const char* loadKind = nullptr;
      bool fromImport = false;
      if (!importEnv.empty()) {
        loadPath = importEnv.c_str();
        loadKind = "KATAGO_VK_TUNE_IMPORT";
        fromImport = true;
      }
      else if (d->cacheFileExists) {
        loadPath = d->cachePath.c_str();
        loadKind = "cache";
      }

      if (loadPath == nullptr) {
        d->cacheState = "absent";
        d->rowSource = "none";
        d->note = "no cache file for this (device, model, S, mb)";
      }
      else if (retune) {
        // Loudly, because "I set RETUNE and nothing happened" must never be the outcome, and
        // because the reader has to be able to tell this leg apart from a normal load.
        d->cacheState = "retune-ignored";
        d->rowSource = "none";
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
        else if (!fromImport && !file.absentKeys.empty()) {
          // Channel-dependent required keys (2026-09-13 review).  Both channels share this
          // loader, but they differ in what they may tolerate:
          //   cache  -- written by us, so there is NO legitimate reason for a header key to
          //             be missing.  "The name routes, the header verifies" only holds if the
          //             header actually carries the key: without MODEL, a cache copied from
          //             another model and renamed to this key's name would pass every check
          //             left.  That is the one path where the model dimension would silently
          //             stop being verified, so the cache channel requires it.
          //   IMPORT -- the operator names the file explicitly, and hand-written fixtures must
          //             stay possible; absence is tolerated and loudly REPORTED ("[not
          //             key-checked: ...]") rather than only noted internally, so a human can
          //             see that the model dimension was not verified on this channel.
          string missing;
          for (size_t i = 0; i < file.absentKeys.size(); i++) {
            missing += (i ? "," : "") + file.absentKeys[i];
          }
          d->cacheState = "rejected";
          refuseLoad(loadKind, "missing header key(s) " + missing
                              + " (a machine-written cache always carries them;"
                                " KATAGO_VK_TUNE_IMPORT tolerates their absence for hand-written files)");
          d->note = string(loadKind) + " is missing " + missing + ", so the model dimension would not be verified";
        }
        else if (file.foundMode && file.mode != d->modeString) {
          // The user-facing mode is part of the key even though the NAME already routes by it:
          // this check is what turns "a file renamed to look like another mode's file" into a
          // REFUSAL instead of an install of rows measured for a different precision.  It is
          // guarded by foundMode so an IMPORTed hand-written fixture that never mentions MODE is
          // still tolerated (and loudly noted) rather than refused as a mismatch -- the same
          // split the channel-dependent required keys use.
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
          else if (!fromImport && file.foundRowSetHash
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
            string keyNote;
            if (!file.absentKeys.empty()) {
              keyNote = " [not key-checked:";
              for (size_t i = 0; i < file.absentKeys.size(); i++) {
                keyNote += (i ? "," : " ") + file.absentKeys[i];
              }
              keyNote += "]";
            }
            if (fromImport && file.foundRowSetHash
                && file.declaredRowSetHash != rowSetHash(file.rows)) {
              // IMPORT: the declared hash is a HINT, never a refusal reason -- a hand-written
              // fixture cannot compute it, and the rows that were actually parsed are what get
              // used.  But it must be LOUD: silently accepting a file that carries a hash for
              // different rows is exactly the "it looked fine" trap (lesson #46).
              logLine("WARNING: " + string(loadKind) + " declares ROWSET_HASH "
                      + file.declaredRowSetHash + " but the parsed rows hash to "
                      + rowSetHash(file.rows) + "; using the parsed rows"
                      " (IMPORT treats the declared hash as a hint)");
            }
            d->imported = fromImport;
            if (fromImport) {
              d->importedFrom = loadPath;
            }
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
                d->rowSource = "none";
              }
            }
            if (d->cacheState != "rejected") {
            d->rows = file.rows;
            d->rowCount = d->rows.size();
            d->rowSource = fromImport ? "import" : "cache";
            // 🔴 ONE fact, ONE spelling (W3b): `cacheState` and `negative` used to be two independent
            // expressions of the same question, so limiting only one of them would have printed
            // `cache_state=negative` next to `d->negative == false` -- two readings of ONE run that
            // contradict each other.  Both derive from the named predicate now.
            const bool adoptedNegative = negativeAdoptable(file, fromImport);
            d->cacheState = adoptedNegative ? "negative" : "loaded";
            d->negative = adoptedNegative;
            if (adoptedNegative) {
              // W4b: the SCOPE travels with the marker.  EXPORT re-emits the line, so dropping it here
              // would make the round trip lose the very field this batch added -- and the cache writer
              // refuses an unscoped marker, so a loaded-then-exported file would become unwritable.
              d->negativeScope = file.negativeScope;
            }
            if (fromImport && file.negative && file.rows.empty()) {
              // The marker is NOT adopted, and that must be LOUD: with an empty row set this file is
              // byte-shape-identical to the trap file that says neither ROW nor NEGATIVE, so
              // `cache_state=loaded` + `rows=0` cannot tell them apart -- the loud is the only
              // discriminator, which is why the acceptance leg asserts on it.  (Registered as a counted
              // line: the smoke's A27 parses the literal below, so it must stay on ONE source line.)
              logLine("WARNING: " + string(loadKind) + " carries a NEGATIVE entry ("
                      + file.negativeReason + ")"
                      + " but NEGATIVE entry ignored: an IMPORT is a request, not a durable verdict"
                      + " (a cache file is this device's memory and is key-checked; an IMPORT file is "
                        "neither)");
            }
            if (d->negative) {
              // Carried onto the Decision, not just into the note: EXPORT has to re-emit the
              // marker (it writes what a cache file would contain), and the cache writer has to
              // check the reason against the closed set.  In the "both ROW and NEGATIVE" case
              // below the reason is deliberately dropped with the stale entry.
              d->negativeReason = file.negativeReason;
              // W2 (post-review): record WHO wrote this marker.  EXPORT re-emits it and the read-side
              // legs assert that "nothing to do" means "nothing written", so the finalizer must be able
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
                       + "), no tuning attempted" + keyNote;
            }
            else {
              d->note = string(loadKind) + " loaded: " + to_string(d->rowCount) + " row(s)"
                      + keyNote;
            }
            }
          }
        }
      }

      // EXPORT writes exactly what a cache file would contain, through the same writer, so
      // the format's round trip is exercised on every run that asks for it (and so a failed
      // write is loud rather than a missing file nobody notices).
      if (!exportEnv.empty()) {
        // W4b (D3b): the EXPORT path is wrapped in its own protection, because `formatRowsFile` REFUSES
        // some decisions by THROWING (a NEGATIVE entry whose reason or scope is outside its closed set,
        // more rows than the cap, ROW and NEGATIVE for the same kind).  Until this batch that refusal
        // escaped to makeDecision's outer catch and killed the run after the work was already done; the
        // contract every other write path already states -- "every load failure is fail-open + loud, never
        // an exception out of the hook" -- applies to a write path at least as much, since a tuner that
        // can take down a search is worse than no tuner.  Same shape as the cache path: convert to an
        // error string and say so, loudly, rather than propagating.
        // R16's clause about this: EVERY declared write path must be exception-safe, export and cache
        // alike -- the declaration is what makes the next write path have to answer the same question.
        string path = exportEnv;
        string err;
        string content;
        // WRITE-PATH: export
        // Declared for guardrail R16.  An export writes a file the OPERATOR named, never the
        // cache, so it is allowed even in read-only mode (vulkanTune=off) -- which is why this
        // is the one write path that does not have to sit behind a policy.off gate.  The cache
        // write path (step 3) carries its own tag and MUST be gated.
        bool wrote = false;
        try {
          content = formatRowsFile(*d);
          wrote = writeFileAtomic(path, content, err);
        }
        catch (const std::exception& e) {
          wrote = false;
          err = string("the export was refused by the formatter: ") + e.what();
        }
        // ONE reporting site for both outcomes, so the success line and the failure line cannot drift
        // apart and neither can be forgotten.
        if (wrote) {
          logLine("export path=" + path + " rows=" + to_string(d->rowCount)
                  + " bytes=" + to_string(content.size()) + " ok");
        }
        else {
          logLine("WARNING: export to " + path + " failed: " + err + " (fail-open: nothing changed)");
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
  //   * NegativeOrigin::AllReject only => a marker READ FROM A FILE belongs to the loader (EXPORT
  //     re-emits it; the read-side legs assert "nothing to do means nothing written"), so all this
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
      if (d.imported) {
        // (a) An IMPORTed row set may not CREATE a durable verdict.  The product says so itself in the
        // same runnable path ("IMPORT ... not evidence that we measured it"): a hand-off is a REQUEST,
        // while the negative entry outlives the process.  Reading an existing file stays allowed; only
        // concluding from an imported set is refused.  The write gate then refuses by itself (rows
        // empty and no negative), which is also why no file appears.
        return Decision::FinalizeVerdict::ImportedRejected;
      }
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
      case Decision::FinalizeVerdict::ImportedRejected:
        logLine("NOTICE: " + d.deviceName + ": every imported row was rejected, but an IMPORT is not "
                "evidence that we measured this key (it is a reflow/development channel): NO negative "
                "entry is created -- nothing durable may be concluded from an imported set");
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
    // the context), so this side reports a failure, and -- B1 -- the IMPORT face: an
    // imported row set is USER DATA that decides which production pipelines get built, and
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
    if (decision->imported && installed.declared > 0) {
      logLine("IMPORT installed " + to_string(installed.installed) + " of "
              + to_string(installed.declared) + " row(s) from " + decision->importedFrom
              + " (not key-checked: IMPORT is a reflow/development channel, not evidence that we measured it)");
    }
  }

}  // namespace VulkanTuner

#endif  // USE_VULKAN_BACKEND
