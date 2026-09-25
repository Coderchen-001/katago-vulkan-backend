#ifndef NEURALNET_VULKANTUNER_H_
#define NEURALNET_VULKANTUNER_H_

// vulkantuner.h -- the CONTROL PLANE of the Vulkan runtime tuner (T3).
//
// WHY THIS FILE EXISTS (t3-shell-plan.md section 9)
// The OpenCL tuner is clean because its measurement can move out of the backend.  Ours
// cannot: a Vulkan pipeline is a runtime object (specialisation constants, an `align`
// side effect on kpad/aligned, a pre-record replay bypass), so the tuner has to live
// inside the process.  What we buy instead is a NARROW interface: the backend exposes
// four functions and nothing else, and every piece of tuner policy -- which candidates
// exist, which are legal, which gate applies, how they are ranked, where the cache lives
// -- lives here (t3-shell-plan.md section 9.1 R1/R5).
//
// The dividing line, mechanically enforced by t3-guardrails.py:
//   * this side never touches a `vk*` type, a pipeline handle or a device object;
//   * the backend never learns what a "candidate" is.
//
// Logging goes to std::cerr, like the backend's T2/T3 probes.  That is deliberate: the
// acceptance for this cut is a readable ORDER of "decision -> install -> frozen ->
// pr_recording", and three lines in three different streams is not readable order.

#include <cstddef>
#include <cstdint>
#include <functional>   // W2: Decision::onFirstEval carries the once hook as DATA (std::function),
                        // not as a fifth interface name -- R1 counts NAMES, and the hook's body stays
                        // in vulkantuner.cpp so the backend never grows a control surface (R5/R6).
#include <memory>
#include <string>
#include <vector>

#include "../core/commontypes.h"   // enabled_t: the USER-FACING precision request, carried as
                                   // identity.  Its toString() is the SAME spelling the backend
                                   // reports in its own `[prec] mode=` line, so "what the user
                                   // asked for" has one name in the log and in the cache.
#include "ggml-backend.h"
#include "vulkan-tile-validity.h"

// The backend interface (defined in the vendored ggml-vulkan.cpp).  Declared HERE, at
// global scope with the C++ linkage the definition has, and NOT in ggml-vulkan.h: that
// header is vendored public API and the ggml CMake surface stays untouched
// (t3-shell-plan.md section 7.1-7).  Four ITEMS, FIVE names (R1): fingerprint / install_rows /
// probe_candidate / export_state / import_state -- and R1's expected set already carries all five,
// so this section's count and the guardrail agree.
//
// Returns the written length (excluding the terminator) or 0 when the backend is not a
// Vulkan one / the buffer is too small.
size_t ggml_backend_vk_tune_fingerprint(ggml_backend_t backend, char* out, size_t outSize, uint32_t* sgOut);
// Installs a row set into ONE context.  Returns false and fills `err` on refusal; the
// backend logs the refusal loudly itself.
//
// `out` (optional, B1) reports the install in three counts plus a per-reason breakdown:
// declared / installed / dropped_*.  The shell needs them because "every row in the file
// is hit at least once" is graded against the rows that INSTALLED, and because the write
// side (step 5) may only write rows this process really installed.
//
// `rowInstalled` / `rowInstalledCap` (W2) are the CALLER'S per-row truth buffer: the
// backend zeroes it, then writes 1 at the index of every row that installed.  It is a
// PARAMETER and not a member of `out` because `out` is reset as a whole on entry, and that
// reset is correct only while every field of `out` belongs to the backend (see the
// ownership note on vk_tune_install_result in vulkan-tile-validity.h).  A buffer that cannot
// hold one byte per declared row is refused loudly -- never truncated.
bool ggml_backend_vk_tune_install_rows(ggml_backend_t backend, const vk_tune_row_spec* rows, size_t nRows,
                                       char* err, size_t errSize,
                                       unsigned char* rowInstalled, unsigned rowInstalledCap,
                                       vk_tune_install_result* out);
// W4e: the MEASUREMENT primitive.  Runs the interleaved paired blocks and returns the differences the
// judgement layer consumes; it decides nothing (see the vk_tune_probe_* block in
// vulkan-tile-validity.h).  Returns false and fills `err` on refusal; the backend logs it loudly.
//
// 🔴 `dOut` is the CALLER'S buffer, laid out [shape][candidate][rep], and `out->samples` says how many
// entries were written.  It is an argument and not a member for the same reason install_rows' per-row
// truth buffer is: the entry resets the result struct as a whole.
bool ggml_backend_vk_tune_probe_candidate(ggml_backend_t backend, const vk_tune_row_spec* cands, size_t nCands,
                                          const vk_tune_probe_params* params,
                                          double* dOut, size_t dCap,
                                          vk_tune_probe_result* out,
                                          char* err, size_t errSize);

namespace VulkanTuner {

  // The model half of the cache's outer key, built by the caller from the loaded model's
  // description (the shell must not learn what a LoadedModel is).  Split into the two forms
  // the format needs:
  //   * `name` goes ONLY into the file header -- a model name can contain characters that
  //     are not safe in a filename, and it is redundant with the sha for routing;
  //   * `fileTag` is the filename-safe `mv<version>-sha<sha16>` form.
  struct ModelIdentity {
    int version = 0;
    std::string sha16;   // sha256[0:16], or "none" when the model has no sha
    std::string name;    // for the header only
    // `mv<version>-sha<sha16>`: used in the file name.
    std::string fileTag() const {
      return "mv" + std::to_string(version) + "-sha" + (sha16.empty() ? std::string("none") : sha16);
    }
    // `mv<version>-sha<sha16>-name<name>`: written to the header's MODEL field.
    std::string headerValue() const {
      return fileTag() + "-name" + (name.empty() ? std::string("none") : name);
    }
  };

  // The USER-facing mode (cfg `vulkanTune`), resolved by the caller because it owns the
  // ConfigParser (the shell must not read one).  The dev master switch (env KATAGO_VK_TUNE=0)
  // is STRONGER and is resolved inside the shell before this is consulted.
  //
  //   Auto -- read the cache; on a miss, tune and save (today: nothing to tune yet).
  //   Off  -- READ-ONLY: read the cache, use a hit, fall back to stock on a miss, never tune,
  //           never write the cache and never write a negative entry.  IMPORT is still
  //           honoured, because it is a read.
  //
  // ⚠️ Off is NOT a reference arm: it may install tuned rows, so "bit-identical to stock"
  // acceptance may only ever use KATAGO_VK_TUNE=0.
  struct TunePolicy {
    bool off = false;
    // What the decision line prints: "auto(cfg)" / "off(cfg)".  The env case is silent by
    // design (the reference arm produces no decision line at all), which the caller documents.
    std::string modeText = "auto(cfg)";
  };

  // One tune decision, taken at most once per (evaluator, device) and reused by every
  // context on that device (t3-shell-plan.md section 2).  Immutable once built.
  struct Decision {
    int deviceIdx = 0;
    std::string deviceName;
    std::string fingerprint;      // full device/driver fingerprint (cache header identity)
    // The cache lives TWO LEVELS down, and the PATH is a pure function of the key -- never a
    // directory listing or a trial read, because that would trade routing for guessing (and could
    // pick up another key's file):
    //   <homeDataDir>/vulkan-tune/<device16>-mv<modelVer>-sha<model16>-m<mode16>/<S>x<Y>-mb<n>.txt
    // The DIRECTORY holds what IDENTIFIES the cache (device, model, mode: stable across a sweep);
    // the FILE holds the two segments a sweep varies, so `benchmark -threads 1,2,4` shows its
    // buckets side by side instead of scattering them, and deleting one directory re-tunes exactly
    // one (device, model, mode).  Version and policy are deliberately NOT in the path: they are
    // the header's business, and putting them here would orphan every file on every bump.
    std::string deviceFile;       // the FILE segment only: "<S>x<Y>-mb<n>.txt" (filename-safe)
    // The USER-FACING precision mode, as an identity string ("fp16=auto").  Its sha16 goes into
    // the directory NAME (one mode = one file, and a FUTURE dimension appends to this string without
    // lengthening the name); the raw string goes into the header, so a hand-renamed or
    // cross-machine-copied file is refuted by its own contents instead of being trusted because
    // its path looks right.  Produced by modeString() -- the ONE producer -- never spelled out
    // at a call site, or the name and the header could disagree.
    std::string modeString;
    std::string cacheDir;         // <homeDataDir>/vulkan-tune/<device16>-mv<ver>-sha<model16>-m<mode16>
                                  // Created at WRITE time, not when the decision is taken: "nothing
                                  // decided => nothing written" has to include the directory, or a
                                  // no-op run would still leave a new path behind (A30 asserts it).
    std::string cachePath;        // cacheDir + "/" + deviceFile (may not exist yet)
    bool cacheFileExists = false;
    // The rest of the cache's outer key (t1-schema.md section 4): the same device can hold
    // a different model, board size or max batch, and none of those may share a row set.
    std::string modelIdentity;    // header form (mv…-sha…-name…)
    // The DEVICE's raw subgroup size, injected by the backend through the fingerprint call
    // (ggml_backend_vk_tune_fingerprint's `sgOut`), NOT a wave: the shell never converts it, it
    // calls vk_tile_wave_for() at each use.  Why the shell needs it at all (G-C3): a row's BS is
    // derived as num_warps * wave, so the geometry table cannot judge a row without knowing the
    // wave of the device the row is FOR -- and the shell's own parse-time validation has no device
    // handle.  Default 0 = "not injected"; 0 is refused downstream (both here and by
    // vk_tile_candidate_reason's wave == 0 clause) rather than silently standing in for wave 32.
    uint32_t deviceSubgroupSize = 0;
    bool negative = false;        // this key's row set is a "deterministically all-rejected" marker
    // HOW THIS DECISION WAS REACHED, as one string from the closed set in vulkantuner.cpp
    // (step "3-4").  The default is NOT writable, deliberately: a decision nobody gave an outcome
    // to has by definition not finished, and the failure this field exists to prevent is
    // remembering a timeout / an IO error / a lost device / a run with no winner as if it were a
    // property of the device ("we ran out of time" must never become "this device cannot be
    // tuned").  Only a COMPLETED measurement or a STRUCTURAL all-reject may be written.
    std::string outcome = "running";
    // Why the reason is a CLOSED SET and not free text (t3-shell-plan.md section 6.4 item 2): a
    // NEGATIVE entry is a claim that OUTLIVES the process -- it suppresses tuning until something
    // invalidates it -- so it has to be mechanically comparable, and a file carrying a reason this
    // tuner cannot interpret must not load (the same rule that refuses an unknown header key).
    // The set is NEGATIVE_REASONS[] in vulkantuner.cpp; class (b) (every candidate rejected by the
    // validation gate) and class (c) (timeout / IO failure / device lost / no winner) are
    // deliberately absent from it, so they cannot be written at all.
    std::string negativeReason;
    // W2 (post-review): WHO put the marker above there, which is what decides who may take it away.
    //   None      -- no marker.
    //   FromFile  -- the LOADER read it from a file (the cache channel or an IMPORT; one entry point,
    //                vulkantuner.cpp load path).  It is load-bearing: EXPORT re-emits the marker, and
    //                the read-side legs assert "nothing to do means nothing written" byte for byte, so
    //                the FINALIZER never touches it -- except for the ONE transition that makes it false
    //                by this process's own account (a durable row was installed for this key), which it
    //                retires loudly.
    //   AllReject -- THIS process concluded "deterministically all-rejected" from the install
    //                accounting.  The finalizer owns this one and recomputes it from scratch on every
    //                call, so calling it twice can never leave a stale marker behind.
    // The distinction is not cosmetic: one field with two writers and no record of who wrote it is
    // exactly the silent sharing this project refuses, and the two markers are removed for different
    // reasons (a loader's marker is superseded by a measurement; ours is retracted by our own rules).
    enum class NegativeOrigin { None, FromFile, AllReject };
    NegativeOrigin negativeOrigin = NegativeOrigin::None;

    // G-D20 (t3-shell-plan.md 6.9): "deterministically all-rejected" is a claim about the DEVICE
    // only when every rejection reason is a device property.  A rejection caused by OUR
    // implementation gap (cm1 not wired, the 64-bit variant not built) is NOT a device verdict, and
    // writing it down would remember our gap forever and never rescan after it closes.
    //
    // Filled by the install accounting.  Default false = "not proven device-only", so an unset flag
    // can never make a structural verdict writable (the safe direction, like the fail-closed
    // outcome lookup below).
    bool allRejectReasonsAreDevice = false;
    // The policy half of the header (t1-schema.md section 4): a row set measured under one
    // policy must not be reused under another.  `policyFingerprint` covers the build
    // revision, the tuner's policy version and the policy-relevant env knobs;
    // `baselineVersion` is the bump-this-when-the-fork's-default-policy-changes constant.
    std::string policyFingerprint;
    std::string baselineVersion;
    std::string rowSetHash;       // computed from the rows, never carried in from a file
    int nnXLen = 0;
    int nnYLen = 0;
    int maxBatchSize = 0;
    bool imported = false;        // row set came from KATAGO_VK_TUNE_IMPORT
    std::string importedFrom;     // path the import came from
    // The rows to hand to the backend, in the wire form both sides share.  Parsed from the
    // cache or the import file; still refused by the backend's T4 gate today (loudly), and
    // never silently dropped.
    std::vector<vk_tune_row_spec> rows;
    // W3-P1: WHERE THE CANDIDATE SET CAME FROM.  Four states, and the fourth is the one the write
    // side reads: "none" (nothing was loaded -- the declared set is empty, or the self-test
    // synthesized it), "cache"/"import" (a FILE produced the candidates), "measured" (THIS process's
    // own enumeration produced them).  The value is what `negativeProvable` tests, so the initial
    // value is the fail-closed one: a decision nobody marked can never mint a durable verdict.
    // 🔴 Production has NO setter of "measured" today -- the enumerator is step 5 -- which is exactly
    // why the key-scoped NEGATIVE entry is unreachable until then.  R42 counts the setters rather
    // than trusting this sentence: production 0, the self-test's one named fixture 1.
    std::string rowSource = "none";   // "none" / "cache" / "import" / "measured"
    // W4b (⑤ first cut): WHICH SUBSET of the declared domain this process actually looked at, as THREE
    // states rather than a bool.  "who measured it" (rowSource above) and "how much of the domain was
    // covered" are ORTHOGONAL questions -- both must be answered before a NEGATIVE entry may be
    // remembered, because "we sampled 6 presets and none worked" is not "this device has no injectable
    // slot".  The third state is what makes this a three-state field and not a flag: a W3-P1 lesson --
    // a bool here would create the illegal combination "not swept and not sampled" with nobody to refuse.
    //   unknown -- nobody said.  The initial value is the FAIL-CLOSED one: `swept-here` tests for the
    //              string "swept", so a decision nobody marked can never mint a durable verdict.
    //   sampled -- a sampling surface: the enumerator walked the declared STOCK VECTOR SET (a strict
    //              subset of the Cartesian domain the count-only probe reports), or any other proper
    //              subset.  A legitimate way to search, and NOT a complete one -- which is all this
    //              state has to mean, so it stays true whether the subset is the stock presets or a
    //              chosen sample.
    //   swept   -- every point of the declared domain was put through the predicate (nothing tonight
    //              produces this: 72 > any sane N_max, which w4b-preview section 9 pre-registers as
    //              "sample it").  It exists so the condition has a satisfying direction to test.
    // It is NOT written into the file and does NOT enter the key on purpose: while sampling cannot
    // produce a NEGATIVE entry, "a file carries NEGATIVE" already implies "swept", so a coverage
    // header key would be a format event that says nothing.  🔴 THE TRIGGER: the day a sampled
    // negative entry is allowed to land, a coverage field MUST be added in the same batch
    // (t3-shell-plan.md F17).
    std::string candidateCoverage = "unknown";   // "unknown" / "sampled" / "swept"
    // W4b: WHAT the enumerated candidate set covered -- the kind TOKENS a NEGATIVE entry speaks for.
    // It is the third axis of the same honesty question: `measured-here` says WHO produced the set,
    // `candidateCoverage` says HOW MUCH of the domain it looked at, and this says WHICH KINDS the
    // conclusion is about.  One file = one key, and a key can carry several kinds (mm today, fa
    // later), so a bare `NEGATIVE <reason>` would assert a kind it never measured.
    // The tokens come from TUNE_KINDS[] (the ONE kind registry); empty = "no scope stated", which
    // the writer refuses on the durable channel rather than reading it as "all kinds".
    std::vector<std::string> negativeScope;
    // W4b (⑤ first cut): THE PRUNE LEDGER -- the names of the candidate-table rules that REALLY
    // excluded a candidate from the set this run searched.  Why a ledger and not a flag: `MUST` and
    // `PRUNE` are not the same claim.  A MUST rule names a configuration already proven unusable
    // (it does not build, or it computes the wrong answer), so excluding it cannot make "nothing in
    // this domain works" false.  A PRUNE rule is "legal but slow / not worth trying", so every one
    // that BITES weakens a negative verdict -- and the refusal sentence has to be able to NAME which
    // one (the W3b lesson: a refusal must point at the right clause).
    // 🔴 EMPTY IS THE NORMAL STATE, and the condition reads the LEDGER, not the table: the table
    // always contains PRUNE entries (declaring a sampling surface is itself a PRUNE decision), so
    // "the table lists a PRUNE clause" would be permanently true and the gate would be welded shut.
    // The mapping is therefore: `no-prune-in-effect` holds <=> pruneExcluded.empty().
    // What does NOT belong here: a value that was never declared at all (BK=16/32 on cm2 are legal
    // but absent from the stock literals).  That is a COVERAGE matter -- a narrower domain, which is
    // exactly what `candidateCoverage == "sampled"` already reports -- not a rule that excluded
    // something.  Counting it here would be the mirror image of "listed but never wired".
    std::vector<std::string> pruneExcluded;
    // What the decision line prints for "which mode is running" (auto(cfg) / off(cfg)): without
    // it the log cannot prove which mode was in effect, and "auto" vs "off" is exactly the
    // difference between "we tried to tune" and "we deliberately did not".
    std::string tuneModeText = "auto(cfg)";
    // "absent" / "loaded" / "negative" / "rejected"
    std::string cacheState = "absent";
    // Cache-growth accounting (t1-schema.md section 1, "pipeline cache growth"): the
    // bytes this process loaded from the persistent pipeline cache.  The "after" reading
    // only exists at the NEXT startup, so a single run can report the "before" half.
    bool hasPipelineCacheBytes = false;
    size_t pipelineCacheBytesBefore = 0;
    // How many rows were handed to the backend.  A COUNT, not the rows themselves: the
    // decision log prints it as the `rows=` field -- naming the member rowCount is what
    // keeps "rows" from reading like a row container.
    size_t rowCount = 0;
    double tuneMs = 0.0;          // wall time of this decision, never mixed into visits/s
    // B1: what the backend actually DID with the rows above, accumulated over the contexts
    // of this device.  The accumulation is deliberately conservative, because the durable
    // set (what step 5 may write) is the rows that installed in EVERY context sharing this
    // file key: `installed` keeps the MINIMUM seen and each drop count the MAXIMUM, so a row
    // that failed anywhere can never be counted as installed everywhere.  Aggregates cannot
    // express the per-row intersection, so this is a sound bound rather than the set itself.
    vk_tune_install_result install {};
    bool installSeen = false;
    // W2 (decision C): the PER-ROW truth, the same order as `rows` (index i describes rows[i]),
    // folded with AND across this device's contexts -- a row is durable only if EVERY context
    // installed it.  🔴 The aggregate fields above can only BOUND this set (min/max), so the WRITE
    // SIDE READS THIS SET: the aggregates keep their old meaning (a conservative upper bound) and are
    // NOT the truth.  Filled by installForHandle, one byte per declared row, zeroed by the backend.
    std::vector<uint8_t> rowInstalled;
    bool rowInstalledSeen = false;
    // W2 (decision B): the contexts that contributed, in install order, by the backend's ordinal.
    // An IDENTITY list, never a count: it is what lets a late install be told "this file does not
    // contain your rows" instead of guessing.
    std::vector<uint32_t> ctxOrdinals;
    // W2 (post-review, 2026-09-20): did ANY context of this device install at least one row?
    // DELIBERATELY an OR fold, NOT the AND fold `install.installed` uses: this answers "does this KEY
    // have an injection slot at all", and ONE context proving that it does is enough.  It gates the
    // NEGATIVE entry, because writing "no-injection-slot" for a key into which this very process
    // installed rows (and hit them) is a self-refuting, sticky, key-scoped verdict -- measured in the
    // first smoke run with the write live, where one key got `wrote 1 row(s)` from one leg and a
    // `NEGATIVE no-injection-slot` from another.  Set in installForHandle; only ever turned ON.
    bool anyRowInstalled = false;
    // W2 (decision E, AMENDED to 2' after source review): the once hook.  Its body lives entirely in
    // vulkantuner.cpp, so this stays DATA: no fifth interface name (R1), no control surface in the
    // backend (R5/R6).
    //
    // 🔴 It takes the Decision BY REFERENCE and is CALLED WHILE THE LOCK IS HELD (the backend's
    // getOutput takes it and calls it inside the same critical section that now also covers
    // installForHandle).  Why not the original `void()` + "take under the lock, call OUTSIDE it, then
    // re-lock": that shape has a SILENT window -- an install arriving between the unlock and the
    // re-lock reads `writeState == NotAttempted`, so its rows are neither written into the file nor
    // reported as late.  Doing the write AND the tri-state in ONE critical section removes the window
    // instead of patching it, and the price is one small file write under the lock, once per device.
    std::function<void(Decision&)> onFirstEval;
    // Set (under the lock) when the hook is TAKEN.  A moved-from std::function is empty exactly like a
    // never-set one, so the two cases need this flag to stay distinguishable: empty + NOT taken = "this
    // context's first eval beat its own install" (LOUD, with ctx=); empty + taken = another thread
    // already consumed it (silent -- N-1 extra lines would be noise, and the late install still has
    // its own loud).
    bool onceHookTaken = false;
    // The resolved policy, copied at decision time so the hook can capture it BY VALUE: the callback is
    // built on the install path (which is what makes "callback empty" mean "install has not run yet")
    // and called from the backend's eval path, which knows nothing about policies.
    TunePolicy tunePolicy;
    // W2 (decision E): what happened at that once point, so a LATE install says the right thing
    // instead of inventing one.  NotAttempted = the hook never ran; Wrote = a file exists and does
    // NOT contain the late context's rows; RefusedByGate = no file was written at all.  "Trigger and
    // consume": a gate refusal is FINAL -- there is deliberately NO retry, because a retry loop would
    // turn one decision into repeated attempts.
    enum class WriteState { NotAttempted, Wrote, RefusedByGate };
    WriteState writeState = WriteState::NotAttempted;
    std::string writeRefusal;     // the gate that refused, when writeState == RefusedByGate
    std::string writeTarget;      // the path the once point wrote (or refused to write), for the log
    // W2 (post-review): what the finalizer concluded, as a VALUE.  The finalizer itself is SILENT (it
    // only computes), which is what makes "one event = one loud line" structural instead of a rule: the
    // conclusion is reported by ONE function, and only when it CHANGES (see reportedVerdict).  The first
    // smoke run with the write live printed every (a)/(b) NOTICE TWICE -- once from the install path and
    // once from the once point -- because the finalizer both computed and printed.
    //   Nothing                      -- nothing to conclude yet (declared == 0): the ordinary production
    //                                   form of step 1b-2b, and never a loud line.
    //   Measured                     -- at least one row is durable; the file (if any) carries rows.
    //   ImportedRejected             -- an IMPORTed row set was all-rejected: (a) an import may not
    //                                   create a durable verdict.
    //   AnyRowInstalled              -- every declared row was rejected HERE, but some context of this
    //                                   device installed a row for this SAME key: (b) the key-scoped
    //                                   marker would be false by our own account.
    //   OurGap                       -- all rejected, but not every reason is a device property (G-D20).
    //   DeviceOnlyAllReject          -- all rejected, every reason a device property: the marker stands.
    //   StaleFromFileNegativeRetired -- a row is durable, so a marker READ FROM A FILE was retired.
    enum class FinalizeVerdict { Nothing, Measured, ImportedRejected, AnyRowInstalled, OurGap,
                                 DeviceOnlyAllReject, StaleFromFileNegativeRetired };
    // The last verdict REPORTED for this decision.  The finalizer runs more than once (once per context
    // install, then once at the once point) and its answer may legitimately CHANGE between those calls,
    // so "report only on a transition" is what makes a repeated line impossible WITHOUT also silencing
    // the second, different conclusion.
    // 🔴 Written only under the owning ComputeContext's tuneMutex -- and that is ONE lock, not two: a
    // Decision lives in ONE (evaluator, device)'s tuneDecisions map (vulkanbackend.cpp:2434-2440), and
    // every thread of that evaluator takes that ctx's tuneMutex for BOTH the install and the once point
    // (vulkanbackend.cpp:2635 / :3067).  No second mutex on the Decision, deliberately.
    FinalizeVerdict reportedVerdict = FinalizeVerdict::Nothing;
    std::string note;             // one short human-readable clause for the log line
  };

  // THE decision entry point.  Returns nullptr when the tuner is off -- including the
  // KATAGO_VK_TUNE=0 case, which must stay a complete no-op except for one loud line if
  // KATAGO_VK_TUNE_IMPORT was also requested (t3-shell-plan.md section 6.1 item 7).
  // Never throws: a failure here must fail open (behave exactly like "no tuned rows") and
  // say so loudly, never take the process down (section 3).
  std::shared_ptr<Decision> makeDecision(
    ggml_backend_t backend,
    int deviceIdx,
    const std::string& homeDataDirOverride,
    int nnXLen,
    int nnYLen,
    int maxBatchSize,
    enabled_t useFP16Mode,        // the mode the BACKEND actually received (not the cfg text):
                                  // F1 showed a False request rewrites the graph as GGML_PREC_F32,
                                  // so this is a request that changes the computation.
    const ModelIdentity& model,
    const TunePolicy& policy
  );

  // Per-context install.  Called once per ComputeHandle, after the backend exists and
  // before the first graph build; idempotent by construction (the context is fresh).
  void installForHandle(const std::shared_ptr<Decision>& decision, ggml_backend_t backend);

  // The ONE row-set file format (t1-schema.md section 1): the cache file, KATAGO_VK_TUNE_IMPORT
  // and KATAGO_VK_TUNE_EXPORT all use it, so a file written by one run can be read by the
  // next, by another machine, and by a hand-written fixture in a test.
  //
  //   VERSION <n>            schema version (refused on mismatch)
  //   TUNER_VERSION <n>      enumerator version (refused on mismatch)
  //   FINGERPRINT <string>   the full device/driver fingerprint (refused on mismatch)
  //   POLICY <string>        policy fingerprint: build revision + policy version + knobs
  //   BASELINE <string>      policy baseline version (bump when a default changes)
  //   MODEL <string>         outer key part 2 (absent = not key-checked)
  //   BOARD <X>x<Y>          outer key part 3
  //   MAXBATCH <n>           outer key part 4
  //   ROWSET_HASH <hex>      hash of the row set, computed from the rows at write time
  //   CREATED <iso8601>      when this file was written (informational)
  //   ROWS <n>               DECLARED row count: a file that lost rows in transit is
  //                          refused instead of being read as a shorter row set
  //   ROW <path> <src0> <src1> <key_m> <key_k> <BM> <BN> <BK> <WM> <WN> <WMITER> <TM> <TN> <BS> <pad> <reshape> <smaller>
  //   NEGATIVE <reason>      this (device, model, S, mb) was DETERMINISTICALLY all-rejected;
  //                          no tuning is attempted.  Never written on a timeout, and never
  //                          written for a partial result.  `<reason>` is one of the CLOSED
  //                          SET declared in vulkantuner.cpp -- a reason outside it refuses the
  //                          file on BOTH channels, because the reason is the part of the claim
  //                          that has to stay comparable across releases.
  //
  // WHICH HEADER KEYS ARE REQUIRED DEPENDS ON THE CHANNEL, and the split follows what each
  // key GUARANTEES (2026-09-13 review):
  //   required on BOTH channels -- POLICY, BASELINE: they carry "the policy changed, so this
  //     row set is void".  A missing one would make that guarantee vanish silently, because
  //     the parser accepts a file that is missing a merely-advisory key.
  //   required on the CACHE channel only -- MODEL, BOARD, MAXBATCH, ROWSET_HASH, CREATED:
  //     a cache is written by us, so there is no legitimate reason for a field to be absent,
  //     and the filename routing alone would otherwise be the only check left.  IMPORT
  //     tolerates their absence (hand-written fixtures must stay possible) and prints
  //     "[not key-checked: ...]" so a human can see what was not verified.
  //   ROWSET_HASH additionally never *refuses* on IMPORT: the parsed rows are hashed live and
  //     the declared value is only a hint there, since a hand-written fixture cannot compute it.
  //
  // ONE FILE = ONE KEY (2026-09-13 decision): the file NAME carries the whole outer key --
  //   <sha256(fingerprint)[0:16]>-<model mvV-shaS>-<X>x<Y>-mb<N>.txt
  // -- while the header's FINGERPRINT/MODEL/BOARD/MAXBATCH stay the AUTHORITY and are still
  // checked item by item.  The name routes, the header verifies: a hand-renamed or
  // cross-machine-copied file is caught by the header, and the two cannot drift because both
  // are built from the same Decision.  Why the key is in the name rather than in per-row
  // columns: with one key per file, "re-tune" is a whole-file overwrite instead of a
  // read-modify-write (the classic way a cache gets corrupted), and only then does a
  // `benchmark -threads 1,2,4` sweep keep every bucket's rows instead of only the last
  // one's.  Precedent: OpenCLTuner::defaultFileName, which the plan says to mirror.
  //
  // Because one file = one key, ROWS and NEGATIVE are mutually exclusive STATEMENTS about
  // that key: the writer refuses to emit both (see formatRowsFileImpl).  The reader still
  // handles a hand-edited file containing both by preferring the rows and saying so loudly.
  std::string formatRowsFile(const Decision& decision);

}  // namespace VulkanTuner

#endif  // NEURALNET_VULKANTUNER_H_
