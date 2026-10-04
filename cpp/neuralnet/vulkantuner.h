#ifndef NEURALNET_VULKANTUNER_H_
#define NEURALNET_VULKANTUNER_H_

// vulkantuner.h -- the CONTROL PLANE of the Vulkan runtime tuner (T3).
//
// WHY THIS FILE EXISTS (t3-shell-plan.md section 9)
// The OpenCL tuner is clean because its measurement can move out of the backend.  Ours
// cannot: a Vulkan pipeline is a runtime object (specialisation constants, an `align`
// side effect on kpad/aligned, a pre-record replay bypass), so the tuner has to live
// inside the process.  What we buy instead is a NARROW interface: the backend exposes
// four ITEMS, five names -- fingerprint / install_rows / probe_candidate / export_state /
// import_state (R1, whose expected set already carries all five) -- and nothing else, and
// every piece of tuner policy -- which candidates
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
// 🔴 `dOut` is the CALLER'S buffer and its layout is DENSE (2c-c): index
// `(shape * nObjects + object) * reps + rep`, one entry per slot of the grid, with the caller's own
// shape and object order.  Slots that were not measured -- the object IS the anchor, or its pipeline
// never got built -- are NaN, never 0: zero is what a tie looks like, and a consumer that ignores the
// marker gets a poisoned interval instead of a quiet tie.  `dCap` must be at least
// nShapes*nObjects*reps or the call is REFUSED (a truncated set would silently shrink the comparison);
// `out->samples` counts the non-NaN entries actually written.  It is an argument and not a member for
// the same reason install_rows' per-row truth buffer is: the entry resets the result struct as a whole.
bool ggml_backend_vk_tune_probe_candidate(ggml_backend_t backend, const vk_tune_row_spec* cands, size_t nCands,
                                          const vk_tune_probe_params* params,
                                          double* dOut, size_t dCap,
                                          vk_tune_probe_result* out,
                                          char* err, size_t errSize);
// (7b) step 4: THE COVERAGE SET'S SHAPES (tune-measurement.md section 3).  One name, no new type: `out`
// receives `vk_tune_probe_shape` entries -- the same type the primitive's params use -- after the
// dmmv-gated shapes have been excluded (R-21), and `counts[i]` is how many graph nodes carried that
// geometry.  `cgraph` crosses as `const void*` so no ggml tensor surface is exported to the shell.  The
// caller sizes the buffer from `ggml_graph_n_nodes`, which is always an upper bound; a too-small cap is
// refused (returns 0) rather than truncated.  It is a HOST walk: nothing is submitted, built or selected.
size_t ggml_backend_vk_tune_graph_shapes(const void* cgraph, vk_tune_probe_shape* out,
                                         uint32_t* counts, size_t cap);

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
  //           never write the cache (since the ONE-cache-form ruling there is no other durable
  //           artifact for it to write).
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
    // 🔴 B3.5 (2026-10-01): `bool negative` USED TO SIT HERE, and so did `negativeReason`,
    // `NegativeOrigin` and `negativeOrigin` -- the in-memory half of the cache's SECOND form ("this key
    // is deterministically all-rejected").  B3 deleted that form from the wire (SCHEMA_VERSION 5 -> 6),
    // which left every one of those fields UNWRITABLE and UNREADABLE (nothing could put one on a file or
    // get one off a file), and B3.5 removed the fields themselves.  What a decision can say is now
    // exactly ONE thing -- the row set below, possibly the EMPTY one -- and `outcome` says how it was
    // reached.  The consequences that used to be carried by those fields are recorded where they
    // mattered: the gate refusal in `setGateRejectMarker` / `rejectBySpikeGate`, the self-falsification
    // guard's changed carrier in `finalizeOutcome`, and the provenance condition in `CONDITION_SPECS`.
    // HOW THIS DECISION WAS REACHED, as one string from the closed set in vulkantuner.cpp
    // (step "3-4").  The default is NOT writable, deliberately: a decision nobody gave an outcome
    // to has by definition not finished, and the failure this field exists to prevent is
    // remembering a timeout / an IO error / a lost device / a run with no winner as if it were a
    // property of the device ("we ran out of time" must never become "this device cannot be
    // tuned").  Only a COMPLETED measurement or a STRUCTURAL all-reject may be written.
    std::string outcome = "running";

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
    // The rows to hand to the backend, in the wire form both sides share.  Parsed from the
    // file at this device's cache path; still refused by the backend's T4 gate today (loudly),
    // and never silently dropped.
    std::vector<vk_tune_row_spec> rows;
    // W3-P1: WHERE THE CANDIDATE SET CAME FROM.  Three states, and the third is the one the write
    // side reads: "none" (nothing was loaded -- the declared set is empty, or the self-test
    // synthesized it), "cache" (a FILE produced the candidates), "measured" (THIS process's
    // own enumeration produced them).  The value is what the `measured-here` condition tests, so the
    // initial value is the fail-closed one: a decision nobody marked can never claim its own set.
    // 🔴 (7a): the third state IS reachable now.  The enumerator left the probe block and became part of
    // the decision point, so `runCandidateCatalog` mints "measured" on every run; and the loader
    // branches below that leave this key with no row set write it too -- there is no file-supplied set,
    // so the FINAL source of the candidate set is this process's own enumeration.  R42(d) counts the
    // setters rather than trusting this sentence: production 1 (the enumerator) + the loader's recording
    // sites + self-test 1.
    // 🔴 B3.5 (2026-10-01): THE LOADER HAD THREE RECORDING SITES AND NOW HAS TWO.  The third was the
    // marker-scope refusal's fallback ("the file's marker does not speak for this build, so its rows are
    // dropped and what is left is this process's own enumeration") -- a branch that could only be reached
    // while a file could carry a marker, and which is deleted with the rest of the read-side marker path.
    // The two that remain are the two the loader really has: no file at all ("absent"), and a file the
    // operator's KATAGO_VK_TUNE_RETUNE threw away whole.  R42's own expectation moved with it (it is a
    // COUNT, so the tool face has to be told -- this is not a site that can be kept for the count's sake).
    //
    // 🔴 THE SEMANTIC, and it is the one thing a reader gets wrong here: this field is the FINAL source
    // ("where the set this decision used came from"), NOT "where it ever came from".  The loader
    // overwriting it is therefore CORRECT -- and that is exactly why "this process searched" needs its
    // own field: on a cache hit the set came from a file, but this process still searched.
    std::string rowSource = "none";   // "none" / "cache" / "measured"
    // (7a) / W4b Stage 1: "THIS PROCESS RAN THE CANDIDATE SEARCH FOR THIS KEY" -- the second, INDEPENDENT
    // fact about the same search, and the one the loader may not answer.  Why it cannot be folded into
    // `rowSource`: that field is the final SOURCE and is overwritten by the load (on a hit it reads
    // "cache" although this process searched exactly as it did on a miss), so it cannot say "we searched".
    // The two are deliberately NOT derivable from one another -- neither is a function of the other.
    //   * EXACTLY ONE SETTER: `runCandidateCatalog`, at the moment it is entered.  The fact is "the search
    //     ran"; whether it then produced a candidate is `rowSource`'s business, and how much it covered is
    //     `candidateCoverage`'s.  R45 censuses the setters (and the loader's silence).
    //   * 🔴 THE LOADER MUST NOT TOUCH IT.  A loader write here would relabel a file-sourced decision as
    //     "this process searched" -- the exact mislabelling the A34/A36 provenance rule exists to prevent.
    //   * The initial value is the fail-closed one: a decision nobody marked cannot claim to have searched.
    bool candidateSearched = false;
    // W4b (⑤ first cut): WHICH SUBSET of the declared domain this process actually looked at, as THREE
    // states rather than a bool.  "who measured it" (rowSource above) and "how much of the domain was
    // covered" are ORTHOGONAL questions -- both must be answered before a durable conclusion may be
    // remembered, because "we sampled 6 presets and none worked" is not "this device has no injectable
    // slot".  The third state is what makes this a three-state field and not a flag: a W3-P1 lesson --
    // a bool here would create the illegal combination "not swept and not sampled" with nobody to refuse.
    //   unknown -- nobody said.  The initial value is the FAIL-CLOSED one: `swept-here` tests for the
    //              string "swept", so a decision nobody marked can never claim a sweep.
    //   sampled -- a sampling surface: the enumerator walked the declared STOCK VECTOR SET (a strict
    //              subset of the Cartesian domain the count-only probe reports), or any other proper
    //              subset.  A legitimate way to search, and NOT a complete one -- which is all this
    //              state has to mean, so it stays true whether the subset is the stock presets or a
    //              chosen sample.
    //   swept   -- every point of the declared domain was put through the predicate (nothing tonight
    //              produces this: 72 > any sane N_max, which w4b-preview section 9 pre-registers as
    //              "sample it").  It exists so the condition has a satisfying direction to test.
    // It is NOT written into the file and does NOT enter the key on purpose: it is a READING about the
    // search, and (7f) made the honest value `swept` by walking the declared domain to exhaustion -- so a
    // coverage header key would be a format event that says nothing.  🔴 THE TRIGGER: the day a run can
    // conclude on a SAMPLING SURFACE and have that conclusion remembered, a coverage field MUST be added
    // in the same batch (t3-shell-plan.md F17).
    std::string candidateCoverage = "unknown";   // "unknown" / "sampled" / "swept"
    // 🔴 B3.5 (2026-10-01): `std::vector<std::string> negativeScope` USED TO SIT HERE -- the kind TOKENS
    // a durable entry spoke for, the third axis of the same honesty question (`measured-here` said WHO
    // produced the set, `candidateCoverage` says HOW MUCH of the domain it looked at, and the scope said
    // WHICH KINDS the conclusion was about).  One file = one key, and a key can carry several kinds (mm
    // today, fa later), so a bare `<reason>` line would have asserted a kind it never measured -- which is
    // why the writer refused an unscoped one.  B3 deleted the line (SCHEMA_VERSION 5 -> 6) and B3.5
    // deleted the field with it.  The KIND REGISTRY itself (`TUNE_KINDS` / `tuneKindKnown`) STAYS: the
    // parser and the candidate catalogue still need it, and the completeness gate still asks it.
    // What the decision line prints for "which mode is running" (auto(cfg) / off(cfg)): without
    // it the log cannot prove which mode was in effect, and "auto" vs "off" is exactly the
    // difference between "we tried to tune" and "we deliberately did not".
    std::string tuneModeText = "auto(cfg)";
    // The loader chain's state: "absent" (no file at this key's path) / "loaded" (the file was accepted
    // and its rows are the answer) / "rejected" (the file was refused -- by the parser or by one of the
    // key comparisons -- so nothing of it is used).  Those are the THREE states the chain can leave.
    // ("retune-ignored" is the branch that never enters the chain: KATAGO_VK_TUNE_RETUNE makes the loader
    // skip the file entirely.  "negative" was a fourth value until B3/B3.5 deleted the second cache form
    // and the field that carried it -- the value is no longer producible and is not named here.)
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
    // ── R-28 step (3): THE MEASUREMENT'S OWN ACCOUNTING ────────────────────────────────────────────
    // Why these live HERE, next to the INSTALL accounting, rather than inside the probe result: the
    // result describes ONE measurement, while a decision is the durable statement made from it -- and
    // the statement's scope depends on facts the result cannot carry (whether a winner exists at all,
    // and how many candidates the caller's enumerator produced).  These are the inputs the
    // three-branch decision reads, so they belong to the thing being decided.
    //
    // 🔴 `pairsComplete` DEFAULTS TO FALSE, and that is the fail-closed direction: it decides whether a
    // "nothing was better" claim may be written, so "nobody told us" must not read as "we measured
    // everything".  It is NOT the same flag as `coverageComplete` (tune-measurement.md section 2.6,
    // "两个旗标别混进一个名字"): that one is the CALLER's statement about the shape set (an input, R-25)
    // and decides whether a whole-slot row is admissible, while this one decides whether the
    // "no gain" claim is.
    bool pairsComplete = false;
    // (7a) / R-25: THE CALLER'S STATEMENT ABOUT THE SHAPE SET -- an INPUT to the measurement primitive
    // (the primitive ECHOES it; it may not derive it, because "looks complete" is the false green the flag
    // exists to prevent).  Here the caller IS the enumerator, since only it knows what set it handed over,
    // so the enumerator is this field's ONE setter and its value is a statement of fact about that set.
    // 🔴 THREE NAMES, THREE QUESTIONS -- deliberately not one flag (tune-measurement.md section 2.6,
    // "两个旗标别混进一个名字"):
    //    coverageComplete  -- is the set I handed over the COMPLETE shape set of this slot?
    //                         (an INPUT, R-25; decides whether a whole-slot row is admissible)
    //    candidateCoverage -- HOW MUCH of the DECLARED DOMAIN the search walked
    //                         ("unknown"/"sampled"/"swept"; decides whether the claim covers a sweep)
    //    pairsComplete     -- did the MEASUREMENT conclude every legal pair?
    //                         (decides whether a "nothing was better" claim may be written)
    // None of the three may be derived from another: different components answer them, at different
    // moments, about different sets.
    // 🔴 DEFAULT FALSE is the fail-closed direction, for the same reason as the line above.
    // 🔴 B4 (2026-10-01): THE DECLARED IMPLICATION HAS NO PRODUCTION PATH TODAY, AND THIS PARAGRAPH IS
    // THE REGISTRATION OF THAT FACT -- measured, not assumed.  The enumerator is this flag's ONLY
    // production setter and it writes `false` EXPLICITLY, with its reason (vulkantuner.cpp, at the
    // catalogue: the walk covers the declared STOCK VECTOR SURFACE, a proper subset of the slot's shape
    // set, so `false` is a measurement of that fact rather than an unfilled default).  The only writer
    // that ever states `true` is the SELF-TEST's fixture.  So the implication stated above -- "complete
    // shape set => a whole-slot row is admissible" -- describes a form NOTHING MINTS YET: `key_m == 0`
    // (the backend's "applies to every shape of this (family, acc)" form) reaches the install path only
    // through a FILE, whose parser accepts it, or through that fixture.  The batch that makes the
    // enumerator walk the WHOLE shape set is the batch that gives this implication its carrier, and it is
    // a BEHAVIOUR + COST change (the enumeration domain widens), which is why it is not this batch.
    // ⚠️ Do not read this as "the flag is unused": it is an INPUT the primitive ECHOES (the measurement
    // line carries `coverage=`), and it is the field a whole-slot row's admissibility is DEFINED by --
    // there is simply no production producer of `true` today.
    bool coverageComplete = false;
    // The DECLARED drop accounting (R-28 step (3)): the reason a candidate LEFT the judgeable set.
    // `CANDIDATE_DROPS[]` in vulkantuner.cpp is the ONE place the name and this field are paired, and it
    // is READ (the suppression loud is built from it), so the name cannot move in one place and be
    // forgotten in the other.
    uint32_t droppedBuildFailed = 0;
    // The decision line's TWO denominators, named so they cannot be confused (section 2.6 requires
    // different names because they have different denominators): `cands=` = candidates MEASURED over
    // candidates ENUMERATED; `pairs=` = legal pairs over shapes.  ⚠️ `cands=0/N` is the honest reading
    // while no measurement has been accounted for (the primitive's production caller is (7b)) -- it must
    // NOT be read as "a complete measurement of nothing".  🔴 (7a) gave the DENOMINATOR a home: the
    // enumerator fills `candidatesEnumerated` from the set it actually returned, so the number on the
    // decision line is the SEARCH's own count rather than a second derivation of it (the numerator stays
    // 0 until (7b), and the two are printed as a pair precisely so that a numerator above its denominator
    // is a wiring bug a reader can SEE).
    uint32_t candidatesMeasured = 0;
    uint32_t candidatesEnumerated = 0;
    uint32_t pairsLegal = 0;
    uint32_t shapesCount = 0;
    // (7b) step 4: WHAT THE GRAPH GAVE US -- the coverage set, read from the built graph at the decision
    // point.  `graphShapes` is the filtered (kept) set in the graph's own order with the dmmv-gated shapes
    // removed (R-21), and `graphNodes` is how many row-table-side MUL_MAT dispatches those shapes carry
    // (the sum of their counts, i.e. the weight the coverage rule will use).  Both are PRINTED on the
    // decision line, which is what makes the new interface's two outputs consumed rather than a parameter
    // nobody reads (R-12), and what lets the smoke pin the set's size instead of trusting it.
    std::vector<vk_tune_probe_shape> graphShapes;
    uint32_t graphNodes = 0;
    // (7b) step 5: THE WEIGHT of each shape in `graphShapes`, same order -- the second output of the
    // enumeration interface.  `graphNodes` is their SUM and stays the decision line's reading; the
    // per-shape vector is what the coverage rule needs ("counts descending until 95% of the weight is
    // covered": tune-measurement.md section 2.2 step 2), because that rule chooses a SUBSET.
    std::vector<uint32_t> graphShapeCounts;
    // (7b) step 5: the enumeration's own CANDIDATES -- the catalogue's geometries, in its order -- and
    // the path they belong to.  The catalogue runs in `makeDecision` and the measurement runs at the
    // END of `installForHandle`, so without these the rows to measure would have to be re-derived
    // there: a second answer to "what did the search return", which is the drift this batch keeps
    // closing.  `candidateGeoms.size() == candidatesEnumerated` is the invariant the two share.
    std::vector<vk_tile_geometry> candidateGeoms;
    int candidatePath = 0;
    // (7b) step 5: the (src0, src1) pair a ROW of that path may declare.  It is deliberately NOT
    // the shape's own pair, and the difference is MEASURED: the backend's cm2 branch has exactly one
    // non-quant tuned-row module ("there is no cm2 f16xf32 module at all", ggml-vulkan.cpp's
    // `ggml_vk_tuned_row_shader_for`), so a cm2 row must say f16 x f16 while a shape may well say
    // f16 x f32 -- the family of a row belongs to its PATH, not to the shape it will serve.  First
    // version declared the shape's pair and every candidate failed to build; the smoke's cm2 fixture
    // row (`ROW cm2 1 1 ...`) is the independent witness that f16 x f16 is the pair that works.
    int candidateRowSrc0 = 0;
    int candidateRowSrc1 = 0;
    // (7b) step 4, first half: THE PRE-CHECK's readings (tune-measurement.md section 2.6's sequence).  Its
    // whole point is to produce `shots` and `reps` for the measurement -- but those are OUTPUTS, and a
    // computation whose output nobody prints is a computation nobody can falsify, so all five travel on
    // the decision line.  `precheckMeasured == false` is the fail-closed state: no shots are invented, and
    // the line says `precheck=none`.
    bool precheckMeasured = false;
    // (7b) step 4 / section 7.11: the pre-check runs at the END of `installForHandle`, which runs once per
    // CONTEXT -- so "did this decision already try" has to be a fact on the decision, or a two-context
    // device would measure twice.
    bool precheckAttempted = false;
    double precheckHotUs = 0.0;     // `t_block_hot`: the median of the K hot stock blocks
    // (B-0) THE AFFINE BLOCK MODEL, measured from two points rather than extrapolated from one: the probe
    // runs at `shots = 1` and at a declared probe count, and `block(s) = fixed + s * marginal` is what
    // turns the 2 ms declaration into a shots value.  Both are printed; a non-positive marginal falls back
    // to the 1-shot block with a loud (fail-closed) instead of inventing a number.
    double precheckFixedUs = 0.0;
    double precheckMarginalUs = 0.0;
    uint32_t precheckShots = 0;     // the inverted time constraint: ceil(min_block_time / per_dispatch)
    uint32_t precheckReps = 0;      // `allocateReps`'s answer for this machine's block-relative scale
    double precheckSigmaRel = 0.0;  // the A/A sentinel's own block-relative paired scale
    // 🔴 (P0, 2026-09-30) THE BUDGET'S TWO OTHER READINGS, so "who decided `reps`" is a printed
    // fact instead of an inference.  `allocateReps` returns three numbers and this struct carried only
    // one of them: the Gaussian formula's own answer (`repsFormula`) and whether that answer was
    // CLAMPED at `REPS_MAX` were written and read by NOTHING (the self-test prints the formula on its own
    // budget line; the clamp flag was printed by no one).  The clamp is not a refusal anywhere in this
    // build -- it is a PRICE, and a dead flag is how a 20x price stayed invisible.
    uint32_t precheckRepsFormula = 0;   // the Gaussian vote (reported, and obeyed whenever > the floor)
    bool precheckClampedAtMax = false;  // true = the vote was cut at REPS_MAX rather than answered
    // (7b) step 5: THE MEASUREMENT (tune-measurement.md section 2.6's main step): one call over the
    // COVERAGE SET, every covered shape against the catalogue's candidates plus the stock anchor and
    // the A/A sentinel.  It lives on the decision for the same reason the pre-check does -- the call
    // happens once per CONTEXT, and "did this decision already measure" has to be a fact, or a
    // two-context device would pay section 2.6's cost twice.
    struct Measurement {
      bool attempted = false;   // the once flag: the gate fired (not "it succeeded")
      bool ok = false;          // the primitive returned a usable comparison (its `OK`)
      std::vector<vk_tune_probe_shape> shapes;   // the COVERAGE set actually measured, in call order
      uint32_t objects = 0;     // that call's nObjects: 2 arms (anchor + sentinel) + candidate rows
      uint32_t reps = 0;        // the budget actually used (the pre-check's own answer)
      uint32_t blocks = 0;      // the primitive's own readings, echoed for the log
      double hotUs = 0.0;       // the block length actually achieved on the heaviest shape (R50's carrier)
      // (7e-3) THE PER-SHAPE RULERS, straight from the primitive's own reading: each shape's anchor median
      // block time, which is the divisor its paired differences were normalised by.  The floor and the
      // bounded-tie rule are decided per shape from these, because one shared ruler was measured to be
      // wrong for six shapes at once.
      std::vector<double> anchorUs;
      uint32_t samplesWritten = 0;
      uint32_t pairsIllegal = 0;
      uint32_t spikes = 0;
      uint32_t droppedBuildFailed = 0;
      bool pairsComplete = false;
      bool coverageEcho = false;   // the enumerator's flag, echoed back by the primitive
      uint32_t shotsMin = 0;
      uint32_t shotsMax = 0;
      uint64_t elapsedUs = 0;
      // Of `candidatesEnumerated`, how many were REALLY measured.  Counted from the NaN discipline
      // below (a finite cell = that candidate ran), never inferred from a count of objects.
      uint32_t candidatesMeasured = 0;
      // THE SAMPLES: dense, `(shape*objects + object)*reps + rep`, NaN = never measured -- the
      // primitive's own answer to "which pairs exist" (it has no side channel, by design).
      // Consumer in THIS batch: `candidatesMeasured` above is counted from it.  The judgement layer
      // of this batch's next segment reads the same buffer for the intervals; keeping it here is what
      // keeps section 2.6's expensive step ONCE per decision rather than once per segment.
      std::vector<double> samples;
    };
    Measurement measurement;
    // P3 (2026-10-01): the SECOND measurement's budget, when the wide-band recovery ran -- 0 means the
    // judgement's intervals all formed on the first pass, which is the only case this machine has
    // produced.  The field exists so that the price of the recovery is on the line a reader greps, and
    // so the outer code can tell "re-measured once, at n'" from "never had to" without re-deriving it
    // from `measurement.reps` (which the second pass overwrites).  Consumer: the judge line's
    // `remeasure=` reading; producer: the recovery block in `installForHandle` (R-12, same batch).
    uint32_t wideBandRemeasure = 0;
    // (7c) THE JUDGEMENT LAYER'S RESULT.  It answers the two questions the decision needs -- "is this
    // comparison trustworthy at all" (the A/A sentinel's EQUIVALENCE test + the spike-rate gate) and
    // "which candidate is not worse than stock on EVERY covered shape" -- and it DECIDES NOTHING about
    // what is written: the three branches and the write are (7d)'s.  `gateRejected` is the one place a
    // judgement may already have changed the decision, and it does so through the landed setter
    // (`rejectBySpikeGate`), never by writing a field from here.
    struct Judgement {
      bool attempted = false;
      bool gateRejected = false;
      // The gate's reading: the i.i.d. Wilson interval it actually read, and the counts it read it
      // from (`objects` is not an input to the interval -- it is the one ingredient a clustering-aware
      // interval will need, so it is printed and the registered gap stays visible).
      double gateLow = 0.0;
      double gateHigh = 0.0;
      uint32_t gateSpikes = 0;
      uint32_t gateBlocks = 0;
      uint32_t gateObjects = 0;
      // The interval work, in pairs: one interval per (covered shape, candidate), and one per covered
      // shape for the sentinel.  `ciMissing` counts pairs whose interval could not be formed at all
      // (`!ciOk`: too few usable pairs) -- that is NO DATA, which the branches answer, and it is
      // deliberately NOT the same field as `winnerUnstable` (the ruling: "a missing interval is not
      // `unstable`").
      uint32_t candidatePairs = 0;
      uint32_t sentinelPairs = 0;
      uint32_t ciMissing = 0;
      // R51/(A): pairs whose differences are ALL exactly zero while the ruler is fine enough to bound them
      // (resolution <= tolerance).  They carry a BOUNDED interval, not a missing one, and this counter is
      // what keeps the two distinguishable on the line.
      uint32_t tiesBounded = 0;
      // (7e-3): `ciMissing` split by cause -- a shape whose block is below the declared floor (its ruler
      // cannot support the claim) versus a band wider than the tolerance (no bound, no data).  Two causes,
      // two readings: that is the whole point of the split.
      uint32_t ciMissingCoarse = 0;
      // (7e-3): the third cause -- fewer than two finite samples, i.e. the primitive never wrote this pair's
      // slot.  It is counted apart because "we did not measure it" must never be read as "we measured it and
      // the ruler was coarse", and because the partition is what makes the counts add up.
      uint32_t ciNever = 0;
      // (R52) THE PAIRING INVARIANT'S READINGS.  `pairsMeasured` is what the judgement found samples in;
      // the primitive's `pairsLegal - nShapes` is what it measured.  Two independently computed sets, so
      // agreement is evidence; the first slot without samples is named because that is the address of a
      // wrong object identity.
      uint32_t pairsMeasured = 0;
      int firstMissingShape = -1;
      int firstMissingObject = -1;
      uint32_t ciMissingWide = 0;
      // P3 (2026-10-01): the WIDEST band observed among those wide pairs, in the same block-relative unit
      // as the tolerance.  `ciMissingWide` says the cause fired; this says HOW FAR outside the tolerance
      // the run's own data was -- and it is the only input of the recovery's `n' = n*(band/tol)^2` that
      // does not come from the pre-check's sample-count-limited sigma.  It stays 0 whenever no pair is
      // wide, which is what this machine has measured so far (`ci_wide=0` on every leg, 2026-10-01).
      double wideBandMax = 0.0;
      // (A6, 2026-10-03) THE MDE, AS ONE QUANTITY RATHER THAN TWO: the WIDEST interval half-width among the
      // CANDIDATE pairs whose interval formed, in the same block-relative unit as the tolerance.  "MDE" and
      // "the half-width of the interval" are two names for one fact, so printing both would read as two
      // different measurements.  The A/A sentinel is deliberately EXCLUDED: it answers "was this comparison
      // comparable" (an equivalence test), while this answers "how fine a difference could this run
      // resolve" -- one number must not answer two questions.  It is a READING with NO threshold:
      // "half-width > tolerance" must never become a write gate (the ruling forbids `unstable => no
      // write`).  0 means no candidate interval formed.
      double ciHalfWidthMax = 0.0;
      // The shapes whose own block is below `MIN_BLOCK_TIME_MS`, and the lightest ruler seen: the floor as
      // a reading rather than a silent gate.
      uint32_t shapesBelowFloor = 0;
      double minAnchorUs = 0.0;
      uint32_t notWorsePairs = 0;
      bool sentinelEquivalent = false;   // EVERY covered shape's sentinel interval inside +-tolerance
      // Did the interval verdicts actually run?  They do now that the primitive hands over the
      // BLOCK-RELATIVE difference (7e-1); the flag stays because a layer that is parked again must never be
      // readable as a verdict -- `verdicts=0` with zero counts is what distinguishes "not evaluated" from
      // "no candidate passed", and the smoke pins both directions.
      bool verdictsComputed = false;
      int winner = -1;                   // the first candidate not-worse on every shape (-1 = none)
      bool winnerUnstable = false;       // the stability boundary, read on that candidate's intervals
      double winnerMargin = 0.0;         // report: the winner's tightest `ciLow + tolerance`
    };
    Judgement judgement;
    // (7d): the branch selection runs ONCE per decision, like the pre-check, the measurement and the
    // judgement.  It is a fact on the decision rather than a local because `installForHandle` runs once
    // per CONTEXT.
    bool branchesConcluded = false;
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
    // have an injection slot at all", and ONE context proving that it does is enough.  It is the input
    // of the finalizer's SELF-FALSIFICATION guard: writing "no-injection-slot" for a key into which this
    // very process installed rows (and hit them) is a self-refuting, sticky, key-scoped claim -- measured
    // in the first smoke run with the write live, where one key got `wrote 1 row(s)` from one leg and a
    // `NEGATIVE no-injection-slot` from another.  Set in installForHandle; only ever turned ON.
    // ⚠️ B3.5 (2026-10-01): the guard it feeds is now a REPORTED verdict only.  While the second cache
    // form existed, `cacheEntryWritable`'s last arm read the marker this field's verdict suppressed, so
    // the guard also REFUSED the write; B3 moved `structural-all-reject`'s effect to `rows`, so that arm
    // no longer decides anything for that row and `anyRowInstalled` does not reach the write gate.  See
    // the consequence note on `finalizeOutcome` -- the field and the verdict stay, the carrier changed.
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
    //   AnyRowInstalled              -- every declared row was rejected HERE, but some context of this
    //                                   device installed a row for this SAME key: the key-scoped claim
    //                                   would be false by our own account (self-falsification guard).
    //   OurGap                       -- all rejected, but not every reason is a device property (G-D20).
    //   DeviceOnlyAllReject          -- all rejected, every reason a device property: the claim stands.
    // 🔴 B3.5 (2026-10-01): `StaleFromFileNegativeRetired` USED TO BE THE SIXTH VALUE -- "a row is
    // durable, so a marker READ FROM A FILE was retired".  A from-file marker cannot be born any more
    // (B3 made the parser refuse a `NEGATIVE` line; B3.5 deleted the field it was carried in), so the
    // verdict had no producer and is gone.  The switch in `reportVerdict` has no `default:`, which is
    // what made removing the enumerator and its case a COMPILE-time pair rather than a silent drift.
    enum class FinalizeVerdict { Nothing, Measured, AnyRowInstalled, OurGap,
                                 DeviceOnlyAllReject };
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
  // KATAGO_VK_TUNE=0 case, which must stay a complete no-op except for one loud line when a
  // request switch that cannot be honoured was set (t3-shell-plan.md section 6.1 item 7).
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
    const TunePolicy& policy,
    // (7b) step 4: the coverage set's shapes and their WEIGHTS, read from the built graph by the
    // backend's ONE walker.  `(shape set, counts)` is the parameter tune-measurement.md section 3.2-3
    // prescribes.  The decision line prints the set's size and the counts' SUM, and the measurement's
    // coverage rule reads the per-shape counts -- so both of the interface's outputs have their
    // consumers in this batch (R-12), one of them being the rule that decides WHICH shapes are measured.
    const std::vector<vk_tune_probe_shape>& graphShapes,
    const std::vector<uint32_t>& graphShapeCounts
  );

  // Per-context install.  Called once per ComputeHandle, after the backend exists and
  // before the first graph build; idempotent by construction (the context is fresh).
  void installForHandle(const std::shared_ptr<Decision>& decision, ggml_backend_t backend);

  // The ONE row-set file format (t1-schema.md section 1): the cache file uses it, so a file
  // written by one run can be read by the next, and a hand-written fixture -- a file dropped at
  // the cache path -- is read by exactly the same parser.  What that does NOT settle: the
  // fingerprint names the device CLASS, not the machine, so a copied file is readable by an
  // identical card; that is an OPEN ITEM registered with the R-29 plan row, not a claim here.
  //
  //   VERSION <n>            schema version (refused on mismatch)
  //   TUNER_VERSION <n>      enumerator version (refused on mismatch)
  //   FINGERPRINT <string>   the full device/driver fingerprint (refused on mismatch)
  //   POLICY <string>        policy fingerprint: build revision + policy version + knobs
  //   BASELINE <string>      policy baseline version (bump when a default changes)
  //   MODEL <string>         outer key part 2 (absent = refused: the name routes, the header verifies)
  //   BOARD <X>x<Y>          outer key part 3
  //   MAXBATCH <n>           outer key part 4
  //   ROWSET_HASH <hex>      hash of the row set, computed from the rows at write time
  //   CREATED <iso8601>      when this file was written (informational)
  //   ROWS <n>               DECLARED row count: a file that lost rows in transit is
  //                          refused instead of being read as a shorter row set
  //   ROW <path> <src0> <src1> <key_m> <key_k> <BM> <BN> <BK> <WM> <WN> <WMITER> <TM> <TN> <BS> <pad> <reshape> <smaller>
  //
  // 🔴 THERE IS NO SECOND FORM (B1's ONE-cache-form ruling, completed by B3/B3.5).  A `NEGATIVE <reason>
  // <kind>[,<kind>...]` line used to be here: "this (device, model, S, mb) was DETERMINISTICALLY
  // all-rejected; no tuning is attempted", whose reason came from a CLOSED SET so it stayed comparable
  // across releases.  That form is gone -- the statement it made ("use the stock configuration
  // everywhere") is now written the only way this file writes anything, as the row set, EMPTY when that
  // is the answer: `ROWS 0` plus its ROWSET_HASH.  A file carrying a `NEGATIVE` line is REFUSED BY NAME
  // (the parser names the removal), which is the same fail-open landing point as any other refusal: the
  // whole key is re-tuned.  SCHEMA_VERSION moved 5 -> 6 for exactly this, and the loader's `cacheState`
  // therefore has no `negative` value to print.
  //
  // WHICH HEADER KEYS ARE REQUIRED (2026-09-13 review; ONE channel left since R-29):
  //   POLICY, BASELINE -- they carry "the policy changed, so this row set is void".  A missing
  //     one would make that guarantee vanish silently, because the parser accepts a file that
  //     is missing a merely-advisory key.
  //   MODEL, BOARD, MAXBATCH, ROWSET_HASH, CREATED, VERSION, TUNER_VERSION, MODE -- a cache is
  //     written by us, so there is no legitimate reason for a field to be absent, and the
  //     filename routing alone would otherwise be the only check left.  R-29 removed the
  //     channel that tolerated their absence, so a missing key is a REFUSAL that names it --
  //     never a note that leaves the check silently undone.
  //   ROWSET_HASH is compared against the hash of the parsed rows; a file that cannot carry a
  //     correct one is refused (tools/rowsethash.py is the independent implementation).
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
  // Because one file = one key, the file carried two mutually exclusive STATEMENTS about that key while
  // the NEGATIVE form existed (`ROWS` / `NEGATIVE`), and the writer refused to emit both.  🔴 B3.5
  // (2026-10-01): with that form deleted there is only ONE statement left to make -- the row set, which
  // may be EMPTY -- so the exclusion, the reader's "prefer the rows and say so loudly" branch for a
  // hand-edited file carrying both, and the fields behind them are all gone.  What remains is what the
  // name already says: `formatRowsFile` emits the durable row set and nothing else.
  std::string formatRowsFile(const Decision& decision);

}  // namespace VulkanTuner

#endif  // NEURALNET_VULKANTUNER_H_
