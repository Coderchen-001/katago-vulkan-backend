// vulkan-tile-validity.h -- pure tile-validity rules for the Vulkan tuner.
//
// WHY THIS HEADER EXISTS (t3-shell-plan.md section 1.7)
// Legality used to live in three places: the rule table in t1-predicate.md, the loader
// check in ggml_vk_tuned_row_check(), and the enumerator in the T3 shell.  That is a
// standing invitation to "the enumerator let it through but the loader rejected it" (or
// the reverse).  The rules are the backend's -- they mirror what the create path and the
// shaders actually accept -- so they live here, dependency-free, and both sides include
// this header:
//
//   * the backend: the KATAGO_VK_MM_TILE=custom: parser
//   * the shell (KataGO side): the candidate enumerator and the IMPORT parser
//
// Deliberately small API (authority order: class, then reason, then canonical key):
//   1. vk_tile_candidate_reason() -- is this concrete geometry legal, and if not, why
//   2. vk_tile_canonical_key()    -- dedup identity (dedup is NOT validity)
//   3. vk_tile_axis_class()       -- per-path axis classification (the AUTHORITY);

//      vk_tile_axis_is_live() is a thin wrapper answering only "may it be swept?"
//
// No vk::, no ggml, no std::string: plain ints in, plain const char* out.  That is what
// lets both sides share it without either one reaching into the other.  (W4e 2c-c adds two
// pure arithmetic helpers at the end -- doubles and uint64_t, still no vk::, no ggml, no
// allocation -- because the measurement primitive's self-test has to drive the same spike
// detector and the same rotation the probe uses; see the section above them.)

#pragma once

#include <stdint.h>

// Which creation path the geometry is destined for: the rules differ per path.
enum vk_tile_path {
    VK_TILE_PATH_SCALAR = 0,  // no cooperative matrix (f16/f32 scalar GEMM)
    VK_TILE_PATH_CM1    = 1,  // VK_KHR_cooperative_matrix
    VK_TILE_PATH_CM2    = 2,  // NV_cooperative_matrix2 (cm2 shader)
};

// Axis ids for vk_tile_axis_class().  These follow the REAL spec layout of mul_mm.comp
// (verified 2026-09-13), not a tuner invention:
//   warptile = { BS, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, WARP }
//     0=BLOCK_SIZE 1=BM 2=BN [3=BK, quant branch only] 4=WM 5=WN 6=WMITER 7=TM 8=TN
//     9=TK ("Only needed for coopmat") 10=WARP
//   11=ALIGNED (host-set, never swept)  12=SHMEM_STRIDE_PAD  13=APPLY_SLM_A_RESHAPE
//     -> note there is NO "10=PAD" slot; PAD is spec 12.
//   cm2's own spec is {BLOCK_SIZE, BM, BN, BK, smaller}, where `smaller` is spec id 4
//   in the shader -- the same number as scalar WM, which is exactly why axis ids must
//   NOT be reused across paths: this enum keeps distinct numbers for distinct meanings,
//   and 14 was chosen for SMALLER because 11 is ALIGNED.
enum vk_tile_axis {
    VK_TILE_AXIS_BS      = 0,
    VK_TILE_AXIS_BM      = 1,
    VK_TILE_AXIS_BN      = 2,
    VK_TILE_AXIS_BK      = 3,
    VK_TILE_AXIS_WM      = 4,
    VK_TILE_AXIS_WN      = 5,
    VK_TILE_AXIS_WMITER  = 6,
    VK_TILE_AXIS_TM      = 7,
    VK_TILE_AXIS_TN      = 8,
    VK_TILE_AXIS_TK      = 9,
    VK_TILE_AXIS_WARP    = 10,
    VK_TILE_AXIS_SPEC_PAD     = 12,
    VK_TILE_AXIS_SPEC_RESHAPE = 13,
    VK_TILE_AXIS_SMALLER = 14,   // cm2 only (spec id 4 in the shader)
};

// What an axis means on a given path.  Two-state "live/dead" was not enough: an axis
// can be a free enumeration axis, a value that must be filled in correctly but must not
// be swept, a derived quantity, or simply absent on that path.
enum vk_tile_axis_class {
    VK_TILE_AXIS_INERT        = 0,  // does not exist on this path (sweeping it is a no-op)
    VK_TILE_AXIS_ENUMERABLE   = 1,  // free axis: may be swept
    VK_TILE_AXIS_PINNED       = 2,  // must be set correctly, must NOT be swept
    VK_TILE_AXIS_DERIVED      = 3,  // computed from other axes
    VK_TILE_AXIS_NOT_APPLICABLE = 4 // not an axis of this path at all
};

// One concrete tile geometry.  Scalar / cm1 use the warptile fields; cm2 uses
// BM, BN, BK, BS (BLOCK_SIZE) and `smaller`.  Unset spec constants are -1.
struct vk_tile_geometry {
    uint32_t BM, BN, BK, WM, WN, WMITER, TM, TN, BS;
    int pad;      // -1 = unspecified (shader default applies)
    int reshape;  // -1 = unspecified
    int smaller;  // -1 = unspecified (cm2 only)
};

// The wire form of ONE tuned row: identity + geometry, and nothing else.  No handles and
// no pointers (t3-shell-plan.md section 7.1-3): the backend resolves the geometry into
// pipelines when the row is installed, so a row is something the cache can store verbatim
// and the shell can hand across the interface without either side knowing the other's
// types.  It lives in this header so both sides share ONE definition of a row -- two
// definitions of a wire format is exactly the drift this header exists to prevent.
//
// `src0_type`/`src1_type` are ggml_type values carried as int on purpose: this header
// knows nothing about ggml, and the backend is the only side that has to interpret them.
// `key_m`/`key_k` are 0 for "whole slot" (matches any shape), which is the v1 form; see
// ggml_vk_tuned_row_matches for the matching rule.
struct vk_tune_row_spec {
    int src0_type;
    int src1_type;
    uint32_t key_m;
    uint32_t key_k;
    int path;                  // vk_tile_path
    struct vk_tile_geometry geom;
};

// ---------------------------------------------------------------------------
// B1 (t3-shell-plan.md section 6.7): install-time outcome reporting.
//
// A row set is USER DATA (a cache file or KATAGO_VK_TUNE_IMPORT), and the loader has
// to be able to say "declared N, installed M, dropped for this reason".  Two reasons
// why a single drop counter is not enough:
//
//   1. "every row in the file is hit at least once" is the acceptance for "no row was
//      tuned for nothing", and its denominator must be M (the rows that installed): a
//      row that cannot install can never be hit, so grading against N would make that
//      leg red for a reason that has nothing to do with tuning.
//   2. the write side (step 5) must write only rows this process really installed, so
//      the reason is what tells a transient failure from a permanent one.
//
// The reason set is closed on purpose: these counts reach the log and (later) the
// durable-effect decision, so free text must not be able to enter them.
enum vk_tune_row_drop {
    VK_TUNE_ROW_DROP_NONE         = 0,
    VK_TUNE_ROW_DROP_ILLEGAL      = 1,  // vk_tile_candidate_reason() refused it
    VK_TUNE_ROW_DROP_UNSUPPORTED  = 2,  // family / path / capability (I6) not usable
    VK_TUNE_ROW_DROP_BUILD_FAILED = 3,  // the pipeline could not be created
};

// Row-level counts, and they are an EQUALITY:
//
//     installed + dropped_illegal + dropped_unsupported + dropped_build_failed == declared
//
// A row is counted as installed when it landed on at least one non-empty target object
// (union, so a row that is usable is never thrown away), and it is counted as dropped only
// when it landed NOWHERE.  That is what keeps the three numbers addable: an earlier shape
// counted `installed` per row but the drops per TARGET, so one row whose two accumulator
// variants failed differently could push installed + dropped past declared.
//
// The per-target detail lives in two fields that are deliberately OUTSIDE that equality
// (they describe one row twice, they do not partition the declared set):
//
//   * partial        -- rows that installed, but not on every non-empty target (a SUBSET
//                       flag of `installed`, not a third bucket: a third bucket would
//                       reopen the "which side does it fall on" question)
//   * variant_failed -- (row, target) build failures
//
// Why "installed" is a union and the authority for "was it really used" is the HIT, not any
// install-time predicate: the accumulator variant is chosen PER NODE from dst->op_params[0],
// and KATAGO_VK_MMPREC_F32 (a policy knob, i.e. a legitimate production configuration) can
// rewrite the whole graph or a subset of nodes (vulkanbackend.cpp:2748-2758), so both
// variants may be in use inside one process and no predicate evaluated at install time can
// name "the" required one.  A dead row -- installed on a variant production never resolves,
// or on a variant that is empty while the other is not -- is what the per-row hit assertion
// catches.
// 🔴 OWNERSHIP (W2): EVERY field of this struct is written by the BACKEND.  Anything the CALLER owns
// -- today the per-row truth buffer -- is an ARGUMENT of the install call, never a member: the
// install entry resets this struct as a whole (`*result = vk_tune_install_result {}`), and that
// reset has to stay correct for every field added later.  A member the backend does not own would be
// silently cleared there (measured: adding `row_installed` as a member would have been wiped at
// ggml-vulkan.cpp:21143 before a single byte could be written), so the OWNERSHIP BOUNDARY IS THE
// TYPE/ARGUMENT SPLIT -- not a comment somebody has to remember.
struct vk_tune_install_result {
    unsigned declared;
    unsigned installed;
    unsigned dropped_illegal;
    unsigned dropped_unsupported;
    // G-D20: the subset of `dropped_unsupported` whose reason names OUR implementation gap.  The
    // class table lives in ggml-vulkan.cpp and decides by the reason STRING; a reason it does not
    // know counts here too (the fail-closed direction, like the loader's unknown-outcome rule).
    // Same shape as `partial` above: it describes a member of the equality without entering it,
    // so `installed + sum(dropped_*) == declared` stays exactly what it was.
    unsigned dropped_unsupported_our_gap;
    unsigned dropped_build_failed;
    unsigned partial;          // subset of `installed` (see above)
    unsigned variant_failed;   // (row, target) build failures (see above)
    // W2: the context's ordinal (ggml-vulkan.cpp's `tune_ctx_ordinal`, assigned once at context
    // construction).  The shell records WHICH contexts contributed to a written file -- an IDENTITY,
    // never a count -- so the file can say "written by ctx 0,1" and a context that arrives after the
    // write can be told "this file does not contain your rows" without anybody guessing.
    unsigned ctx_ordinal;
};

// ---------------------------------------------------------------------------
// W4e (t3-shell-plan.md section 9.1's `probe_candidate`): the MEASUREMENT primitive.
//
// It measures BLOCKS -- one shape x one object x `shots` dispatches, timed as one graph_compute plus
// one synchronize -- and returns the paired differences that the judgement layer consumes
// (cpp/neuralnet/tunestats.h).  It is the PRODUCER of the differences and it decides NOTHING: the
// non-inferiority interval, the A/A equivalence test, the stability boundary and the budget all live
// in tunestats.h, which is pure and device-free.
//
// 🔴 THE ONE HARD PRECONDITION (tune-measurement.md section 2.4): it must run BEFORE `pr_recording` is
// first set.  From then on graph_compute takes the replay branch and returns early -- so the
// measurement would be silently skipped -- and any block it did submit would be recorded into the
// USER's graph.  A violated precondition REFUSES; it never "measures as well as it can" (fail-closed,
// like every other gate in this file).
//
// 🔴 OWNERSHIP: every field of vk_tune_probe_result is written by the BACKEND, and the caller's sample
// buffer is an ARGUMENT rather than a member -- the entry resets the result as a whole, so a member the
// backend does not own would be silently cleared.  Same boundary as vk_tune_install_result.
struct vk_tune_probe_shape {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    // The FAMILY, and it is not decoration: the two tensors a block dispatches have types, and
    // a row is keyed by (family, ...), so a shape without its family cannot be turned into a
    // graph at all.  Same transcoded vk_tune_src_type numbers used elsewhere in this header.
    int src0_type;
    int src1_type;
};

// One measured pipeline.  `stock` means the backend's OWN choice for that shape -- the anchor every
// candidate is compared against inside the same rep; otherwise `rowIndex` indexes the caller's
// candidate array.  Two objects may point at the SAME pipeline on purpose: that is what the A/A
// sentinel is (tune-measurement.md section 2.6's `n_sentinel = 2`).
struct vk_tune_probe_object {
    bool stock;
    uint32_t rowIndex;
    bool aligned;   // which of the row's two variants.  The caller passes the SAME align rule the
                    // dispatcher uses, or the number measured is not the number that takes effect
                    // (tune-measurement.md section 2.6's K1 same-source requirement).
};

struct vk_tune_probe_params {
    const vk_tune_probe_shape* shapes;
    uint32_t nShapes;
    const vk_tune_probe_object* objects;
    uint32_t nObjects;
    // The SAME `shots` for every object in one comparison: per_dispatch = block/shots carries a
    // 1/shots submission term, so a differing `shots` biases the comparison instead of lengthening it.
    uint32_t shots;
    uint32_t warmupBlocks;   // per object (ruled): warming only the shape would be an inference about
                             // every other object
    uint32_t reps;           // R, the DECLARED budget -- it is never adjusted by looking at the data
    uint32_t stallBlocks;    // stall detection: consecutive blocks with no measurable progress
    // `coverageComplete` is an INPUT, not an answer: only the caller knows whether the set it handed
    // in is the slot's WHOLE shape set (its enumerator built that set).  The primitive echoes it into
    // vk_tune_probe_result, so producer and result cannot disagree, and it must never derive it --
    // "the set looks complete" is the false green this flag exists to prevent.
    bool coverageComplete;
    // Per-shape `shots` (section 2.2 allows the shapes to differ): one entry per shape, or null with
    // nShotsPerShape == 0 to use the single `shots` above for every shape.  The reason it must exist
    // is that a block shortened by VRAM does not carry the quantization argument of a full-length
    // block (that section's consequence (1)), so a heavy shape has to be able to shorten ITSELF --
    // with one global value the caller can only shorten every shape by proxy.
    const uint32_t* shotsPerShape;
    uint32_t nShotsPerShape;
    // ---- W4e 2c-c: the two things the block sequence and the spike detector need, handed over ----
    // Both are DECLARED constants of the judgement layer (cpp/neuralnet/tunestats.h) and arrive as
    // arguments on purpose: the backend must not carry its own copy of a number the gate was measured
    // with (tune-measurement.md section 2.3.1 -- one source), and a value that only the producer reads
    // is exactly the kind of field that reaches production with nothing able to change it.
    //
    // `seed`: the randomized Latin square that orders the block sequence is reproducible only if its
    // seed is declared, and "which order the blocks were submitted in" is part of what a measurement
    // MEANS (a per-rep position effect is not common mode unless the order rotates).
    uint64_t seed;
    // `spikeBandLo`/`spikeBandHi`: the amplitude band of the spike detector, relative to THAT OBJECT's
    // median -- the envelope generator's own definition, so a spike means the same thing at runtime as
    // it did when the 0.15 gate was measured.  The probe refuses a band that is not a usable interval,
    // because a silently-passed-through band would change the reported rate without changing anything
    // a reader could see.
    double spikeBandLo;
    double spikeBandHi;
    // `tooSlowW`: the `too-slow` drop ratio, handed in for the same one-source reason as the band above.
    // A CANDIDATE whose median block on a shape is more than `tooSlowW` times that shape's own stock
    // anchor median leaves the JUDGEABLE set (see `dropped_too_slow` below).  The declared value lives in
    // cpp/neuralnet/tunestats.h (TOO_SLOW_W, the same number and meaning as the estimate's `W_cand`), and
    // the probe REFUSES anything <= 1: "not slower than the anchor" would discard nearly every candidate,
    // and the exclusion rather than the measurement would decide the comparison.
    double tooSlowW;
};

// The runtime spike rate is the fraction of blocks the amplitude-band detector marks.  The band is the
// envelope generator's OWN definition (tunestats.cpp's ENV_CASES: a spike is +40..100% against that
// object's median), REUSED rather than invented, because the 15% gate was measured with that
// definition -- a different detector would make 0.15 a different quantity.
enum vk_tune_probe_status {
    VK_TUNE_PROBE_OK = 0,
    VK_TUNE_PROBE_REFUSED_PRECONDITION,   // section 2.4 interference check, fail-closed
    VK_TUNE_PROBE_STALLED,                // stallBlocks consecutive blocks with no progress
    VK_TUNE_PROBE_NOT_MEASURED,           // ran, but produced no usable comparison
};

struct vk_tune_probe_result {
    enum vk_tune_probe_status st;
    uint32_t shapes;
    uint32_t objects;
    uint32_t blocks;         // blocks actually timed (warmup excluded)
    uint32_t samples;        // paired samples written into the caller's buffer
    // The DENOMINATOR, decided by the same predicate the production lookup uses
    // (ggml_vk_tuned_row_matches): a (shape, candidate) pair whose routing key does not serve
    // that shape is NOT an error and NOT a refusal -- the row simply would never be picked for
    // it, so measuring it there would report another pipeline's number.  Such a pair enters
    // neither the measurement nor the denominator, and an empty denominator must never be read
    // as "complete" (tune-measurement.md section 2.6).
    uint32_t pairs_legal;
    uint32_t pairs_illegal;
    uint32_t spikes;         // blocks the amplitude-band detector marked
    // Pairs the POST-HOC `too-slow` exclusion removed from the judgeable set (w4e-preview.md R-27):
    // every legal pair ran its full budget FIRST -- the grid is never shortened, which is what separates
    // this from section 2.6's early-stop form -- and this count is subtracted exactly once:
    // judgeable = pairs_measured - dropped_too_slow, which is what `pairs_complete` compares against
    // `pairs_legal`.  Each dropped pair also prints its own line carrying the two medians and their
    // ratio: a READING, so that the decision can be re-derived from the log, while the decision itself
    // stays the bare comparison.  It is a THIRD bucket alongside `dropped_build_failed`, and it is
    // reported instead of being folded into the flag, so "nothing was slow" cannot be confused with
    // "pairs were excluded and the run therefore cannot claim no-gain".
    // The dropped pair's cells in the caller's buffer are deliberately NOT re-accounted, and that is
    // what keeps the pinned grid intact (blocks and samples do not move).  Section 2.6's "the dropped
    // candidate's measured blocks enter only the log" was written for the early-stop form, where it
    // means "do not keep the PARTIAL blocks measured before the drop"; post hoc there are no partial
    // blocks -- the whole grid was paid for and measured -- so what changes is which pairs may be judged.
    uint32_t dropped_too_slow;
    uint32_t dropped_build_failed;
    bool pairs_complete;     // every legal pair of every covered shape has a full budget
    // ⚠️ This is the CALLER's own statement, passed through UNCHANGED: the primitive never verifies
    // it, and a reader must not read "true" as "somebody checked that the set is complete".  The
    // completeness judgement lives in the enumerator that built the set (see the INPUT field in
    // vk_tune_probe_params) -- the two are one fact, not two.
    bool coverage_complete;
    // What was ACTUALLY used per run: the range of the per-shape `shots` (equal when one value was
    // passed).  The judgement layer divides each block by ITS OWN shape's shots, so a consumer that
    // cannot see this cannot compute a per-dispatch time at all.
    uint32_t shots_min;
    uint32_t shots_max;
    double t_block_hot_us;   // the estimate's input: median of the hot blocks of the heaviest shape
    uint64_t elapsed_us;
};

// ---------------------------------------------------------------------------
// W4e 2c-c: the two DEFINITIONS the measurement primitive shares with its self-test.
//
// Why they are functions in this header instead of five lines inside ggml-vulkan.cpp: the probe's
// self-test leg runs in the SHELL (cpp/neuralnet/vulkantuner.cpp) and has to drive the very rule the
// probe applies.  "Feed the detector a series with a known rate and check it comes back" is a claim
// about the produced rate only if both sides call ONE function; a second copy would make the leg
// verify the copy.  Neither function touches the device, allocates, or knows about vk:: -- they are
// arithmetic on plain numbers, which is what this header is for.

// One spike mark.  The band is relative to THAT OBJECT's median and both edges are inclusive: the
// generator draws its delay as 1 + U[lo,hi), so the inclusive upper edge is a measure-zero difference
// and stating the convention beats leaving it to whoever reads a `<` next.  The VALUES are arguments
// because the declared band lives in the judgement layer (tunestats.h SPIKE_BAND_LO/HI) and the probe
// receives it from its caller -- see vk_tune_probe_params.
static inline bool vk_tune_spike_band_marked(double t, double median_t, double lo, double hi) {
    return t >= median_t * (1.0 + lo) && t <= median_t * (1.0 + hi);
}

// The rotation.  The block sequence visits every (shape, object) slot once per rep, and the ORDER is a
// randomized Latin square rather than a fixed walk (tune-measurement.md section 2.2: the granularity of
// the rotation is (object x shape); the deterministic circular shift that used to do this was replaced
// because a fixed order leaks a position effect -- what the previous object left in cache/clocks -- into
// the paired difference, and that term is not common mode).
//
// Two properties are the whole point, and both are asserted by the shell-side leg:
//   * BALANCE: inside one cycle of `n` passes, every index occupies every position exactly once (a
//     rotation of one base permutation), so a position effect cancels over the cycle instead of
//     accumulating -- which randomization alone does not achieve at the rep counts we can afford;
//   * REPRODUCIBILITY: the base permutation is a Fisher-Yates shuffle driven by an integer-only stream
//     derived from (seed, cycle).  Integer-only and allocation-free on purpose: the standard library's
//     engines are allowed to differ between toolchains, and an order that differs between two builds of
//     the same source is a measurement difference no header could explain.
//
// `perm` is caller-provided scratch (n entries): the call sites are a measurement loop and a self-test,
// and neither should pay for a container to ask "which index is next".
static inline uint64_t vk_tune_mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static inline void vk_tune_rotation_permutation(uint64_t seed, uint32_t n, uint32_t cycle, uint32_t* perm) {
    for (uint32_t i = 0; i < n; i++) {
        perm[i] = i;
    }
    uint64_t s = vk_tune_mix64(seed ^ ((uint64_t) cycle * 0x9E3779B97F4A7C15ULL));
    for (uint32_t i = n; i > 1; i--) {
        s = vk_tune_mix64(s);
        const uint32_t j = (uint32_t) (s % (uint64_t) i);
        const uint32_t tmp = perm[i - 1];
        perm[i - 1] = perm[j];
        perm[j] = tmp;
    }
}

// Which index sits at position `k` of pass `phase`.  The rotation IS the balance mechanism: the same
// base permutation, shifted by the pass index, so index `i` is at position `(i + phase) % n`.
static inline uint32_t vk_tune_rotation_at(const uint32_t* perm, uint32_t n, uint32_t phase, uint32_t k) {
    if (n == 0) {
        return 0;
    }
    return perm[(k + phase) % n];
}


//
// Same sharing rule as vk_tile_candidate_reason: the enumerator and the loader MUST
// answer this identically, because a candidate the loader would always drop is a
// permanent loud line per run -- i.e. "the enumerator proposed something that can
// never install" is a bug in the enumerator, not a device property.
//
// The ggml_type numbers are transcribed deliberately (this header cannot include
// ggml): GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1.  The backend asserts both values at
// compile time (see ggml_backend_vk_tune_install_rows), so a renumbering upstream
// becomes a build error instead of a wrong-family install.
enum vk_tune_src_type {
    VK_TUNE_SRC_F32 = 0,
    VK_TUNE_SRC_F16 = 1,
};

// Returns nullptr when a row with this (family, path) has an implementation, else a
// static reason naming the missing piece.  `path` is only checked for existence here;
// whether this DEVICE can use that path is capability (I6), which is the backend's to
// answer at install time -- the shell must not guess it from the fingerprint.
// ONE authority for "is this (src0, src1) family wired at all".  This predicate and the backend's
// object lookup (ggml_vk_tuned_row_targets) used to answer that question with two complementary
// spellings; adding a family to one and not the other made the other's zero-return unreachable --
// a branch no input could reach.  The OBJECT mapping stays in the backend (it needs the device),
// so this owns the SET only, and the backend derives its early-out from it.
static inline bool vk_tune_row_family_wired(int src0_type, int src1_type) {
    return src0_type == VK_TUNE_SRC_F16
        && (src1_type == VK_TUNE_SRC_F16 || src1_type == VK_TUNE_SRC_F32);
}

static inline const char * vk_tune_row_family_reason(int src0_type, int src1_type, int path) {
    if (!vk_tune_row_family_wired(src0_type, src1_type)) {
        return "no tuned-row implementation for this (src0, src1) family "
               "(only f16xf16 and f16xf32 are wired)";
    }
    // cm1 was refused here until cm1-b2.  What justified the refusal was a MEASUREMENT:
    // the geometry table accepted scalar-shaped rows for cm1, and the shader declares its coopmat
    // types with the ROW's own TM/TN/TK (mul_mm.comp:272-277 -- cache_a/cache_b/sums, fed by the
    // `for (...; i += TK)` k-loop at :312), so an accepted but
    // unsupported fragment shape made the driver abort with STATUS_INTEGER_DIVIDE_BY_ZERO.
    // The repair is in the two places the crash actually lived: vk_tile_candidate_reason now
    // has a COOPMAT clause set for cm1 (the scalar inner-loop rules do NOT describe it -- the
    // stock cm1 buckets carry wniter == 0 and compile), and I6 refuses a row whose fragment
    // shape is not this device's cooperative-matrix shape.
    if (path != VK_TILE_PATH_SCALAR && path != VK_TILE_PATH_CM1 && path != VK_TILE_PATH_CM2) {
        // ONE spelling for ONE condition (G-D26).  ggml_vk_tuned_row_capability_reason answers this
        // same question with the SAME string, and its class lives in VK_TUNE_CAPABILITY_CLASSES.
        // Two spellings made that table row unreachable: the install loop asks THIS function first,
        // so the capability function never saw an unknown path -- a class row no reason could match.
        return "path is neither scalar, cm1 nor cm2";
    }
    return nullptr;
}

// The ONE place that names where a wave/subgroup size comes from.  On the scalar path stock fills
// the warptile's WARP field with subgroup_size_8 = max(device->subgroup_size, 8)
// (ggml-vulkan.cpp:4735-4742, used at :4823-4825 and :5854-5855).  The row build and the custom
// probe must put the SAME value into their warptile, so that the predicate, the build and stock
// agree by construction; writing max(subgroup_size, 8) at each call site is how two truths are born.
static inline uint32_t vk_tile_wave_for(uint32_t device_subgroup_size) {
    return device_subgroup_size > 8 ? device_subgroup_size : 8;
}

// Returns nullptr when the geometry is legal, otherwise a static reason string naming
// the first violated clause.  Callers log it as-is, which is what makes the enumerator
// and the loader produce the SAME reason for the same bad input.
static inline const char * vk_tile_candidate_reason(int path, const vk_tile_geometry & g, uint32_t wave) {
    // The wave (subgroup) size is an INPUT, never a constant: on a wave64 device the stock scalar
    // warptiles set the warptile's WARP field to 64 (subgroup_size_8 = max(device->subgroup_size, 8),
    // ggml-vulkan.cpp:4735-4742, used at :4823-4825 and :5854-5855), so a hardcoded 32 here would
    // refuse every legal wave64 geometry and accept illegal ones -- and for BS "refused" is the
    // silent-wrong-answer direction.  Callers pass vk_tile_wave_for(device->subgroup_size).  Zero is
    // a caller-contract violation and is refused rather than divided by (fail-closed, same style as
    // the clause list below: an unknown input is never "legal").
    // wave == 0 is refused for EVERY path, including cm2.  cm2's spec (BLOCK_SIZE/BM/BN/BK/smaller)
    // has no wave semantics at all, so this is a deliberate over-refusal: the wave is a property of
    // the DEVICE, and "this device's wave is unknown" means the row cannot be attributed to a device
    // class yet -- refusing it is the safe direction (the row is dropped loudly, not judged wrongly).
    //
    // Fail-closed vs fail-open, side by side, so nobody reads them as a contradiction:
    //   * the JUDGEMENT side (here) is fail-closed -- an unknown input is never "legal";
    //   * the FUNCTION side is fail-open -- the shell refuses to tune at all (no rows injected) and
    //     the production path keeps the stock heuristic.
    if (wave == 0) {
        return "wave == 0 (the device's wave/subgroup size must be known before validating)";
    }
    if (path == VK_TILE_PATH_CM2) {
        // cm2's spec is {BLOCK_SIZE, BM, BN, BK, enable_smaller_matrices}
        // (mul_mm_cm2.comp:31-36).  The scalar warptile clauses below do NOT describe
        // it, so this branch stands alone.  Bounds beyond these sanity checks (BK
        // sweep verdict, bucket membership) belong to the candidate table, not here.
        if (g.BM == 0 || g.BN == 0 || g.BK == 0 || g.BS == 0) {
            return "cm2: zero dimension (BM/BN/BK/BLOCK_SIZE must be > 0)";
        }
        if (g.BS > 1024) {
            return "cm2: BLOCK_SIZE > 1024";
        }
        if (g.smaller >= 0 && g.smaller != 0 && g.smaller != 1) {
            return "cm2: smaller not in {0,1}";
        }
        if (g.smaller == 1 && (g.BN % 2) != 0) {
            return "cm2: smaller=1 needs an even BN (effective BN = BN/2)";
        }
        return nullptr;
    }

    // Scalar / cm1 warptile family.  Clauses transcribed from the
    // KATAGO_VK_MM_TILE=custom: parser (same accept/reject set; the leading zero check
    // also removes a divide-by-zero the original expression had on WM/WN == 0).
    if (g.BM == 0 || g.BN == 0 || g.WM == 0 || g.WN == 0 || g.WMITER == 0 || g.TM == 0 || g.TN == 0) {
        return "zero dimension (BM/BN/WM/WN/WMITER/TM/TN must be > 0)";
    }
    const uint32_t num_warps = (g.BM / g.WM) * (g.BN / g.WN);
    if (g.BM % g.WM != 0)               return "BM % WM != 0";
    if (g.BN % g.WN != 0)               return "BN % WN != 0";
    if (g.WM % g.WMITER != 0)           return "WM % WMITER != 0";
    if (g.BS != num_warps * wave)       return "BS != num_warps * wave (BS is derived, not free)";
    if (g.BS > 1024)                    return "BS > 1024";
    if (g.pad >= 0 && g.pad > 4)        return "PAD outside [0,4]";
    if (g.reshape >= 0 && g.reshape != 0 && g.reshape != 1) return "RESHAPE not in {0,1}";

    // The two paths diverge HERE, and the divergence is the point (cm1-b2).  The clauses below
    // the cm1 branch describe the SCALAR inner loop, which the COOPMAT path does not run:
    // mul_mm.comp computes WNITER and WSUBN in BOTH paths (:178-180), but every USE of them is on
    // the `#else` (scalar) side: WNITER at :280, :289, :338, :372, :442 and WSUBN at :341-344,
    // :446.  The COOPMAT branch (:182-192) works from cms_per_row / cms_per_col / storestride
    // instead.  And the stock cm1 buckets carry WNITER == 0 while compiling and running every
    // day, so "wniter >= 1" (and the divisibility clauses around it) are rules about SCALAR, not
    // about cm1.
    //
    // 🔴 FRAGILE BY CONSTRUCTION -- read this before changing the shader: WNITER == 0 is safe for
    // cm1 ONLY because nothing on the COOPMAT side uses WSUBN or WNITER, so `WN / WNITER` (:180)
    // is dead code there and the compiler drops it.  If upstream ever uses WSUBN or WNITER inside
    // a COOPMAT branch, the "wniter >= 1" rule MUST come back for cm1 -- and so must WMITER as a
    // real axis: it reaches this path ONLY through WSUBM/WNITER, so cm1 currently enumerates an
    // axis the COOPMAT shader never reads (every other WMITER use is on the `#else` side too).
    // What backs the clause above is an INFERENCE plus a daily-running fact -- the stock cm1
    // buckets carry wniter == 0 and compile and run, i.e. the backend drops the dead
    // `WN / WNITER` -- NOT a measurement of this hazard: every cm1 row the old table ACCEPTED
    // had wniter >= 1 (2 and 8 in the two crashing fixtures), so that division never executed.
    // The cm1-b crash was a DIFFERENT clause: the fragment shape, now refused by I6 and pinned
    // by the smoke leg A3k.
    //
    // What cm1 DOES need is the array arithmetic the COOPMAT branch really performs:
    // cms_per_row = WM / TM, cms_per_col = WN / TN, storestride =
    // WARP / TM, and the type declarations coopmat<..., TM, TN|TK, ...> (:272-277) -- all of
    // which want a non-zero fragment per warp, i.e. WM >= TM and WN >= TN.
    // The FRAGMENT SHAPE ITSELF (TM/TN == this device's cooperative-matrix shape) is a DEVICE
    // property, so I6 refuses it, not this table: the table must stay answerable without a
    // device (the shell parses rows with it, and the fingerprint carries no coopmat shape).
    if (path == VK_TILE_PATH_CM1) {
        // BK on the wire is SYNTAX ONLY here: the f16 COOPMAT variant compiles `#define BK 32`
        // (mul_mm.comp:113-115) and has no constant_id=3, so the row's BK is not read at all.
        // Both spellings stay legal because {16,32} is what the wire and the stock cm1 warptiles
        // actually use, and refusing 16 would refuse rows that are fine.  (cm2 is the opposite
        // case: that spec really is {BLOCK_SIZE, BM, BN, BK, smaller}.)
        if (g.BK != 16 && g.BK != 32) {
            return "cm1: BK outside {16, 32} (wire syntax only: the f16 coopmat shader compiles "
                   "#define BK 32 and does not read the row's BK)";
        }
        // PAD / RESHAPE, the two cm1-only spec constants (12/13).  ⚠️ This note is stated ONCE, in
        // the present tense, because the version that stood here mixed a cm1-c-era sentence with the
        // cm1-d update and ended up contradicting the code two lines below it (2026-09-24).
        //
        // TODAY: `-1` ("unspecified") is legal for both; the row's values DO reach the spec
        // (ggml-vulkan.cpp:21035 passes them into the spec builder, 20868 is the Intel refusal), so
        // any value the range checks admit is a value the pipeline really runs with -- which is why
        // the generic checks above (`PAD ∈ [0,4]`, `RESHAPE ∈ {0,1}`) are the whole rule here.
        // `0` is NOT a synonym for "default" on PAD: it is the Intel-proprietary stride, 16+0
        // instead of 16+4, i.e. a real (and different) layout.
        //
        // 🔴 RESHAPE is additionally not row-overridable any more (2026-09-24, runtime-tuner-spec
        // 5.3.1/5.3.2): it selects WHICH shared-memory index formula is compiled in
        // (`mul_mm_funcs.glsl:3`), so the driver's constant folding is load-bearing for correctness
        // rather than for quality, and the ordinary guards are blind to a wrong fold (it changes
        // neither the declared shared size nor the output).  That clause lives in the ROW check
        // (ggml-vulkan.cpp), not here: this table is shared with the init path, which legitimately
        // fills spec 12/13 with its own values.
        if (g.WM / g.TM < 1 || g.WN / g.TN < 1) {
            return "cm1: WM / TM or WN / TN is 0 (the shader's cms_per_row / cms_per_col)";
        }
        return nullptr;
    }

    const uint32_t wniter = g.WM * g.WN / (wave * g.TM * g.TN * g.WMITER);
    if ((g.WM / g.WMITER) % g.TM != 0)  return "(WM / WMITER) % TM != 0";
    if (wniter < 1)                       return "wniter < 1 (TM*TN*WMITER too large for WM*WN/wave)";
    if (g.WN % wniter != 0)               return "WN % wniter != 0";
    if ((g.WN / wniter) % g.TN != 0)      return "(WN / wniter) % TN != 0";
    if (g.TM * g.TN > g.WM * g.WN / wave) return "TM * TN > WM * WN / wave";
    if (g.TM % 2 != 0)                  return "TM % 2 != 0";
    if (g.pad >= 0 && g.pad > 4)        return "PAD outside [0,4]";
    if (g.reshape >= 0 && g.reshape != 0 && g.reshape != 1) return "RESHAPE not in {0,1}";
    return nullptr;
}

// Dedup identity: two geometries that create the same effective tile must share a key.
// cm2 halves BN when `smaller` is set (mul_mm_cm2.comp:37-38), which is why
// {BN=256,smaller=1} and {BN=128,smaller=0} are the same candidate -- the same
// phenomenon that makes the driver report identical shared memory for cm2 l and m
// (t2-report.md section 5.1).
static inline uint64_t vk_tile_canonical_key(int path, const vk_tile_geometry & g) {
    uint32_t bn_eff = g.BN;
    if (path == VK_TILE_PATH_CM2 && g.smaller == 1 && (g.BN % 2) == 0) {
        bn_eff = g.BN / 2;
    }
    // cm1-d: PAD and RESHAPE joined the key in the same batch that gave them a reader.  Until then
    // they were absent for the honest reason the axis comment gave ("no reader anywhere"); with the
    // row's values reaching spec 12/13, two rows that differ only in them are DIFFERENT pipelines,
    // and a key that cannot say so would collapse them into one candidate.
    const uint32_t parts[12] = {
        (uint32_t) path, g.BM, bn_eff, g.BK, g.WM, g.WN, g.WMITER, g.TM, g.TN, g.BS,
        (uint32_t) (g.pad < 0 ? 0xFFFFFFFFu : (uint32_t) g.pad),
        (uint32_t) (g.reshape < 0 ? 0xFFFFFFFFu : (uint32_t) g.reshape),
    };
    // Standard FNV-1a 64 (offset basis 0xCBF29CE484222325, prime 0x100000001B3), fed one BYTE at
    // a time in little-endian order -- the same wire protocol the shell's rowSetHash() uses, so
    // both hashes are "FNV-1a" in the standard sense.  R0 changed the basis to the standard value
    // and the feed from one u32 per step to bytes; the function has no callers, is never
    // persisted and never compared across repos, so this only changes an in-process dedup key.
    // NOTE: `bn_eff` above exists for DEDUP only (two geometries that create the same tile must
    // share a key) -- it must NOT be reused as the pipeline align (t3-shell-plan.md 6.9: the
    // align rule is per path and is derived from the stock buckets, not from this identity).
    uint64_t h = 14695981039346656037ull;
    for (int i = 0; i < 12; i++) {
        const uint32_t v = parts[i];
        for (int b = 0; b < 4; b++) {
            h ^= (uint64_t) ((v >> (8 * b)) & 0xffu);
            h *= 1099511628211ull;
        }
    }
    return h;
}

// The align a row's ALIGNED variant must be created with, per path (t3-shell-plan.md 6.9).
// Derived from the stock buckets so that a row and the bucket it shadows agree:
//   cm2   : warptiles are {BLOCK_SIZE,BM,BN,BK,smaller} = {256,128,256,64,1} / {256,128,128,64,0}
//           / {128,64,64,64,0} and stock aligns l/m/s = 128/64/32 = BN/2 (ggml-vulkan.cpp:4806-4808);
//   scalar: warptiles have BM == BN and stock aligns 128/64/32 (ggml-vulkan.cpp:4875-4877),
//           which max(BM,BN) reproduces exactly.
// DIRECTION DISCIPLINE (the same one KATAGO_VK_MM_TILE=custom: states): too small means the
// shader reads across rows -- a silent wrong answer; too large only means fewer aligned hits --
// slower.  So there is nothing to measure when the two candidates disagree: take the larger.
// Taking the smaller needs shader-level evidence, never an A/B run.  An odd cm2 BN falls back to
// BN itself because BN/2 would truncate (i.e. round down = the dangerous direction).
static inline uint32_t vk_tile_row_align_for(int path, const vk_tile_geometry & g) {
    if (path == VK_TILE_PATH_CM2) {
        return (g.BN % 2 == 0) ? (g.BN / 2) : g.BN;
    }
    return g.BM > g.BN ? g.BM : g.BN;
}

// Per-path axis classification: the enumeration space, not validity.  Kept here so the
// enumerator and any validator read one table.  Sources: t1-predicate.md section 2
// (2026-09-13 rewrite), t1-candidates.md section 2.2, VENDOR-BASE.md section 2, and the
// measured BK verdict (census-t05-shapes.md section 4.4).
//
// Scalar: BM/BN/WM/WN/WMITER/TM/TN are ENUMERABLE; BS is DERIVED (num_warps*wave);
//   TK and WARP are PINNED (TK==1, WARP==vk_tile_wave_for(subgroup_size) -- fill them in from the
//   device, do not sweep: the wave is a hardware property, not a tunable);
//   BK is INERT on the f16 path (compile-time constant; spec id 3 exists only for the
//   quant branch), and SHMEM_STRIDE_PAD / APPLY_SLM_A_RESHAPE are INERT because on the
//   scalar path they are NOT specs at all (mul_mm.comp's COOPMAT guard and its explicit
//   "DO NOT spec-ize" note -- lesson #41: a multi-level spec array made the Blackwell
//   driver emit BM-dependent garbage).  That is a stronger statement than "measured
//   inert": the earlier experiment could not have shown anything else.
// cm1: as scalar, except that WMITER, SHMEM_STRIDE_PAD, APPLY_SLM_A_RESHAPE and BK are INERT
//   (cm1-c).  The reason is NOT "this hardware cannot" and NOT "our build has not got there": it
//   is that the ROW'S value never reaches the shader -- WMITER/WSUBM/WNITER are read only on the
//   `#else` side of mul_mm.comp, and spec 12/13 are filled from the global mm_spec_pad /
//   mm_spec_reshape rather than from the row (see the case bodies below for the line-level
//   evidence).  The cm1 row path itself IS wired since cm1-b2 -- rows install and hit (smoke A3j)
//   and a row whose fragment shape is not this device's cooperative-matrix shape is refused by I6
//   -- so this is a statement about the shader and the spec builder, not about our progress.
//   What stays ENUMERABLE on cm1 is exactly the geometry: BM/BN/WM/WN/TM/TN.
//   The older text here said the opposite ("ENUMERABLE but UNMEASURED, and the reason is OURS: the
//   cm1 ROW path is not wired yet"): true when it was written, stale since cm1-b2, and removed in
//   cm1-c because a comment that describes a past build state is a comment that will be believed.
//   Still true and still worth keeping: compat mode (GGML_VK_DISABLE_COOPMAT2=1) DOES select the
//   cm1 path and runs, but it is a PROXY, not a native cm1 device (fa-tune-exec.md section 9
//   boundary 1); it is the only way to exercise this path today, and its numbers must be labelled
//   as such wherever they are quoted.
// cm2: BM/BN/BK/BS/smaller are ENUMERABLE (BLOCK_SIZE is a real cm2 spec, not derived).
static inline constexpr int vk_tile_axis_class(int path, int axis) {
    if (path == VK_TILE_PATH_CM2) {
        switch (axis) {
            case VK_TILE_AXIS_BS:
            case VK_TILE_AXIS_BM:
            case VK_TILE_AXIS_BN:
            case VK_TILE_AXIS_BK:
            case VK_TILE_AXIS_SMALLER:
                return VK_TILE_AXIS_ENUMERABLE;
            case VK_TILE_AXIS_TK:
            case VK_TILE_AXIS_WARP:
                // NOT_APPLICABLE, not INERT: these are not axes of the cm2 path at all
                // (there is no warptile semantics there), so the enumerator should not
                // even consider them.  INERT would mean "exists but sweeping does nothing".
                return VK_TILE_AXIS_NOT_APPLICABLE;
            default:
                return VK_TILE_AXIS_NOT_APPLICABLE;
        }
    }
    if (path != VK_TILE_PATH_SCALAR && path != VK_TILE_PATH_CM1) {
        return VK_TILE_AXIS_NOT_APPLICABLE;
    }
    switch (axis) {
        case VK_TILE_AXIS_BM:
        case VK_TILE_AXIS_BN:
        case VK_TILE_AXIS_WM:
        case VK_TILE_AXIS_WN:
            return VK_TILE_AXIS_ENUMERABLE;
        case VK_TILE_AXIS_TM:
        case VK_TILE_AXIS_TN:
            // F14 (G-C4): PINNED on cm1 -- not ENUMERABLE, and NOT INERT either.
            //   * Not ENUMERABLE: the only value this device accepts is its own cooperative-matrix
            //     shape (I6, ggml-vulkan.cpp:20701), so a sweep would emit rows that can only be
            //     refused.  Those refusals are classified as DEVICE verdicts
            //     (VK_TUNE_CAPABILITY_CLASSES: our_gap = false), and `structural-all-reject` is
            //     writable exactly when every reason is a device property (vulkantuner.cpp:388 /
            //     :405-407) -- so sweeping an axis WE chose to sweep could mint a durable "this
            //     device cannot do cm1".  The plumbing that fills that flag from the install
            //     accounting is what waits for B5/B6 (:1375), which is why this is closed at the
            //     SOURCE now instead of relying on the write side to notice.
            //   * Not INERT: INERT means "exists, but sweeping it does nothing" (see the cm2 note at
            //     :440-443).  Here the value must be RIGHT; INERT would turn "fill the device's
            //     shape" into "fill anything".
            //   * The scalar path keeps both: its spec really carries constant_id 7/8 and no device
            //     constrains them (pinned by the `F14: TM/TN ...` assert below -- named by its MESSAGE
            //     rather than by an ordinal, because the ordinals moved when the E1 self-assert, which
            //     used to sit between them, was deleted).
            return path == VK_TILE_PATH_CM1 ? VK_TILE_AXIS_PINNED : VK_TILE_AXIS_ENUMERABLE;
        case VK_TILE_AXIS_WMITER:
            // Scalar: ENUMERABLE.  cm1: INERT (cm1-c).  WMITER reaches mul_mm.comp's COOPMAT
            // branch only through WSUBM/WNITER, and every USE of those is on the `#else` (scalar)
            // side (see the long note in vk_tile_candidate_reason above) -- so on cm1 a WMITER
            // sweep would produce N candidates that compile to the same pipeline.
            return path == VK_TILE_PATH_CM1 ? VK_TILE_AXIS_INERT : VK_TILE_AXIS_ENUMERABLE;
        case VK_TILE_AXIS_BS:
            return VK_TILE_AXIS_DERIVED;
        case VK_TILE_AXIS_TK:
        case VK_TILE_AXIS_WARP:
            return VK_TILE_AXIS_PINNED;
        // The two cm1-only spec constants (12/13) took ONE case until 2026-09-25; they are split now
        // because the closure gave them different answers on cm1.  What they still share is the SCALAR
        // story: there they are not specs at all -- SHMEM_STRIDE_PAD / APPLY_SLM_A_RESHAPE stay
        // compile-time consts (mul_mm.comp:124-133, the "DO NOT spec-ize" note, lesson #41: spec-ing
        // them miscompiles on the Blackwell driver) -- so a scalar row carrying them is a silent no-op,
        // which is what INERT is for.  On cm1 both values DO reach their spec: cm1-d wired them (spec
        // builder call at ggml-vulkan.cpp:21035) and added them to the dedup key.
        //
        // (This block used to also carry a cm1-c-era sentence saying the two fields had "NO reader
        // anywhere", with the cm1-d correction appended after it -- so it said both things and
        // contradicted the `return` below it.  The historical text is recoverable from git.)
        case VK_TILE_AXIS_SPEC_PAD:
            // PAD stays SWEEPABLE on cm1: M tier, real measured effect (shmem 10240 vs 12288), and it
            // is how "this device wants a different layout" is expressed.  Scalar stays INERT.
            return path == VK_TILE_PATH_CM1 ? VK_TILE_AXIS_ENUMERABLE : VK_TILE_AXIS_INERT;
        case VK_TILE_AXIS_SPEC_RESHAPE:
            // ⚠️ 2026-09-25: the paragraph that stood here argued RESHAPE could stay ENUMERABLE because
            //    "row-override closure is a separate decision".  THAT POSITION IS WITHDRAWN, because the
            //    closure has to live in the CLASSIFICATION or it does not live anywhere: `ALL_AXES` in
            //    the tuner's domain audit sweeps exactly the axes this function reports ENUMERABLE, so
            //    leaving RESHAPE there meant (a) it was still treated as a sweepable axis, and (b) the
            //    audit REQUIRED a declared domain for it -- i.e. adding a cm1 domain would silently
            //    un-do the closure.  Dormant until then only because no cm1 domain exists yet.
            //
            //    Why PINNED and not INERT: the value still has to be filled in CORRECTLY (it reaches
            //    spec 13, see above), it simply may not be SWEPT.  INERT would claim the value never
            //    arrives, which the cm1-d wiring made false.  Same shape and reason as the TM/TN PINNED
            //    entries: a value the decision dictates.  The basis is 5.3.1's H tier -- a wrong
            //    constant fold there is a DRIVER FAULT, so SWEEPING the axis is what creates the hazard.
            return path == VK_TILE_PATH_CM1 ? VK_TILE_AXIS_PINNED : VK_TILE_AXIS_INERT;
        case VK_TILE_AXIS_BK:
            // f16 family: compile-time constant in the shader.  Per path: on cm2 the spec really
            // is {BLOCK_SIZE, BM, BN, BK, smaller}, so BK is live THERE -- this table is only ever
            // asked about the scalar/cm1 family, which is why one answer is enough.
            return VK_TILE_AXIS_INERT;
        default:
            return VK_TILE_AXIS_NOT_APPLICABLE;
    }
}

// ── E1: the one OBSERVED driver boundary, and the one spelling of it ────────────────────────────────
// 🔴 THIS BLOCK IS DOWN HERE, AFTER vk_tile_axis_class(), AND THE PLACEMENT IS LOAD-BEARING.  The file's
// order is otherwise "rules first, classification later", and moving this up with the other tile rules
// is the obvious-looking refactor that BREAKS THE BUILD: `vk_tile_axis_class` names BOTH the enum type
// (declared near the top) and the function below, so a call written above the function's declaration
// resolves to the TYPE and is parsed as a function-style cast ("cannot convert from 'initializer list'
// to 'vk_tile_axis_class'").  Measured with MSVC 14.51; neither `const T x = f(...)` nor `&f` fixes it,
// because both meet the same incomplete name.  One order, one reason.
//
// `WM == 16 && WN == 16 && BS >= 256` HUNG THE DRIVER (t1-predicate.md:136, rule "E1").  The custom:
// parser lets it through, so the boundary is real, reachable, and not a preference.
//
// WHY THIS LIVES IN THE SHARED HEADER RATHER THAN IN EITHER READER.  It has two readers and they must
// not be able to disagree:
//   * the SHELL's candidate table (vulkantuner.cpp, CANDIDATE_RULES -> rule "E1-blacklist"), which
//     keeps the configuration out of the enumerated set -- a MUST rule, so it is not even a ledger
//     entry;
//   * the BACKEND's install firewall (ggml_vk_tuned_row_capability_reason), which must refuse it as a
//     NAMED reason, because the enumerator is not the only door: a row arriving from the cache or from
//     KATAGO_VK_TUNE_IMPORT never passes through enumeration at all.
// One fact, two independent gates -- and the gates guard DIFFERENT doors, which is why both are
// needed.  Two hand-written copies of the sentence would be the next drift, so the sentence is
// returned from ONE function here (G-D26: one spelling for one condition; precedent is the "path is
// neither scalar, cm1 nor cm2" string, shared the same way with a row in VK_TUNE_CAPABILITY_CLASSES).
//
// WHY IT IS NOT A CLAUSE IN vk_tile_candidate_reason().  That function is DEVICE-FREE by contract --
// the shell calls it while parsing rows, with no device handle (see the I6 note in the cm1 branch
// below).  The boundary is stated in terms of BLOCK_SIZE, whose meaning in it is `num_warps * wave`
// (the `BS != num_warps * wave` clause), so judging it needs the DEVICE's wave.  `BS >= 256` is
// therefore a boundary ON A DEVICE CLASS, not a self-consistency rule about a geometry, and the table
// that may not consult a device cannot hold it.
//
// 🔴 `BS >= 256` IS A FLOOR, NOT AN EQUALITY.  E1 was measured at 256 and never swept upward; the
// refusal is the whole half-line because a configuration BELOW a measured hang is not thereby shown
// to be safe.  Reading this as `== 256` is the way this rule silently loses its teeth.
//
// 🔴 SCOPE IS THE AXIS TABLE'S ANSWER, not a hardcoded path test: vk_tile_axis_class() reports WM/WN as
// INERT on cm2 and ENUMERABLE on scalar and cm1 -- i.e. exactly the surface where those two numbers are
// things a ROW can choose.  Deriving it that way means the day an axis is reclassified, this rule
// follows instead of going stale.  (BS is deliberately NOT part of the scope test: it is DERIVED, see
// the note inside the function.  Requiring it to be ENUMERABLE skipped cm1 entirely -- a bug that was
// caught, and the assert that caught it has since been deleted as self-verifying; see the deletion note
// further down.)
#define VK_TILE_E1_WM        16
#define VK_TILE_E1_WN        16
#define VK_TILE_E1_BS_FLOOR  256
// `constexpr` is KEPT, and the honest reason is NOT "something still needs it at compile time" -- the
// assert that did is gone, and a check of both callers shows neither is a constant-expression context:
// the shell stores this as a plain function pointer in a `const` table (vulkantuner.cpp, the
// "E1-blacklist" row of CANDIDATE_RULES), and the firewall calls it at runtime.  It is kept because it
// costs nothing and keeps the predicate usable in a constant expression if a future guard needs one --
// stated this way so the next reader is not told a requirement that does not exist.  What IS structural
// is that both readers call THIS function rather than restating the boundary, so an edit here reaches
// both doors at once.
static inline constexpr bool vk_tile_e1_blacklist_hit(int path, const vk_tile_geometry & g) {
    switch (path) {
    case VK_TILE_PATH_SCALAR:
    case VK_TILE_PATH_CM1:
        break;
    default:
        return false;   // cm2 and anything unknown: these are not row-settable axes there
    }
    // 🔴 SCOPE = "the path lets a ROW choose WM and WN", and it is the AXIS TABLE's answer rather than a
    // hardcoded path test: both are ENUMERABLE on scalar and cm1 and INERT on cm2, so the day an axis is
    // reclassified this rule follows instead of going stale.
    //
    // 🔴 AND THE CHECK IS WM/WN ONLY -- BS IS NOT TESTED FOR ENUMERABILITY, because it is not a free axis
    // anywhere: on cm1 the table classes it DERIVED (`BS == num_warps * wave`, recomputed from the four
    // free axes plus the device's wave) and on scalar it is a free-but-derived field.  Requiring BS to be
    // ENUMERABLE made this predicate return FALSE on cm1 and the static_assert below caught it -- which
    // is the whole reason the scope is asserted at compile time instead of described in a comment.  The
    // boundary still READS BS; it just must not demand that BS be sweepable, or the one path whose
    // geometry most easily reaches the measured hang would be the path this rule skipped.
    if (vk_tile_axis_class(path, VK_TILE_AXIS_WM) != VK_TILE_AXIS_ENUMERABLE
     || vk_tile_axis_class(path, VK_TILE_AXIS_WN) != VK_TILE_AXIS_ENUMERABLE) {
        return false;
    }
    return g.WM == VK_TILE_E1_WM && g.WN == VK_TILE_E1_WN && g.BS >= VK_TILE_E1_BS_FLOOR;
}

// The refusal sentence, returned by the ONE function both readers call.  It names the measured fact,
// the three numbers, and why the predicate could not have decided it -- the reader of a dropped row
// has to be able to tell "the driver hung on this" from "we chose not to expose this".
static inline const char * vk_tile_e1_blacklist_reason(int path, const vk_tile_geometry & g) {
    if (!vk_tile_e1_blacklist_hit(path, g)) {
        return nullptr;
    }
    return "WM == 16 && WN == 16 && BS >= 256, which hung the driver (E1: t1-predicate.md:136); "
           "the custom: parser admits it, this geometry is not row-overridable, and the boundary "
           "needs the device's wave so the device-free predicate cannot state it";
}

// ── E1 HAS NO static_assert HERE, AND THAT IS THE CORRECT SHAPE (2026-09-24, deletion review) ──────
// A `vk_tile_e1_boundary_pinned()` helper stood here, folding eight geometries through the predicate.
// It was REMOVED, and the reason is worth keeping so it is not rebuilt: **it verified its own input.**
// `vk_tile_e1_blacklist_hit` is `constexpr` and the boundary is a `#define`, so the assertion compared
// a function against constants that are its own definition -- it could only fail if the predicate were
// edited into contradicting itself on the same line it defines itself.  What it COULD NOT check is the
// thing actually in question: **whether the clause is WIRED into the install path** (a deleted call
// site leaves it green, which was measured).  A safety assertion that guards nothing while looking
// like it guards the boundary is worse than no assertion, because it is what a later reader cites.
//
// Where the boundary IS pinned instead, and why each of those is a different kind of check:
//   * the two literals `16` / `256` exist exactly ONCE, in the `#define`s below -- one spelling;
//   * the SHARED PREDICATE is what both the shell's rule table and the firewall call, so there is no
//     second boundary to drift (see the two-reader note above);
//   * the FIREWALL'S call site is NOT held by anything yet, and this line used to claim otherwise.
//     The text-level guardrail that would require the call is PLANNED, not written (handoff-algo.md
//     section 8a/8b): today a deleted call site leaves every check green, which is the measured gap.
//     Stated as MISSING rather than as present, because a check holds this is exactly the kind of
//     sentence a later reader stops verifying;
//   * "no row reaches this clause through the install path" is a measured property of the derived-BS
//     clause, recorded in handoff-algo.md section 8a -- not a compile-time claim.
// The measured number's own history (why a FLOOR and not an equality, why the scope is the axis
// table's answer) stays in the comments above, which is where a reader needs it.  

// Convenience for the enumerator: may this axis be swept on this path?
static inline int vk_tile_axis_is_live(int path, int axis) {
    return vk_tile_axis_class(path, axis) == VK_TILE_AXIS_ENUMERABLE;
}

// cm1-c: the classification above is DATA with no device in it, so the compiler can check the whole
// claim instead of a reader having to trust the comment -- and "this axis does nothing" is exactly
// the kind of claim that rots silently.  Making vk_tile_axis_class() constexpr (it is a pure switch
// over two ints) is what buys the asserts below; nothing else about it changed.
//
// 1) cm1's DECORATIVE axes must not be swept -- and the two kinds are named separately, because
//    collapsing them into one word is exactly how this comment went stale: WMITER and BK never
//    reach the shader on either path, while PAD and RESHAPE have really been wired since cm1-d
//    (their row values DO reach spec 12/13, which is why they are ENUMERABLE).  The assert pins
//    BOTH halves, so neither word can drift over the other again.
static_assert(vk_tile_axis_class(VK_TILE_PATH_CM1, VK_TILE_AXIS_WMITER) == VK_TILE_AXIS_INERT &&
              vk_tile_axis_class(VK_TILE_PATH_CM1, VK_TILE_AXIS_BK) == VK_TILE_AXIS_INERT,
              "cm1: WMITER / BK must not be swept on cm1 -- sweeping them would emit N identical "
              "rows (the shader does not read the row's value on either path).");
// 2) The other direction, so that INERT cannot spread over the real geometry by accident: what cm1
//    may sweep is the four FREE geometry axes (BM/BN/WM/WN) PLUS pad -- FIVE, was six until RESHAPE
//    left the set on 2026-09-25 -- and BS is DERIVED from the geometry plus the wave.  TM/TN left this
//    set in F14: they are this device's fragment shape, so they are PINNED (case body above).  The
//    value must be right, but sweeping it would only produce rows the device refuses, and those
//    refusals are classed as DEVICE verdicts -- i.e. a sweep of OUR choosing could mint a durable
//    device claim.  RESHAPE joined them for a DIFFERENT reason (5.3.1's H tier: sweeping it is what
//    creates a driver-fault hazard, not what creates a wrong answer), which is why it is PINNED and
//    not INERT -- INERT would say the value never arrives, and it does.
static_assert(vk_tile_axis_class(VK_TILE_PATH_CM1, VK_TILE_AXIS_BM) == VK_TILE_AXIS_ENUMERABLE &&
              vk_tile_axis_class(VK_TILE_PATH_CM1, VK_TILE_AXIS_BN) == VK_TILE_AXIS_ENUMERABLE &&
              vk_tile_axis_class(VK_TILE_PATH_CM1, VK_TILE_AXIS_WM) == VK_TILE_AXIS_ENUMERABLE &&
              vk_tile_axis_class(VK_TILE_PATH_CM1, VK_TILE_AXIS_WN) == VK_TILE_AXIS_ENUMERABLE &&
              vk_tile_axis_class(VK_TILE_PATH_CM1, VK_TILE_AXIS_SPEC_PAD) == VK_TILE_AXIS_ENUMERABLE &&
              // 2026-09-25: RESHAPE left the sweepable set.  PINNED, not INERT: the value still has to
              // be right (it reaches spec 13), it just may not be swept.  This is the line that makes
              // the closure real -- changing the case body alone would leave this failing to compile,
              // which is the intended behaviour of an assert that pins a decision.
              vk_tile_axis_class(VK_TILE_PATH_CM1, VK_TILE_AXIS_SPEC_RESHAPE) == VK_TILE_AXIS_PINNED &&
              vk_tile_axis_class(VK_TILE_PATH_CM1, VK_TILE_AXIS_BS) == VK_TILE_AXIS_DERIVED &&
              // F14: and the two the device dictates are PINNED, not INERT -- the value they carry
              // must be the device's, they simply may not be swept.  Without these two claims the
              // `path ==` test in the case body could be dropped or the answer changed to INERT and
              // nothing would redden (R35 checks that these claims EXIST; the compiler checks them).
              vk_tile_axis_class(VK_TILE_PATH_CM1, VK_TILE_AXIS_TM) == VK_TILE_AXIS_PINNED &&
              vk_tile_axis_class(VK_TILE_PATH_CM1, VK_TILE_AXIS_TN) == VK_TILE_AXIS_PINNED,
              "cm1: the sweepable cm1 axis space is the free geometry (BM/BN/WM/WN) plus PAD (five, was "
              "six until RESHAPE was closed on 2026-09-25), with BS derived and the fragment shape "
              "(TM/TN) pinned -- RESHAPE is PINNED too, for the H-tier reason, not because its value "
              "does not arrive");
// 3) WMITER is the one axis whose answer DIFFERS between the two warptile paths, so it is the one a
//    refactor can collapse without noticing: scalar runs the `#else` side that reads WSUBM/WNITER.
static_assert(vk_tile_axis_class(VK_TILE_PATH_SCALAR, VK_TILE_AXIS_WMITER) == VK_TILE_AXIS_ENUMERABLE &&
              vk_tile_axis_class(VK_TILE_PATH_SCALAR, VK_TILE_AXIS_SPEC_PAD) == VK_TILE_AXIS_INERT,
              "cm1-c: WMITER is a live scalar axis (the scalar inner loop reads WSUBM/WNITER); its "
              "cm1-INERT half is asserted above, not here");
// 4) The scalar half of the per-path split, RESTORED after a wrong deletion (2026-09-24).  The deletion
//    review filed this as a restatement of assert 2 and removed it; that was wrong, and the check that
//    caught it is worth naming: EVERY axis in assert 2 is guarded under VK_TILE_PATH_CM1, so removing
//    this line left scalar TM/TN with NO assert at all -- and the scalar case body is
//    `path == CM1 ? PINNED : ENUMERABLE`, so a dropped `path ==` test would silently turn a real scalar
//    tuning axis INERT.  The claim is about SCALAR; assert 2's is about CM1.  Two paths, two asserts.
static_assert(vk_tile_axis_class(VK_TILE_PATH_SCALAR, VK_TILE_AXIS_TM) == VK_TILE_AXIS_ENUMERABLE &&
              vk_tile_axis_class(VK_TILE_PATH_SCALAR, VK_TILE_AXIS_TN) == VK_TILE_AXIS_ENUMERABLE,
              "F14: TM/TN are pinned on cm1 but must stay sweepable on the scalar path");
