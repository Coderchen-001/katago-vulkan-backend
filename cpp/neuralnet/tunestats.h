#ifndef KATAGO_NEURALNET_TUNESTATS_H_
#define KATAGO_NEURALNET_TUNESTATS_H_

#include <cstddef>
#include <string>
#include <vector>

// Tuner statistics -- the comparison ALGORITHM, as pure functions.
//
// WHY A SEPARATE, DEPENDENCY-FREE TU (2026-09-21; user directive in
// project-max-principle.md section 9: "the algorithm IS the core of tuning"):
//   what the tuner writes into the cache is PERMANENT -- a row set (or a negative entry) is loaded
//   and used WITHOUT re-measuring -- so these statistics are not a report, they are the product.
//   Keeping them pure (no ggml, no logger, no device, no file system) is what lets them be checked
//   against KNOWN GROUND TRUTH on the CPU, before any GPU code exists.  That is the acceptance the
//   user demanded: "anti-jitter + little tune time + accurate" must be PROVEN, not claimed.
//
// The algorithm's reference is the industrial microbenchmark implementation under
// others-src-to-referrence/benchmark (Google Benchmark): time-targeted blocks, a rank-based
// comparison with a declared alpha and a repetition floor, and a stability discipline.  What that
// reference cannot give us is enumerated in tune-algorithm-audit.md section 11.
//
// 2026-09-21, delete-dead-criteria batch: the two ORDER-STATISTIC intervals are GONE from the
// shipped path.  Both were MEASURED unusable under our spike shape -- a one-sided spike shifts
// every Walsh average that contains that d_i, so intervals built on Walsh order statistics are
// wrong there, and the percentile bootstrap was 40% wide and fired the A/A sentinel in 86% of
// sessions.  One interval ships: median +- t * 1.2533 * MAD / sqrt(n), whose multiplier is
// Student's t with n-1 degrees of freedom because the scale is ESTIMATED from the same n pairs (a
// Gaussian z would claim more confidence than the data supports).  Deleting them also removed about
// 95% of this self-test's cost (the bootstrap sorted 400 Walsh medians per trial; a probe leg that
// took 225 s now takes seconds).  The generator and the six criteria stay, so either interval can be
// put back as a candidate and re-measured.  The pair-count floor was re-measured on the NEW interval
// and did NOT move down: at n=16 the out-of-sample reading is 0.927 / 0.933 (1000 / 4000 trials,
// logs-t3/stats-v3-sweep-*.txt), i.e. ON the 0.93 line, while n=25 sits at 0.943-0.945 -- so 25 stays
// as a deliberately conservative floor whose cost is counted (about 3.5-5 s per tune).
// See tune-algorithm-audit.md section 7 for the delete list this file implements.
namespace TuneStats {

  // ---- the DECLARED protocol constants: ONE source, so the probe batch cannot re-copy them ----
  constexpr double MIN_BLOCK_TIME_MS = 2.0;  // declared block; the us clock's 0.19% quantisation stays under the 0.1% sigma budget
  constexpr long   PAIRS_FLOOR = 25;         // MEASURED floor: n=16 is on the 0.93 line, n=25 is 0.943-0.945 (logs-t3/stats-v3-sweep-*.txt)
  constexpr double ALPHA = 0.05;             // declared, not hidden inside a formula
  constexpr double TOLERANCE = 0.01;         // 1% : the product tolerance
  constexpr double SPIKE_RATE_GATE = 0.15;   // MEASURED bound: 15% passes, 18% fails on THREE criteria at once (the rate12/15/18 sweep)
  constexpr long   REPS_MAX = 512;
  constexpr long   DEV_TRIALS_DEFAULT = 200; // the self-test's size when KATAGO_VK_TUNE_STATS_TRIALS is unset

  // ---- descriptive: REPORTING only.  No threshold hangs on these. ----
  double median(std::vector<double> v);
  double madScale(const std::vector<double>& v);   // 1.4826 * MAD (textbook robust scale)

  // Student's t quantile -- the small-sample multiplier for an interval whose scale is ESTIMATED.
  // Textbook part with no fitted constant: inverted from the regularised incomplete beta (Lentz
  // continued fraction) by bisection, and pinned against published table values by the self-test.
  double tQuantile(double p, double df);

  // ---- the paired decision machinery ----
  // Input: d_i = t_stock,i - t_cand,i for ONE rep each (the pairing cancels the common-mode drift a
  // cross-run comparison would carry: measured as a 1.42x scale difference, logs-t3/calib-d-paired.txt).
  struct PairedResult {
    std::size_t pairs;        // usable pairs (exact zeros dropped, the signed-rank convention)
    std::size_t zerosDropped;
    double hl;                // Hodges-Lehmann estimate of the median difference (Walsh averages)
    double ciLow;             // hl - t_{n-1,1-alpha/2} * 1.2533 * MAD / sqrt(n)
    double ciHigh;
    bool ciOk;                // false when n cannot support an interval at all (fail closed)
    double pValue;            // two-sided Wilcoxon signed-rank
    bool pExact;              // true when the exact null distribution was used
    double wPlus;             // the statistic, recorded for the evidence
  };
  PairedResult pairedSignedRank(const std::vector<double>& d, double alpha);

  // The two decisions, both reading the ONE interval:
  //   non-inferiority -- the persisted row-set claim: the whole interval above -tolerance
  //   equivalence     -- the A/A sentinel: the whole interval inside +-tolerance (failing to reject
  //                      equality is not evidence of equality, which is how a broken comparison
  //                      would slip past the older "the sentinel did not win" phrasing)
  bool notWorseThan(const PairedResult& r, double tolerance);
  bool equivalentWithin(const PairedResult& r, double tolerance);
  bool intervalCovers(const PairedResult& r, double value);

  // ---- the budget: solved ONCE, up front (no optional stopping) ----
  struct Budget {
    std::vector<long> repsFormula;   // the Gaussian formula's answer: REPORTED, and systematically low under spikes
    std::vector<long> repsRequired;  // max(floorReps, formula) -- what is actually run
    long totalReps;
    long floorReps;
    bool anyClampedAtMax;
  };
  // reps_s = ceil( (z_{1-alpha/2} * sigmaRel_s / tolerance)^2 ), clamped to [floorReps, maxReps].
  // sigmaRel_s is the BLOCK-relative paired scale: see tune-measurement.md section 2.5 -- the
  // per-dispatch-relative figure benchgemm prints is NOT the right normalisation.
  Budget allocateReps(const std::vector<double>& sigmaRel, double tolerance, double alpha,
                      long floorReps, long maxReps);

  // The DEV knob: the self-test's trial count (default DEV_TRIALS_DEFAULT, clamped to [10, 20000]).
  // Logging/flow only -- it cannot change what the tuner may select, so it is NOT a policy knob and
  // deliberately does not enter policyFingerprint.
  long devTrials();

  // ---- the self-test: known ground truth, pure CPU, deterministic ----
  // Returns ONE line; the shell logs it (the shell owns the logging conventions).
  std::string selfTestSummary();
}

#endif
