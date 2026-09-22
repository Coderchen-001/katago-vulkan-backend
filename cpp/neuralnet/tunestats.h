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
// WHAT IS DELIBERATELY **NOT** HERE: half-split self-checks, spike replacement, k/k2 thresholds and
// CV gates.  All four were ours, all four were unpinned or biased, and all four are on the delete
// list in tune-algorithm-audit.md section 7.
namespace TuneStats {

  // ---- descriptive: REPORTING only.  No threshold hangs on these. ----
  double median(std::vector<double> v);
  double madScale(const std::vector<double>& v);   // 1.4826 * MAD (textbook robust scale)

  // Which interval the decision reads.  All three are computed and carried, because the self-test's
  // whole job is to MEASURE them against injected ground truth and retire the ones that fail:
  //   CI_MAD       : median +- z * 1.2533 * MAD / sqrt(n)
  //   CI_WILCOXON  : the distribution-free signed-rank interval (order statistics of the Walsh
  //                  averages, trimmed by the alpha/2 quantile of W+ with the ties correction)
  //   CI_BOOTSTRAP : percentile bootstrap over pairs -- MEASURED UNUSABLE (interval 40% wide, the
  //                  A/A sentinel firing in 86% of sessions).  Kept only so the self-test keeps
  //                  measuring its failure and nobody re-introduces it blind.
  enum Interval { CI_MAD = 0, CI_WILCOXON = 1, CI_BOOTSTRAP = 2 };
  const char* intervalName(Interval iv);

  // ---- the paired decision machinery ----
  // Input: d_i = t_stock,i - t_cand,i for ONE rep each (the pairing cancels the common-mode drift a
  // cross-run comparison would carry: measured as a 1.42x scale difference, logs-t3/calib-d-paired.txt).
  struct PairedResult {
    std::size_t pairs;        // usable pairs (exact zeros dropped, the signed-rank convention)
    std::size_t zerosDropped;
    double hl;                // Hodges-Lehmann estimate of the median difference (Walsh averages)
    double ciLowMad, ciHighMad;
    bool madOk;
    double ciLowWil, ciHighWil;
    bool wilOk;
    double ciLowBoot, ciHighBoot;
    bool bootOk;
    double pValue;            // two-sided Wilcoxon signed-rank
    bool pExact;              // true when the exact null distribution was used
    double wPlus;             // the statistic, recorded for the evidence
  };
  PairedResult pairedSignedRank(const std::vector<double>& d, double alpha, unsigned long long seed);

  // The two decisions, both reading ONE chosen interval:
  //   non-inferiority -- the persisted row-set claim: the whole interval above -tolerance
  //   equivalence     -- the A/A sentinel: the whole interval inside +-tolerance (failing to reject
  //                      equality is not evidence of equality, which is how a broken comparison
  //                      would slip past the older "the sentinel did not win" phrasing)
  bool notWorseThan(const PairedResult& r, double tolerance, Interval iv);
  bool equivalentWithin(const PairedResult& r, double tolerance, Interval iv);
  bool intervalCovers(const PairedResult& r, Interval iv, double value);

  // ---- the budget: solved ONCE, up front (no optional stopping) ----
  struct Budget {
    std::vector<long> repsPerShape;
    long totalReps;
    long floorReps;
    bool anyClampedAtMax;
  };
  // reps_s = ceil( (z_{1-alpha/2} * sigmaRel_s / tolerance)^2 ), clamped to [floorReps, maxReps].
  // sigmaRel_s is the BLOCK-relative paired scale: see tune-measurement.md section 2.5 -- the
  // per-dispatch-relative figure benchgemm prints is NOT the right normalisation.
  Budget allocateReps(const std::vector<double>& sigmaRel, double tolerance, double alpha,
                      long floorReps, long maxReps);

  // ---- the self-test: known ground truth, pure CPU, deterministic ----
  // Returns ONE line; the shell logs it (the shell owns the logging conventions).
  std::string selfTestSummary();
}

#endif
