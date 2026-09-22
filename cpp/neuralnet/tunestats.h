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
// put back as a candidate and re-measured.
//
// WHY THE FLOOR IS 25 (corrected 2026-09-21, third review round -- this text used to claim "16 sits on
// the line, 25 has margin", and our own four-seed readings REFUTED that): n=16 and n=25 are NOT
// DISTINGUISHABLE in this experiment.  The out-of-sample gap is +0.0180 on the default seed and
// -0.0040 on seed 0x1234 (16 HIGHER), i.e. its SIGN is unstable, and the continuous comparison would
// need ~20000 trials for 4 sigma -- so it is not a criterion.  25 is a DECLARED value plus a
// CONSERVATIVE choice, and what it buys is MARGIN AGAINST THE UNVERIFIED PART OF THE ENVELOPE (the
// harsh 20% row does collapse), NOT "16 is worse".  Its cost is counted: +9 pairs, about 2.4 s ->
// 3.5-5 s per tune.  The answerable question is binary -- does 16 pass the six criteria on the rows at
// or below the gate -- and the self-test measures exactly that (pairfloor[...], four pre-registered
// verdicts in stats-followup-preview.md section 12.5).  What catches a floor quietly lowered to 16 is
// the DECLARATION (smoke A44 pins pairs_floor=25), not a statistic.  And because no statistical reason
// exists in either direction, lowering it back to 16 later is EQUALLY defensible: that is a
// declared-value change plus a cost/benefit account, not a technical impossibility.
//
// WHAT THE t MULTIPLIER IS, AND WHAT IT IS NOT (review point, 2026-09-21): the sampling distribution
// of a MEDIAN is not Student's t.  1.4826 and 1.2533 are NORMAL-world consistency factors, and
// t_{n-1} is a HEURISTIC that pays back the fact that the scale was estimated from the same n pairs.
// Its warrant is therefore MEASURED coverage (n=25: 0.9450 / 0.9430 over 1000 / 4000 trials), NOT
// distribution theory -- and n=16 landing anywhere from 0.927 to 0.943 across seeds (it does not
// pay back cleanly) is
// precisely the evidence that it is not a theoretical solution.  Nobody may argue "theory says it
// should be t, so the floor can go back to 16": that needs new READINGS, not an argument.
//
// See tune-algorithm-audit.md section 7 for the delete list this file implements.
namespace TuneStats {

  // ---- the DECLARED protocol constants: ONE source, so the probe batch cannot re-copy them ----
  constexpr double MIN_BLOCK_TIME_MS = 2.0;  // declared block; the us clock's 0.19% quantisation stays under the 0.1% sigma budget
  constexpr long   PAIRS_FLOOR = 25;         // DECLARED + conservative: 16 and 25 are indistinguishable here (see the header)
  constexpr double ALPHA = 0.05;             // declared, not hidden inside a formula
  constexpr double TOLERANCE = 0.01;         // 1% : the product tolerance
  constexpr double SPIKE_RATE_GATE = 0.15;   // MEASURED bound: 15% passes, 18% fails on THREE criteria at once (the rate12/15/18 sweep)
  constexpr long   REPS_MAX = 512;
  constexpr long   DEV_TRIALS_DEFAULT = 200; // the self-test's size when KATAGO_VK_TUNE_STATS_TRIALS is unset
  // The trial count below which the coverage assertion is LOOSE, and SAYS SO in the printed line.  With
  // the SE taken from the measured coverage (a plug-in estimate -- that is WHY the line floats with the
  // observation), "TARGET - 3*SE" admits 89.1% at 200 trials and 91.9% at 4000: a default run can pass
  // it without really testing anything.  The marker is a READING with a criterion on both sides (smoke
  // A44: a >=1000-trial leg must carry NO marker, the 200-trial leg must carry one), not a comment.
  constexpr long   COVERAGE_STRICT_MIN_TRIALS = 1000;
  // The effective-coverage TARGET: the line the pair-count floor has to clear.  Three things are
  // written down here, because a bare 0.93 in prose is how a threshold becomes homeless:
  //   (i)   it is OUR requirement for "the interval really covers", and it is NOT the band's lower
  //         edge inside passes() -- that one is 0.90 and means "this row of the six criteria passed";
  //   (ii)  the self-test asserts n=25 >= TARGET - 3*SE with SE computed from THIS run's trial count.
  //         A fixed 0.93 would be a coin flip at the default 200 trials (17.6% false red; the
  //         arithmetic is tabulated in stats-followup-preview.md section 7) and trivially green at
  //         4000 -- one assertion, three trial counts, one meaning;
  //   (iii) NO practical assertion separates n=16 from n=25: the measured gap is 0.010-0.012, which
  //         needs about 20000 trials for 4 sigma (4000 trials gives 1.85).  The floor of 25 is
  //         therefore a DECLARED value backed by the recorded readings and the cost account; what
  //         catches a floor quietly lowered to 16 is the DECLARATION (smoke A44 pins pairs_floor=25),
  //         not a statistic.  n=16 is accordingly REPORTED with its distance to the target and never
  //         asserted: a "must be below" clause would be red about 22% of the time.
  //   (iv)  SCOPE: this assertion covers the MEASURED row only.  It must NOT be generalised to the
  //         envelope rows: at 4000 trials the scaled line is 0.9191 while the gate row's own coverage
  //         is 0.921-0.928, i.e. 0.4-2.1 sigma of headroom -- making it a coin flip there.  Those
  //         rows are judged by passes()'s six criteria, which is what the envelope was MEASURED with.
  constexpr double COVERAGE_TARGET = 0.93;
  // THE STABILITY BOUNDARY (2026-09-21): how close to the decision line a PASS may sit before the
  // verdict is recorded as UNRESOLVED (outcome `unstable`, nothing written) instead of being written.
  //
  // It is a DECLARED choice, and the number it implies is part of the declaration: a pass means
  // `CI_low + tolerance > k * SE`, so the probability that a re-measurement does NOT re-confirm it is
  // about P(Z < -k) -- k=1 ~= 16%, k=2 ~= 2.3%, k=3 ~= 0.13%.  We take k = 1.0 BECAUSE an unresolved
  // verdict does not erase anything (no conclusion => no write): the cost of a flip is "not
  // re-confirmed", not "the decision was reversed".  The 16% is written down so the next reader knows
  // what is being accepted; anyone who wants ~2.3% moves this one number and re-runs the pairfloor
  // table, because it is a decision rule (the build revision already covers the identity -- it does NOT
  // need a KNOBS[] entry).
  constexpr double STABILITY_BOUNDARY_SE = 1.0;

  // ---- descriptive: REPORTING only.  No threshold hangs on these. ----
  double median(std::vector<double> v);
  double madScale(const std::vector<double>& v);   // 1.4826 * MAD (textbook robust scale)

  // Student's t quantile -- the small-sample multiplier for an interval whose scale is ESTIMATED.
  // Textbook part with no fitted constant: inverted from the regularised incomplete beta (Lentz
  // continued fraction) by bisection, and pinned against published table values by the self-test.
  // Its ROLE here is the heuristic described in the header comment, not a claim about the median.
  double tQuantile(double p, double df);

  // ---- the paired decision machinery ----
  // Input: d_i = t_stock,i - t_cand,i for ONE rep each (the pairing cancels the common-mode drift a
  // cross-run comparison would carry: measured as a 1.42x scale difference, logs-t3/calib-d-paired.txt).
  struct PairedResult {
    std::size_t pairs;        // usable pairs (exact zeros dropped, the signed-rank convention)
    std::size_t zerosDropped;
    double hl;                // Hodges-Lehmann estimate of the median difference (Walsh averages)
    double ciLow;             // hl - t_{n-1,1-alpha/2} * 1.2533 * MAD / sqrt(n); the multiplier is the
                              // HEURISTIC from the header, and its warrant is MEASURED coverage
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

  // TRUE when the decision is too close to the line to be written (see STABILITY_BOUNDARY_SE): the
  // residual `CI_low + tolerance` is within k * SE of zero.  SE needs no new instrument -- the interval
  // half-width IS t_{n-1,1-alpha/2} * SE, so it is recovered from the same interval the decision reads.
  // Fails closed: no interval (or too few pairs) means nothing is decided, hence unresolved.
  bool decisionUnstable(const PairedResult& r, double tolerance, double alpha);

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

  // The sibling DEV knob: the self-test's BASE SEED (default 0x5EEDC0DE; accepts 0x... or decimal;
  // 0 means default).  Every seed inside the self-test is DERIVED from it, so changing it re-runs the
  // whole thing out of sample -- which is what turns "re-verify under another seed" into a command
  // instead of a source edit, and a source edit is the shape that produced this session's -dirty
  // stamp incident twice.  Same class as the trial count: logging/flow only, never a policy knob.
  unsigned long long devSeed();

  // ---- the self-test: known ground truth, pure CPU, deterministic ----
  // Returns ONE line; the shell logs it (the shell owns the logging conventions).
  std::string selfTestSummary();
}

#endif
