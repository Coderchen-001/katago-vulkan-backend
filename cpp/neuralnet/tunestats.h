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
  // declared block; a 2x MARGIN on the sigma budget this file's own argument needs.  That argument: the µs
  // clock's 1 µs quantum has to stay under the 0.1% sigma budget, and it does from a 1000 µs block upwards
  // (1/1000 = 0.1% exactly); this value is twice that.  ⚠️ The "0.19%" this comment used to quote is
  // 1 µs / 526 µs -- a block length from the era it was written: at today's 2000 µs the quantisation is
  // 0.05%, four times better (corrected 2026-10-05, batch b).
  // 🔴 IT IS A TARGET AND A READING, NOT AN ENFORCED GATE.  The pre-check SOLVES for it, `hot_us=`/`anchor_us=`
  // report what was reached, and the mechanism that used to re-measure a short shape was DELETED in batch b
  // (2026-10-05) after step 0(a) measured that the verdicts do not degrade without it
  // (`logs-t3/step0a/step0a-correction-off.md`).  Two block-length sanity louds take over the watching job,
  // and the deletion's reasoning plus its resurrection recipe live in `correction-removal-inventory.md`.
  constexpr double MIN_BLOCK_TIME_MS = 2.0;
  constexpr long   PAIRS_FLOOR = 25;         // DECLARED + conservative: 16 and 25 are indistinguishable here (see the header)
  constexpr double ALPHA = 0.05;             // declared, not hidden inside a formula
  constexpr double TOLERANCE = 0.01;         // 1% : the product tolerance
  constexpr double SPIKE_RATE_GATE = 0.15;   // MEASURED bound: 15% passes, 18% fails on THREE criteria at once (the rate12/15/18 sweep)
  constexpr long   REPS_MAX = 512;
  // ---- W4e: the two constants the MEASUREMENT primitive consumes, declared here so it has no copy ----
  // THE SPIKE BAND.  A block is a spike when its time sits inside [1+lo, 1+hi] times THAT OBJECT's
  // median -- the envelope generator's own definition (the "measured" row of ENV_CASES below, where a
  // spike is a multiplicative delay 1 + U[lo,hi) on the block).  Two reasons the numbers are here and
  // not in the backend:
  //   (i)  the 0.15 gate was MEASURED with this band, so a different pair of numbers would make 0.15 a
  //        different quantity (w4e-preview.md R-3(a): reuse the definition, do not invent one);
  //   (ii) tune-measurement.md section 2.3.1: the declared constants have ONE source, and the probe
  //        receives them as parameters -- the backend holds no second copy to drift.
  // The self-test asserts ENV_CASES' measured rows carry exactly these values, so "same definition" is
  // checked rather than promised.
  constexpr double SPIKE_BAND_LO = 0.40;
  constexpr double SPIKE_BAND_HI = 1.00;
  // ---- (7b): THE PRE-CHECK'S OWN NUMBERS (w4e-preview.md section 10 N4; ruled 2026-09-27) ----------
  // N4 registered that "the first block's `shots` and the value of `K` were never written down", and
  // R-5 turns that into a rule for whoever produces them: a constant that comes out of a calibration
  // lands WITH its derivation and its provenance.  The shape it prevents is a number nobody can
  // recompute -- this file's own history has three of them (`k_spike`, `sigma_max`, the MDE tiers) and
  // 6b37a1d deleted exactly those.  All three are in POLICY_CONSTANTS, so changing one invalidates the
  // row sets measured under the old one by construction, while TUNER_POLICY_VERSION does NOT move: this
  // is a policy change, not a protocol change (R-30 cell (v)).
  //   WARMUP_BLOCKS -- the per-eval medians in section 2.2 are 117.180 / 6.420 / 5.220 / 5.220 us for
  //   evals 1..4: the cold transient is entirely in eval 1 (18.3x), but eval 2 is STILL 23% above the
  //   plateau, so TWO warm-up blocks are what reaches it.  The primitive refuses warmupBlocks < 2 for
  //   the same reason, which makes this the measured floor rather than a tuning knob -- and the value
  //   is that floor, not something above it.
  constexpr long WARMUP_BLOCKS = 2;
  //   HOT_BLOCKS_K -- `t_ref` = t_block_hot = the MEDIAN of K hot blocks (section 2.6's sequence), so K
  //   must be big enough for a median to reject a spike: the measured spike rate is 10.6% on the
  //   calibration arm and 15% is the gate's own bound, and a median of 3 survives one spike while a
  //   "median" of 2 is just the mean (zero rejection).  Cost = K blocks = 3 x ~2 ms, so the value that
  //   buys the property is also the cheap one.  RULED 2026-09-27 (the one number N4 left open).
  constexpr long HOT_BLOCKS_K = 3;
  //   CALIB_PAIRS_FLOOR -- the PRE-CHECK's OWN sample count: how many paired differences the scale
  //   estimate (`1.4826 x MAD`) is computed from.  It is NOT `HOT_BLOCKS_K` and must never borrow it
  //   again: that constant's declared job is a MEDIAN's window ("big enough for a median to reject one
  //   spike"), and a SCALE estimator's sample count is a different question with a different answer --
  //   at n = 3 one spike IS a third of the sample (past the MAD's own 25% breakdown point), and two
  //   nearly-equal differences return sigma = 0 outright.  Measured before this cut: the same binary,
  //   machine and shape set reported sigma in {0, 0.024, 0.052, 0.072, 0.098, 0.14, 0.21, 1.31, 1.55}
  //   and `reps` followed sigma^2 to the cap, because that estimate was three points wide.
  //   DECLARED, inside two bounds that are PINNED rather than restated (smoke A30 pins the printed
  //   `calib_pairs=` and the criterion pins both inequalities):
  //     LOWER: >= HOT_BLOCKS_K.  The same probe call's `reps` IS that median's window (its
  //            `t_block_hot_us` is the median over the reps it ran), so a smaller value here would
  //            quietly downgrade a constant that is in the policy fingerprint for its own reason.
  //     UPPER: <= PAIRS_FLOOR.  This calibration carries NO claim: by the ruling of 2026-10-01 sigma and
  //            its Gaussian answer are REPORTED, and (iii) above is this header's own record that no
  //            practical assertion separates n = 16 from n = 25 (the measured gap needs ~20000 trials
  //            for 4 sigma).  Paying 25 here would buy a difference the project itself could not measure.
  //   Cost: `reps` blocks once per startup (~16 x ~2 ms), against the main measurement's 5700.
  constexpr long CALIB_PAIRS_FLOOR = 16;
  // The two bounds above are ENFORCED, not just written down (the header's own idiom, cf. the ENV_CASES
  // assert in tunestats.cpp): a later edit that makes the calibration cheaper by breaking either property
  // fails the BUILD, which is the only place this constant is read before a device is involved.
  static_assert(CALIB_PAIRS_FLOOR >= HOT_BLOCKS_K,
                "the calibration's `reps` IS the same call's median window: a value below HOT_BLOCKS_K "
                "would silently downgrade a property that constant is in the policy fingerprint for");
  static_assert(CALIB_PAIRS_FLOOR <= PAIRS_FLOOR,
                "the calibration carries no claim (ruling 2026-10-01: sigma is REPORTED), and this "
                "header records that no practical assertion separates n=16 from n=25 -- paying the "
                "declared CLAIM floor here would buy a difference the project itself could not measure");
  //   STALL_BLOCKS -- section 2.6's ruling detects a stall at SECOND granularity, and the only honest
  //   unit to convert that with is the block the protocol defines by TIME (MIN_BLOCK_TIME_MS).  Written
  //   as an expression rather than 500 so that if that block length ever moves this moves with it,
  //   instead of becoming a number nobody can recompute.  It can only ever ABORT a run (no verdict
  //   changes meaning because of it); it is in the fingerprint by ruling, so a change invalidates the
  //   row sets rather than silently mixing two regimes.
  constexpr long STALL_BLOCKS = (long) (1000.0 / MIN_BLOCK_TIME_MS);
  // 🔴 (2026-10-05, batch b) THE CORRECTION'S DEAD BAND LIVED HERE AND IS DELETED WITH THE MECHANISM IT
  // SERVED.  It was `CORRECTION_DEADBAND = 1 - 1/(1+TOLERATED_RESOLUTION_LOSS)^2` (4.82% at a 2.5% loss) and
  // it existed to stop a run that reached 99.8% of the declared block from re-measuring the whole grid.  The
  // correction itself is gone because step 0(a) MEASURED that the verdicts do not move without it while each
  // measuring leg drops from 24.9 s to 13.5 s (`logs-t3/step0a/step0a-correction-off.md`); with the mechanism
  // gone, a band on its trigger has no subject, and a policy constant nothing reads is the other half of
  // R38's one-name-one-meaning.  The block length is still watched -- by two sanity louds after the
  // measurement line, whose thresholds need no declared constant (one is derived from the sigma budget
  // `MIN_BLOCK_TIME_MS` declares, the other is measured-relative).  Resurrection recipe and the full
  // reasoning: `correction-removal-inventory.md`.
  // THE ROTATION SEED: the block sequence rotates with a randomized Latin square (tune-measurement.md
  // section 2.2, one block per (shape, object) slot), and "fixed seed, reproducible" is only true if
  // the seed is declared.  It is the same value the statistics self-test uses as its default base seed
  // (devSeed() reads it instead of repeating the literal), so one number serves both and a rerun of
  // either reproduces the other.  It is a CODE constant, not an env knob: it is covered by the build
  // revision in the policy fingerprint, and making it settable would let two runs of the same machine
  // differ in a way no header records.
  constexpr unsigned long long PROBE_ROTATION_SEED = 0x5EEDC0DEULL;
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

  // ---- W4e: the SPIKE RATE's interval (a proportion is not a location) ----
  // The judgement layer's one interval is a MEDIAN interval (hl +/- t * 1.2533 * MAD / sqrt(n)): it
  // describes a continuous location parameter and has no meaning for "k spikes in n blocks".  A rate
  // needs its own family, so it gets a declared one here rather than an anonymous formula at the call
  // site.  Wilson score interval -- analytic, alpha declared, the same alpha the decisions use.  No
  // bootstrap and no order statistics: those two were deleted for the median (they cannot survive our
  // spike shape) and a proportion does not need them; spike-source-preview.md already fixed Wilson for
  // a spike rate.
  // ⚠️ IT ASSUMES INDEPENDENT BLOCKS.  Spikes are clustered, not drifting (logs-t3/calib-b-sigma-robust.txt:
  // "散发、不是漂移"), so the effective n is smaller than the block count and this interval is if
  // anything too NARROW.  Clustering-aware estimation is registered, not implemented (w4e-preview.md
  // R-4(ii)): until then a caller that reports this interval must say that it is the i.i.d. one.
  struct RateInterval {
    double low;
    double high;
    double p;      // the point estimate hits/trials
    bool ok;       // false when trials == 0 (an empty denominator is not a rate of zero)
  };
  RateInterval rateInterval(long hits, long trials, double alpha);
  // W4e 2c-c: the generator's ARM (one object's block times) as a series, injected with a rate the
  // caller KNOWS.  It exists so the measurement primitive's self-test can point the RUNTIME spike
  // detector at a series whose answer is known in advance: "the detector and the envelope share a
  // band" is then a check rather than a resemblance (w4e-preview.md R-3(a), R-4(i)).  The
  // injection has one definition -- this loop calls the same arm draw the envelope's pairs use --
  // and the draw order is part of the contract (see the .cpp), so a refactor that reorders it
  // changes the envelope's numbers and the self-test says so.
  std::vector<double> spikeSyntheticArm(long n, double base, double sigmaRel, double spikeProb,
                                        double spikeLo, double spikeHi, unsigned long long seed);

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
    std::vector<long> repsFormula;   // the Gaussian formula's answer: REPORTED ONLY (ruling 2026-10-01)
    std::vector<long> repsRequired;  // == floorReps, ALWAYS: the DECLARED floor is the budget
    long totalReps;
    long floorReps;
    bool anyClampedAtMax;
  };
  // 🔴 THE RULE (ruling 2026-10-01): reps_s = floorReps, ALWAYS.  The Gaussian answer
  // `ceil((z_{1-alpha/2}*sigmaRel_s/tolerance)^2)` is REPORTED in `repsFormula` and never obeyed: its input
  // is a three-point MAD, `reps ∝ sigma^2` turns one outlier in that estimate into ~20x the work, and this
  // project's own measurement (tune-measurement.md section 2.5) puts the true block-relative scale at
  // 0.057-0.092% at n = 50 -- R about 1 at the 1% tolerance, so the noise is not the constraint and the
  // test's own sample floor is.  `anyClampedAtMax` is therefore an ALARM ABOUT THE ESTIMATE ("the formula
  // asked for more than maxReps"), not a budget anyone pays.
  // sigmaRel_s is the BLOCK-relative paired scale: see tune-measurement.md section 2.5 -- the
  // per-dispatch-relative figure benchgemm prints is NOT the right normalisation.
  Budget allocateReps(const std::vector<double>& sigmaRel, double tolerance, double alpha,
                      long floorReps, long maxReps);

  // 🔴 P3 (2026-10-01): THE WIDE-BAND RECOVERY -- "buy resolution, do not give up".  The judgement's
  // interval needs at least FOUR non-zero paired differences (`pairedSignedRank` leaves `ciOk` false
  // below that), so a pair whose differences are mostly exact zeros with a few entries WIDER than the
  // tolerance ends in `ciMissingWide`: no interval at all, and the branches can neither claim a win nor
  // honestly declare "nothing won".  The old answer was to write nothing and pay for the measurement
  // again at the next startup -- a shape the ruling forbids.  The recovery instead reads the band the
  // run ACTUALLY observed: the interval half-width falls as 1/sqrt(n), so the pair count that would
  // bring an observed band `band` down to `tolerance` is `n' = ceil(n * (band / tolerance)^2)`.
  //   * It GROWS the budget or does nothing: a band at or below the tolerance is not this rule's case
  //     (that is what the bounded-interval rule is for), and the answer is never below `reps`.
  //   * It is clamped at `maxReps` -- the declared cap -- and the caller REPORTS both numbers; hitting
  //     the cap is an alarm about the device, never a reason to refuse to write (`REPS_MAX` is an
  //     alarm-only reading by ruling).
  //   * It is arithmetic, not a policy knob: the two inputs come from this run's own measurement.
  long repsForWideBand(long reps, double band, double tolerance, long maxReps);

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
