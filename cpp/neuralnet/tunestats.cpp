#include "../neuralnet/tunestats.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <sstream>

using namespace std;

namespace TuneStats {

  // ---------------------------------------------------------------- helpers: RNG + normal
  namespace {
    // xorshift64* -- deterministic and portable, which is what makes the self-test's numbers
    // reproducible evidence rather than a story about one run.
    struct Rng {
      uint64_t s;
      explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
      uint64_t next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 2685821657736338717ULL;
      }
      double uniform() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
      double normal() {
        double u1 = uniform(); double u2 = uniform();
        if (u1 < 1e-12) u1 = 1e-12;
        return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
      }
    };

    // Acklam's inverse normal CDF: the ALLOCATOR needs z_{1-alpha/2} for a DECLARED alpha, and a
    // hard-coded 1.96 would quietly contradict that declaration.
    double normInv(double p) {
      static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                                   1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
      static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                                   6.680131188771972e+01, -1.328068155288572e+01};
      static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                                  -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
      static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                                  3.754408661907416e+00};
      const double pLow = 0.02425, pHigh = 1.0 - pLow;
      double q, r;
      if (p <= 0.0) return -1e9;
      if (p >= 1.0) return 1e9;
      if (p < pLow) {
        q = sqrt(-2.0 * log(p));
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
      }
      if (p > pHigh) {
        q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
      }
      q = p - 0.5; r = q * q;
      return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q /
             (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
    }

    // Regularised incomplete beta I_x(a,b), Lentz continued fraction (the textbook standard part).
    // Needed to invert Student's t: the interval's scale is ESTIMATED from the same n pairs, so the
    // multiplier must be the small-sample one, and t's CDF is exactly this beta.
    double betacf(double a, double b, double x) {
      const int MAXIT = 300;
      const double EPS = 3.0e-14, FPMIN = 1.0e-300;
      const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
      double c = 1.0;
      double d = 1.0 - qab * x / qap;
      if (fabs(d) < FPMIN) d = FPMIN;
      d = 1.0 / d;
      double h = d;
      for (int m = 1; m <= MAXIT; m++) {
        const double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        const double del = d * c; h *= del;
        if (fabs(del - 1.0) < EPS) break;
      }
      return h;
    }

    double betai(double a, double b, double x) {
      if (x <= 0.0) return 0.0;
      if (x >= 1.0) return 1.0;
      const double bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x));
      if (x < (a + 1.0) / (a + b + 2.0)) return bt * betacf(a, b, x) / a;
      return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
    }

    // P(T <= t) for Student's t with df degrees of freedom.
    double tCdf(double t, double df) {
      const double x = df / (df + t * t);
      const double half = 0.5 * betai(0.5 * df, 0.5, x);
      return t > 0.0 ? 1.0 - half : half;
    }

    vector<double> walshAverages(const vector<double>& d) {
      vector<double> w;
      w.reserve(d.size() * (d.size() + 1) / 2);
      for (size_t i = 0; i < d.size(); i++)
        for (size_t j = i; j < d.size(); j++)
          w.push_back(0.5 * (d[i] + d[j]));
      sort(w.begin(), w.end());
      return w;
    }

    // Exact two-sided signed-rank p-value from the null distribution of W+ (no ties assumed): a DP
    // over subset sums of ranks 1..n.  Exact for n <= 25, far beyond the repetition floor we budget
    // for, so in practice an untied decision always uses the exact branch.
    double signedRankExactP(size_t n, double wPlus) {
      const size_t maxW = n * (n + 1) / 2;
      vector<double> cnt(maxW + 1, 0.0);
      cnt[0] = 1.0;
      for (size_t r = 1; r <= n; r++)
        for (size_t w = maxW; w >= r; w--)
          cnt[w] += cnt[w - r];
      double total = 0.0;
      for (size_t w = 0; w <= maxW; w++) total += cnt[w];
      double below = 0.0, above = 0.0;
      const long wl = (long)llround(wPlus);
      for (size_t w = 0; w <= maxW; w++) {
        if ((long)w <= wl) below += cnt[w];
        if ((long)w >= wl) above += cnt[w];
      }
      double p = 2.0 * min(below, above) / total;
      return p > 1.0 ? 1.0 : p;
    }
  }

  // ---------------------------------------------------------------- descriptive
  double median(vector<double> v) {
    if (v.empty()) return 0.0;
    sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }

  double madScale(const vector<double>& v) {
    if (v.empty()) return 0.0;
    const double m = median(v);
    vector<double> dev;
    dev.reserve(v.size());
    for (size_t i = 0; i < v.size(); i++) dev.push_back(fabs(v[i] - m));
    return 1.4826 * median(dev);
  }

  double tQuantile(double p, double df) {
    if (df <= 0.0) df = 1.0;
    if (p <= 0.0) return -1e9;
    if (p >= 1.0) return 1e9;
    if (p == 0.5) return 0.0;
    // Bisection, not an asymptotic expansion: the CDF is monotone, a bracket is found by doubling,
    // and the result is then good to the double's own resolution.  NO fitted constant enters here.
    const bool upper = p > 0.5;
    const double target = upper ? p : 1.0 - p;
    double lo = 0.0, hi = 1.0;
    while (tCdf(hi, df) < target && hi < 1e12) hi *= 2.0;
    for (int it = 0; it < 300; it++) {
      const double mid = 0.5 * (lo + hi);
      if (tCdf(mid, df) < target) lo = mid; else hi = mid;
      if (hi - lo <= 1e-14 * (1.0 + hi)) break;
    }
    const double q = 0.5 * (lo + hi);
    return upper ? q : -q;
  }

  // ---------------------------------------------------------------- the paired test
  PairedResult pairedSignedRank(const vector<double>& dIn, double alpha) {
    PairedResult r;
    r.pairs = 0; r.zerosDropped = 0; r.hl = 0.0;
    r.ciLow = r.ciHigh = 0.0; r.ciOk = false;
    r.pValue = 1.0; r.pExact = false; r.wPlus = 0.0;

    vector<double> d;
    for (size_t i = 0; i < dIn.size(); i++) {
      if (dIn[i] == 0.0) { r.zerosDropped++; continue; }
      d.push_back(dIn[i]);
    }
    r.pairs = d.size();
    if (d.empty()) return r;
    const size_t n = d.size();

    // ---- the statistic: W+ over AVERAGE ranks of |d| (what ties require) ----
    vector<double> absd;
    absd.reserve(n);
    for (size_t i = 0; i < n; i++) absd.push_back(fabs(d[i]));
    vector<size_t> order(n);
    for (size_t i = 0; i < n; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](size_t a, size_t b) { return absd[a] < absd[b]; });
    vector<double> rank(n, 0.0);
    bool hasTies = false;
    double tieSum = 0.0;
    size_t i = 0;
    while (i < order.size()) {
      size_t j = i;
      while (j + 1 < order.size() && absd[order[j + 1]] == absd[order[i]]) j++;
      const double avg = 0.5 * (double)(i + 1 + j + 1);
      for (size_t k = i; k <= j; k++) rank[order[k]] = avg;
      const double t = (double)(j - i + 1);
      if (j > i) hasTies = true;
      tieSum += t * t * t - t;
      i = j + 1;
    }
    double wPlus = 0.0;
    for (size_t k = 0; k < n; k++) if (d[k] > 0.0) wPlus += rank[k];
    r.wPlus = wPlus;

    // ---- p-value: exact when untied and small, else the normal approximation with the standard tie
    //      correction.  Which branch produced it is reported. ----
    const double meanW = n * (n + 1) / 4.0;
    const double varW = n * (n + 1) * (2.0 * n + 1) / 24.0 - tieSum / 48.0;
    if (!hasTies && n <= 25) {
      r.pValue = signedRankExactP(n, wPlus);
      r.pExact = true;
    } else if (varW <= 0.0) {
      r.pValue = 1.0;
    } else {
      const double z = (wPlus - meanW);
      const double zc = (fabs(z) - 0.5) * (z >= 0 ? 1.0 : -1.0);   // continuity correction
      r.pValue = erfc(zc / sqrt(2.0 * varW));
      if (r.pValue > 1.0) r.pValue = 1.0;
    }

    // ---- the estimate ----
    r.hl = median(walshAverages(d));

    if (n >= 4) {
      // CI: the median +- t_{n-1,1-alpha/2} * 1.2533 * MAD / sqrt(n).
      // NEITHER scale constant is fitted, and that matters: a multiplier tuned against the same
      // spike model that validates it would make "n=25 passes" an in-sample claim.
      //   1.4826 = 1 / Phi^-1(0.75): the MAD's consistency factor for a normal scale (textbook)
      //   1.2533 = sqrt(pi/2): SE(median) / SE(mean) for a normal distribution, since
      //            SE(median) = 1/(2 f(0) sqrt(n)) and f(0) = 1/(sigma sqrt(2 pi))
      //   t_{n-1}: the scale is ESTIMATED from these n pairs, so the Gaussian z would claim more
      //            confidence than the data supports.  This is a small-sample CORRECTION, not a fit:
      //            the degrees of freedom follow from n, and alpha is the declared one.
      //            WHAT IT IS NOT: a median's sampling distribution is not t, and 1.4826/1.2533 are
      //            normal-world consistency factors -- the multiplier is a HEURISTIC whose warrant is
      //            MEASURED coverage.  "Theory says t, so n=16 is fine" is not an argument here.
      // The only quantity the sweep chooses is the pair count, and that choice is then re-tested OUT
      // OF SAMPLE (another seed, shapes around the measured one) before anything calls it validated.
      const double sig = madScale(d);
      const double half = tQuantile(1.0 - alpha / 2.0, (double)n - 1.0) * 1.2533 * sig / sqrt((double)n);
      r.ciLow = r.hl - half;
      r.ciHigh = r.hl + half;
      r.ciOk = true;
    }
    return r;
  }

  // The spike rate's interval (declaration and rationale in the header).  Wilson score interval:
  // the same alpha the decisions use, no fitted constant, and it stays inside [0,1] at the edges
  // (k = 0 and k = n are exactly where a normal approximation would leave the interval).
  RateInterval rateInterval(long hits, long trials, double alpha) {
    RateInterval r { 0.0, 0.0, 0.0, false };
    if (trials <= 0) return r;
    if (hits < 0) hits = 0;
    if (hits > trials) hits = trials;
    const double n = (double) trials;
    const double p = (double) hits / n;
    const double z = normInv(1.0 - alpha / 2.0);
    const double z2 = z * z;
    const double denom = 1.0 + z2 / n;
    const double center = (p + z2 / (2.0 * n)) / denom;
    const double half = (z / denom) * sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n));
    r.p = p;
    r.low = center - half;
    r.high = center + half;
    if (r.low < 0.0) r.low = 0.0;
    if (r.high > 1.0) r.high = 1.0;
    r.ok = true;
    return r;
  }

  bool intervalCovers(const PairedResult& r, double value) {
    return r.ciOk && r.ciLow <= value && r.ciHigh >= value;
  }

  bool notWorseThan(const PairedResult& r, double tolerance) {
    if (!r.ciOk) return false;               // fail closed: no interval, no claim
    return r.ciLow >= -tolerance;
  }

  bool equivalentWithin(const PairedResult& r, double tolerance) {
    if (!r.ciOk) return false;
    return r.ciLow >= -tolerance && r.ciHigh <= tolerance;
  }

  bool decisionUnstable(const PairedResult& r, double tolerance, double alpha) {
    if (!r.ciOk || r.pairs < 2) return true;                    // no interval => nothing is decided
    const double tq = tQuantile(1.0 - alpha / 2.0, (double)r.pairs - 1.0);
    if (!(tq > 0.0)) return true;
    const double se = (r.ciHigh - r.ciLow) / (2.0 * tq);         // half-width = t * SE
    if (!(se > 0.0)) return true;
    return fabs(r.ciLow + tolerance) <= STABILITY_BOUNDARY_SE * se;
  }

  // ---------------------------------------------------------------- the budget
  Budget allocateReps(const vector<double>& sigmaRel, double tolerance, double alpha,
                      long floorReps, long maxReps) {
    Budget b;
    b.floorReps = floorReps;
    b.totalReps = 0;
    b.anyClampedAtMax = false;
    // The FORMULA is the Gaussian face of the same problem (z, not t) and it is REPORTED, not obeyed:
    // under our measured spikes it is systematically optimistic (the scale the formula assumes is the
    // one the spikes inflate), which is exactly why the REQUIRED count is the max with the MEASURED
    // floor.  Printing both is what keeps "the floor is doing the work here" visible instead of
    // implied by a single number that looks like a formula's output.
    const double z = normInv(1.0 - alpha / 2.0);
    for (size_t s = 0; s < sigmaRel.size(); s++) {
      long reps;
      if (tolerance <= 0.0 || sigmaRel[s] <= 0.0) { reps = floorReps; }
      else {
        const double need = (z * sigmaRel[s] / tolerance);
        reps = (long)ceil(need * need);
      }
      if (reps < 1) reps = 1;
      b.repsFormula.push_back(reps);
      long req = reps < floorReps ? floorReps : reps;
      if (req > maxReps) { req = maxReps; b.anyClampedAtMax = true; }
      b.repsRequired.push_back(req);
      b.totalReps += req;
    }
    return b;
  }

  unsigned long long devSeed() {
    const unsigned long long kDefault = PROBE_ROTATION_SEED;   // ONE source: the rotation's declared seed
    const char* v = getenv("KATAGO_VK_TUNE_STATS_SEED");
    if (v == nullptr || v[0] == '\0') return kDefault;
    const unsigned long long parsed = strtoull(v, nullptr, 0);   // base 0: accepts 0x... and decimal
    return parsed == 0ull ? kDefault : parsed;
  }

  long devTrials() {
    long t = DEV_TRIALS_DEFAULT;
    const char* v = getenv("KATAGO_VK_TUNE_STATS_TRIALS");
    if (v != nullptr && v[0] != '\0') {
      const long parsed = strtol(v, nullptr, 10);
      if (parsed > 0) t = parsed;
    }
    if (t < 10) t = 10;
    if (t > 20000) t = 20000;
    return t;
  }

  // ---------------------------------------------------------------- the self-test
  // Ground truth is INJECTED, so every number below is a MEASUREMENT of the algorithm:
  //   coverage    : how often the nominal 95% interval really contains the true effect
  //   power_at_tol: how often a candidate 1.01x the tolerance worse is refused
  //   power@noise : how often a candidate only noise-level worse is refused (should be LOW)
  //   spikes+-    : whether the verdict agrees with the spike-free verdict
  //   sentinel    : how often an honest A/A comparison is declared broken (should be LOW) AND how
  //                 often a REAL bias at the tolerance is caught (should be HIGH -- a sentinel with
  //                 no power is indistinguishable from a device that is never biased)
  // ...and the pair-count choice is re-tested on shapes and seeds the sweep did not select on.
  namespace {
    // One synthetic pair difference: two independent blocks, each base*(1 + N(0,sigma) + spike).
    // Both arms draw their spike uniforms even when spikeProb is 0, so the spike-free variant sees
    // the SAME normal draws -- the comparison is paired at the generator level too.
    void synthDifferences(vector<double>& d, size_t pairs, double base, double sigmaRel,
                          double spikeProb, double spikeLo, double spikeHi, double effect, Rng& rng) {
      d.clear();
      d.reserve(pairs);
      for (size_t k = 0; k < pairs; k++) {
        double a = base * (1.0 + sigmaRel * rng.normal());
        double b = base * (1.0 + sigmaRel * rng.normal());
        const double ua = rng.uniform();
        const double ub = rng.uniform();
        const double ra = rng.uniform();
        const double rb = rng.uniform();
        if (ua < spikeProb) a += base * (spikeLo + (spikeHi - spikeLo) * ra);
        if (ub < spikeProb) b += base * (spikeLo + (spikeHi - spikeLo) * rb);
        d.push_back((a - b) + effect * base);
      }
    }

    struct Rates {
      double coverage, powerTol, powerNoise, spikesWith, spikesWithout, sentinelFalse, sentinelPower,
             sentinelPowerHalf, ciWidth;
    };

    template <typename Pred>
    double runRate(size_t trials, size_t pairs, double base, double sigmaRel, double spikeProb,
                   double spikeLo, double spikeHi, double effect, double alpha,
                   unsigned long long seed, Pred pred, double* widthOut) {
      // PER-TRIAL SEED DERIVATION (2026-09-21, correction batch): the random stream must be a
      // function of the TRIAL, not of the call.  With one Rng carried across trials, an arm with more
      // pairs consumes more of it per trial, so from trial 2 onwards the two arms sit at different
      // positions in the stream -- which makes "the same seed for both arms" a NAME rather than common
      // random numbers, and their difference as noisy as two independent runs.  Deriving the stream per
      // trial makes both arms see the SAME draws for every pair they share.
      vector<double> d;
      size_t hit = 0;
      vector<double> widths;
      if (widthOut != nullptr) widths.reserve(trials);
      for (size_t t = 0; t < trials; t++) {
        Rng rng(seed + 0x9E3779B97F4A7C15ULL * (unsigned long long)(t + 1));
        synthDifferences(d, pairs, base, sigmaRel, spikeProb, spikeLo, spikeHi, effect, rng);
        const PairedResult r = pairedSignedRank(d, alpha);
        if (pred(r)) hit++;
        if (widthOut != nullptr && r.ciOk) widths.push_back(r.ciHigh - r.ciLow);
      }
      if (widthOut != nullptr) *widthOut = median(widths);
      return (double)hit / (double)trials;
    }

    Rates measure(size_t trials, size_t pairs, double alpha, double tolerance, double sigmaRel,
                  double spikeProb, double spikeLo, double spikeHi, double base,
                  unsigned long long seed) {
      Rates r;
      const double effectCover = 0.002;
      r.coverage = runRate(trials, pairs, base, sigmaRel, spikeProb, spikeLo, spikeHi, effectCover,
                           alpha, seed, [&](const PairedResult& p) { return intervalCovers(p, effectCover); },
                           &r.ciWidth);
      r.powerTol = runRate(trials, pairs, base, sigmaRel, spikeProb, spikeLo, spikeHi, -1.01 * tolerance,
                           alpha, seed + 11, [&](const PairedResult& p) { return !notWorseThan(p, tolerance); },
                           nullptr);
      r.powerNoise = runRate(trials, pairs, base, sigmaRel, spikeProb, spikeLo, spikeHi, -sigmaRel,
                             alpha, seed + 23, [&](const PairedResult& p) { return !notWorseThan(p, tolerance); },
                             nullptr);
      r.spikesWith = runRate(trials, pairs, base, sigmaRel, spikeProb, spikeLo, spikeHi, -0.5 * tolerance,
                             alpha, seed + 37, [&](const PairedResult& p) { return notWorseThan(p, tolerance); },
                             nullptr);
      r.spikesWithout = runRate(trials, pairs, base, sigmaRel, 0.0, spikeLo, spikeHi, -0.5 * tolerance,
                                alpha, seed + 37, [&](const PairedResult& p) { return notWorseThan(p, tolerance); },
                                nullptr);
      // THE SENTINEL HAS TWO JOBS, and only the first one was measured before:
      //   false-failure rate on an honest A/A (must be low) and POWER against a real bias (must be
      //   high).  Without the second, "the sentinel stayed quiet" cannot be told apart from "the
      //   sentinel has no power" -- the textbook "passed because nothing was measured".
      // The power is measured where it belongs: at a bias equal to the tolerance.  A bias SMALLER
      // than the tolerance is deliberately NOT flagged, because the equivalence check is a claim
      // about the tolerance, not about zero -- that is why the half-tolerance case is reported too.
      r.sentinelFalse = runRate(trials, pairs, base, sigmaRel, spikeProb, spikeLo, spikeHi, 0.0,
                                alpha, seed + 53, [&](const PairedResult& p) { return !equivalentWithin(p, tolerance); },
                                nullptr);
      r.sentinelPower = runRate(trials, pairs, base, sigmaRel, spikeProb, spikeLo, spikeHi, 1.01 * tolerance,
                                alpha, seed + 61, [&](const PairedResult& p) { return !equivalentWithin(p, tolerance); },
                                nullptr);
      r.sentinelPowerHalf = runRate(trials, pairs, base, sigmaRel, spikeProb, spikeLo, spikeHi, 0.5 * tolerance,
                                    alpha, seed + 67, [&](const PairedResult& p) { return !equivalentWithin(p, tolerance); },
                                    nullptr);
      return r;
    }

    // The SIX criteria.  They are deliberately shape-agnostic thresholds on measured rates -- no
    // threshold here is a fitted scale, which is what lets the same function judge shapes that were
    // never used to pick the pair count.
    bool passes(const Rates& r) {
      // 0.90 here is the BAND'S LOWER EDGE for the six criteria -- NOT COVERAGE_TARGET (0.93), which
      // is a separate requirement asserted in the self-test with a trial-count-scaled margin.  Two
      // numbers, two meanings, two names: do not merge them.
      return r.coverage >= 0.90 && r.coverage <= 0.99 && r.powerTol >= 0.80 &&
             r.powerNoise <= 0.20 && fabs(r.spikesWith - r.spikesWithout) <= 0.05 &&
             r.sentinelFalse <= 0.05 && r.sentinelPower >= 0.80;
    }

    // EVERY criterion this row violates, from a FIXED list -- not just the first one checked.
    //
    // The single-name version this replaces was ORDER-DEPENDENT, and that is not a wording problem: on
    // the gate row n=16 misses coverage by 0.001 and the sentinel by ~78%, so "refused_by=cov" and
    // "refused_by=snF" are the SAME row depending on which check runs first.  One of those readings says
    // "precision is slightly off" and the other says "sentinel trustworthiness is far off", and they
    // support different sentences about what the floor of 25 buys.  Membership is the meaning; the order
    // is fixed only so two runs of the same row print the same string.
    //
    // The separator is '+' and NOT a comma: the pairfloor block is comma-separated, so a comma inside a
    // field would break every parser that keys on that shape.
    std::string failingCriteria(const Rates& r) {
      std::string s;
      auto add = [&s](const char* name) { if (!s.empty()) s += "+"; s += name; };
      if (r.coverage < 0.90 || r.coverage > 0.99) add("cov");
      if (r.powerTol < 0.80) add("pw");
      if (r.powerNoise > 0.20) add("pn");
      if (fabs(r.spikesWith - r.spikesWithout) > 0.05) add("sp");
      if (r.sentinelFalse > 0.05) add("snF");
      if (r.sentinelPower < 0.80) add("snP");
      if (s.empty()) s = "none";
      return s;
    }

    // The applicability ENVELOPE: the conclusion only holds inside the noise shapes that were
    // synthesised, so the same six criteria are re-run on shapes around the measured one.  Each row
    // declares its EXPECTED verdict, and the gate constant must agree with the rows -- a gate that
    // says "up to 15%" while a 15% row is declared FAIL (or a passing 18% row sits above it) is a
    // dead judgement, and the self-test reddens instead of printing it politely.
    struct EnvCase { const char* name; double p; double lo; double hi; unsigned long long seedOff; bool expectPass; };
    // The rows whose band IS the measured one reference SPIKE_BAND_LO/HI instead of repeating 0.4/1.0:
    // the runtime detector is fed those same two constants by the probe's caller, so "the gate was
    // measured with the detector's band" is a single value rather than two literals that happen to
    // agree today.  `mild` and `big` deliberately carry OTHER bands (fewer/smaller and larger spikes):
    // they are envelope cases, not the regime the gate was measured in, and they must not be folded
    // into the measured band.
    constexpr EnvCase ENV_CASES[8] = {
      { "measured",  0.106, SPIKE_BAND_LO, SPIKE_BAND_HI, 0ull,      true  },   // the measured regime (spike_p 10.6%, +40..100%)
      { "alt-seed",  0.106, SPIKE_BAND_LO, SPIKE_BAND_HI, 0x1234ull, true  },   // the same shape, an independent seed (out of sample)
      { "mild",      0.050, 0.2, 0.5, 0x2345ull, true  },   // fewer, smaller spikes
      { "big",       0.106, 1.0, 3.0, 0x3456ull, true  },   // the same rate, much larger spikes
      { "rate12",    0.120, SPIKE_BAND_LO, SPIKE_BAND_HI, 0x5555ull, true  },   // the gap between the measured rate and the gate
      { "rate15",    0.150, SPIKE_BAND_LO, SPIKE_BAND_HI, 0x6666ull, true  },   // the gate itself
      { "rate18",    0.180, SPIKE_BAND_LO, SPIKE_BAND_HI, 0x7777ull, false },   // above the gate: must FAIL
      { "harsh20",   0.200, SPIKE_BAND_LO, SPIKE_BAND_HI, 0x4567ull, false },   // double the measured rate: must FAIL
    };
    static_assert(ENV_CASES[0].lo == SPIKE_BAND_LO && ENV_CASES[0].hi == SPIKE_BAND_HI,
                  "the measured envelope row must carry the runtime spike band: the 0.15 gate was "
                  "measured with THIS band (w4e-preview.md R-3(a)), so a row that drifts off it makes "
                  "the gate a statement about a detector nobody runs");
  }

  string selfTestSummary() {
    const double alpha = ALPHA;
    const double tolerance = TOLERANCE;
    const double sigmaRel = 0.001;      // 0.1% : the BLOCK-relative paired scale measured on the 5080
    const double base = 1.0;
    const size_t trials = (size_t)devTrials();
    const size_t pairs = (size_t)PAIRS_FLOOR;
    // The BASE seed is a dev knob: every seed below is DERIVED from it, so one env value re-runs the
    // whole self-test out of sample -- the "re-verify under another seed" discipline, executable.
    const unsigned long long seed = devSeed();

    // (0) the multiplier against PUBLISHED table values: the t quantile is a textbook part, and this
    //     is what stops it from being a claim.  The tolerance separates it from a wrong multiplier
    //     (z at df=24 differs by 0.10) by three orders of magnitude.
    struct TCase { double p; double df; double want; };
    const TCase TCASES[4] = { {0.975, 1.0,   12.706204736},
                              {0.975, 8.0,    2.306004135},
                              {0.975, 24.0,   2.063898562},
                              {0.975, 100.0,  1.983971519} };
    double tErr = 0.0;
    ostringstream tt;
    for (int i = 0; i < 4; i++) {
      const double got = tQuantile(TCASES[i].p, TCASES[i].df);
      tErr = max(tErr, fabs(got - TCASES[i].want));
      tt << (i ? " " : "") << "df=" << (int)TCASES[i].df << ":" << setprecision(9) << got;
    }

    // (1) the pair-count evidence at the measured shape.  n=9 must FAIL (the negative control that
    //     keeps the criteria from being vacuous), n=25 must pass, and n=16 is quoted on an
    //     INDEPENDENT seed -- the out-of-sample reading that keeps the floor at 25: it lands ON the
    //     0.93 line (0.927 at 1000 trials, 0.933 at 4000) while 25 sits at 0.943-0.945.
    const Rates n9  = measure(trials, 9,    alpha, tolerance, sigmaRel, 0.106, 0.4, 1.0, base, seed);
    const Rates n16 = measure(trials, 16,   alpha, tolerance, sigmaRel, 0.106, 0.4, 1.0, base, seed + 0x9E37ull);
    const Rates n25 = measure(trials, pairs, alpha, tolerance, sigmaRel, 0.106, 0.4, 1.0, base, seed);

    // (2) OUT OF SAMPLE: the same six criteria on shapes and seeds around the measured one, with no
    //     re-selection, plus the gate row check.
    bool envPass[8];
    bool envPass16[8];
    bool envExpect[8];
    ostringstream env;
    ostringstream pf;
    int envOkCount = 0;
    int envMismatch = 0;
    int pfBothFailInsideGate = 0, pfFailInsideGate = 0, pfFailAboveGate = 0, pfPassAll = 0;
    for (int i = 0; i < 8; i++) {
      const Rates r = measure(trials, pairs, alpha, tolerance, sigmaRel, ENV_CASES[i].p,
                              ENV_CASES[i].lo, ENV_CASES[i].hi, base, seed + ENV_CASES[i].seedOff);
      envPass[i] = passes(r);
      if (envPass[i]) envOkCount++;
      envExpect[i] = (ENV_CASES[i].p <= SPIKE_RATE_GATE);
      if (envExpect[i] != envPass[i]) envMismatch++;
      // THE SAME STREAM FOR BOTH ARMS (see runRate): with per-trial seed derivation, n=16 and n=25 see
      // the same draws for the pairs they share, so this is common random numbers rather than "the same
      // seed, twice".  The question is deliberately BINARY -- does 16 pass the six criteria on the rows
      // at or below the gate -- because the CONTINUOUS difference is unanswerable (its sign flips
      // between seeds).  Nothing here is asserted: it is evidence, classified into the four
      // pre-registered verdicts so that nobody can report a single convenient row.
      const Rates r16e = measure(trials, 16, alpha, tolerance, sigmaRel, ENV_CASES[i].p,
                                 ENV_CASES[i].lo, ENV_CASES[i].hi, base, seed + ENV_CASES[i].seedOff);
      envPass16[i] = passes(r16e);
      // The rows ABOVE the gate are NEGATIVE CONTROLS: n25 is REQUIRED to fail there, so "both failed"
      // carries no information about the floor.  Only a both-fail INSIDE the gate would be a problem --
      // with the floor, or with the gate itself.  (First version classified outside-gate both-fails and
      // therefore always reported gate-problem, which is a verdict that can never be acted on.)
      if (ENV_CASES[i].p <= SPIKE_RATE_GATE) {
        if (!envPass[i] && !envPass16[i]) pfBothFailInsideGate++;
        else if (!envPass16[i]) pfFailInsideGate++;
        else pfPassAll++;
      } else {
        if (!envPass16[i]) pfFailAboveGate++;
        else pfPassAll++;
      }
      pf << (i ? "," : "") << ENV_CASES[i].name << ":n16=" << (envPass16[i] ? "OK" : "FAIL")
         << "/n25=" << (envPass[i] ? "OK" : "FAIL");
      if (!envPass16[i]) {
        // The MEASUREMENTS, not just the verdict: an OK/FAIL alone cannot support a sentence about what
        // the floor buys.  Appended AFTER the n16/n25 pair so the parsers keyed on that shape keep working.
        pf << " n16_fail=" << failingCriteria(r16e)
           << " n16_cov=" << setprecision(3) << r16e.coverage
           << " n16_pn=" << r16e.powerNoise
           << " n16_snF=" << r16e.sentinelFalse
           << " n16_sp=" << fabs(r16e.spikesWith - r16e.spikesWithout) << setprecision(4);
      }
      env << (i ? "," : "") << ENV_CASES[i].name << (envPass[i] ? ":OK" : ":FAIL")
          << "(cov=" << setprecision(3) << r.coverage << " pn=" << r.powerNoise
          << " snF=" << r.sentinelFalse << " snP=" << r.sentinelPower
          << " sp=" << fabs(r.spikesWith - r.spikesWithout) << ")" << setprecision(4);
    }

    // (3) the budget: two block-relative scales, so "the measured floor is what sets the count" is
    //     visible instead of implied.  The formula is the Gaussian face and is REPORTED; the count
    //     that is run is the max with the measured floor.
    vector<double> sig;
    sig.push_back(sigmaRel);   // 0.1% : the measured block-relative scale
    sig.push_back(0.01);       // 1%   : a scale an order of magnitude worse, i.e. formula-bound
    const Budget bud = allocateReps(sig, tolerance, alpha, PAIRS_FLOOR, REPS_MAX);

    // (4) the coverage TARGET, asserted with the SE of THIS run's trial count: a fixed 0.93 would be
    //     a coin flip at the default 200 trials (17.6% false red) and trivially green at 4000.  n=16
    //     is only REPORTED, with its distance to the target in sigmas: no practical assertion
    //     separates 16 from 25 (0.010-0.012 needs ~20000 trials for 4 sigma), and a "must be below"
    //     clause would be red ~22% of the time.  What catches a floor lowered to 16 is the
    //     DECLARATION (smoke A44 pins pairs_floor=25), not a statistic.
    // se25 is a PLUG-IN estimate: it uses the coverage THIS run measured, not the target -- which is
    // exactly why the resulting floor floats with the observation.  State that as the REASON, and quote
    // the printed min_ok= instead of hand-computing the number (two reviewers have now got it wrong by
    // hand; the print is the only trustworthy source).
    const double se25 = sqrt(n25.coverage * (1.0 - n25.coverage) / (double)trials);
    const double covMinOk = COVERAGE_TARGET - 3.0 * se25;
    // SCOPE: this is the MEASURED row's coverage only (see COVERAGE_TARGET's note (iv)).  The envelope
    // rows are judged by passes(), not by this line -- at 4000 trials the line is 0.9191 and the gate
    // row's own coverage is 0.921-0.928, so extending it there would be 0.4-2.1 sigma of headroom.
    const bool n25CoverageOk = n25.coverage >= covMinOk;
    const bool loose = trials < (size_t)COVERAGE_STRICT_MIN_TRIALS;
    const double se16 = sqrt(n16.coverage * (1.0 - n16.coverage) / (double)trials);
    const double vsTarget16 = se16 > 0.0 ? (n16.coverage - COVERAGE_TARGET) / se16 : 0.0;
    const double deltaN = n25.coverage - n16.coverage;
    const double seDelta = sqrt(se25 * se25 + se16 * se16);
    const double sigmaDelta = seDelta > 0.0 ? deltaN / seDelta : 0.0;

    // (5) THE STABILITY RULE, PROVED IN BOTH DIRECTIONS (2026-09-21): a borderline series must be
    //     flagged as UNRESOLVED and a comfortable one must NOT be -- a rule that flags everything, or
    //     nothing, would pass a one-sided test.  The effects are built in units of the run's own SE,
    //     recovered from a pilot interval (half-width / t), and the borderline case uses t*SE: with
    //     `CI_low + tol ~ (k - t) * SE` and t_24 ~ 2.06, a k of 1 would land at 1.06*SE and NOT be
    //     flagged -- a mistake worth leaving in the comment.
    double stabilityBorder = 0.0, stabilityClear = 0.0;
    {
      Rng rngPilot(seed + 0x51ABull);
      vector<double> dPilot;
      synthDifferences(dPilot, pairs, base, sigmaRel, 0.106, 0.4, 1.0, -tolerance, rngPilot);
      const PairedResult rPilot = pairedSignedRank(dPilot, alpha);
      const double tq = tQuantile(1.0 - alpha / 2.0, (double)pairs - 1.0);
      const double sePilot = rPilot.ciOk ? (rPilot.ciHigh - rPilot.ciLow) / (2.0 * tq) : 0.0;
      size_t nb = 0, nc = 0;
      for (size_t t = 0; t < trials; t++) {
        vector<double> d1, d2;
        Rng r1(seed + 0xB0A1ull + t);
        synthDifferences(d1, pairs, base, sigmaRel, 0.106, 0.4, 1.0, -(tolerance - tq * sePilot), r1);
        if (decisionUnstable(pairedSignedRank(d1, alpha), tolerance, alpha)) nb++;
        Rng r2(seed + 0xC1EAull + t);
        synthDifferences(d2, pairs, base, sigmaRel, 0.106, 0.4, 1.0, -(tolerance - 4.0 * tq * sePilot), r2);
        if (decisionUnstable(pairedSignedRank(d2, alpha), tolerance, alpha)) nc++;
      }
      stabilityBorder = (double)nb / (double)trials;
      stabilityClear  = (double)nc / (double)trials;
    }

    const bool n9FailsAsExpected = !passes(n9);
    // The four pre-registered classes must PARTITION the eight rows: a report whose own classes do not
    // add up is the quiet version of a wrong verdict.
    const int pfClassTotal = pfBothFailInsideGate + pfFailInsideGate + pfFailAboveGate + pfPassAll;
    const char* pairVerdict =
      (pfBothFailInsideGate > 0) ? "gate-problem" :
      (pfFailInsideGate > 0) ? "25-has-support" :
      (pfPassAll == 8)       ? "25-is-declaration-only" :
                               "no-support-above-gate-only";

    const bool selfOk = passes(n25) && n25CoverageOk && n9FailsAsExpected && envMismatch == 0 && tErr < 1e-4 &&
                        stabilityBorder >= 0.50 && stabilityClear <= 0.05 &&
                        pfClassTotal == 8 &&
                        envPass[0] && envPass[1] && envPass[2] && envPass[3];

    ostringstream os;
    os << fixed << setprecision(4)
       << "seed=" << seed << " trials=" << trials
       << " alpha=" << setprecision(3) << alpha << setprecision(4)
       << " tol=" << tolerance << " sigma_rel=" << sigmaRel
       << " block_ms=" << setprecision(2) << MIN_BLOCK_TIME_MS << setprecision(4)
       << " pairs_floor=" << pairs << " gate=" << setprecision(3) << SPIKE_RATE_GATE << setprecision(4)
       << " | t_inv[" << tt.str() << " max_err=" << scientific << setprecision(2) << tErr << fixed << setprecision(4) << "]"
       << " | n9[cov=" << n9.coverage << " snF=" << n9.sentinelFalse
       << " sp=" << fabs(n9.spikesWith - n9.spikesWithout) << "] must-fail=" << (n9FailsAsExpected ? 1 : 0)
       << " | n16_altseed[cov=" << n16.coverage << " pw=" << n16.powerTol
       << " vs_target=" << setprecision(2) << vsTarget16 << "sigma]" << setprecision(4)
       << " | n16_minus_n25[delta=" << deltaN << " se_diff=" << seDelta
       << " sig=" << setprecision(2) << sigmaDelta << "sigma]" << setprecision(4)
       << " | n25[cov=" << n25.coverage << " pw=" << n25.powerTol << " pn=" << n25.powerNoise
       << " sp=" << fabs(n25.spikesWith - n25.spikesWithout) << " snF=" << n25.sentinelFalse
       << " snP=" << n25.sentinelPower << " snP_half=" << n25.sentinelPowerHalf << " w=" << n25.ciWidth << "]"
       << " pass=" << (passes(n25) ? 1 : 0)
       << " | coverage[nominal=" << (1.0 - alpha) << " effective=" << n25.coverage << "]"
       << " | coverage_target[target=" << COVERAGE_TARGET << " min_ok=" << covMinOk
       << " se_cov=" << se25 << " trials_used=" << trials << (loose ? " LOOSE" : "") << "]"
       << " | reps[formula_0.1pct=" << bud.repsFormula[0] << " formula_1pct=" << bud.repsFormula[1]
       << " floor=" << bud.floorReps << " required=" << bud.repsRequired[0] << "/" << bud.repsRequired[1] << "]"
       << " | stability[k=" << setprecision(1) << STABILITY_BOUNDARY_SE << setprecision(4)
       << " border_unresolved=" << stabilityBorder << " clear_unresolved=" << stabilityClear << "]"
       << " | pairfloor[" << pf.str() << " inside_gate_fail=" << pfFailInsideGate
       << " above_gate_fail=" << pfFailAboveGate << " both_fail_in_gate=" << pfBothFailInsideGate
       << " verdict=" << pairVerdict << "]"
       << " | env[" << env.str() << "] ok_env=" << envOkCount << "/8 expect_mismatch=" << envMismatch
       << " | " << (selfOk ? "ok" : "FAILED");
    if (!selfOk) {
      os << " (n25_pass=" << (passes(n25) ? 1 : 0) << " cov_ok=" << (n25CoverageOk ? 1 : 0)
         << " stab_border=" << stabilityBorder << " stab_clear=" << stabilityClear
         << " n9_fails=" << (n9FailsAsExpected ? 1 : 0)
         << " t_err=" << tErr << " env_mismatch=" << envMismatch << ")";
    }
    return os.str();
  }
}
