#include "../neuralnet/tunestats.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

    // Acklam's inverse normal CDF: the allocator and the intervals need z_{1-alpha/2} for a DECLARED
    // alpha, and a hard-coded 1.96 would quietly contradict that declaration.
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

    vector<double> walshAverages(const vector<double>& d) {
      vector<double> w;
      w.reserve(d.size() * (d.size() + 1) / 2);
      for (size_t i = 0; i < d.size(); i++)
        for (size_t j = i; j < d.size(); j++)
          w.push_back(0.5 * (d[i] + d[j]));
      sort(w.begin(), w.end());
      return w;
    }

    double walshMedian(const vector<double>& d) {
      if (d.empty()) return 0.0;
      vector<double> w = walshAverages(d);
      return median(w);
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

  const char* intervalName(Interval iv) {
    switch (iv) {
      case CI_MAD: return "mad";
      case CI_WILCOXON: return "wilcoxon";
      case CI_BOOTSTRAP: return "bootstrap";
    }
    return "?";
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

  // ---------------------------------------------------------------- the paired test
  PairedResult pairedSignedRank(const vector<double>& dIn, double alpha, unsigned long long seed) {
    PairedResult r;
    r.pairs = 0; r.zerosDropped = 0; r.hl = 0.0;
    r.ciLowMad = r.ciHighMad = 0.0; r.madOk = false;
    r.ciLowWil = r.ciHighWil = 0.0; r.wilOk = false;
    r.ciLowBoot = r.ciHighBoot = 0.0; r.bootOk = false;
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
    const vector<double> w = walshAverages(d);
    r.hl = median(w);

    if (n >= 4) {
      // (a) CI_MAD: the median's standard error is ~1.2533 * sigma / sqrt(n) for a normal bulk, and
      //     the MAD is the scale that survives the spikes.
      const double sig = madScale(d);
      const double half = normInv(1.0 - alpha / 2.0) * 1.2533 * sig / sqrt((double)n);
      r.ciLowMad = r.hl - half;
      r.ciHighMad = r.hl + half;
      r.madOk = true;

      // (b) CI_WILCOXON: the distribution-free signed-rank interval -- order statistics of the Walsh
      //     averages, trimmed at each end by the alpha/2 quantile of W+.  With ties the critical
      //     value comes from the normal approximation with the ties correction (the standard
      //     practical form).  This interval assumes NOTHING about the noise shape.
      const long N = (long)w.size();
      const double z = normInv(1.0 - alpha / 2.0);
      long k = (long)floor(meanW - z * sqrt(varW > 0.0 ? varW : 0.0));
      if (k < 0) k = 0;
      if (2 * k < N) {
        r.ciLowWil = w[(size_t)k];
        r.ciHighWil = w[(size_t)(N - 1 - k)];
        r.wilOk = true;
      }

      // (c) CI_BOOTSTRAP: kept only so its measured failure stays on the record.
      {
        const int B = 400;
        vector<double> hlB;
        hlB.reserve(B);
        Rng rng(seed);
        vector<double> sample(n, 0.0);
        for (int b = 0; b < B; b++) {
          for (size_t kk = 0; kk < n; kk++) sample[kk] = d[(size_t)(rng.next() % n)];
          hlB.push_back(walshMedian(sample));
        }
        sort(hlB.begin(), hlB.end());
        const size_t lo = (size_t)floor((alpha / 2.0) * B);
        const size_t hi = (size_t)ceil((1.0 - alpha / 2.0) * B) - 1;
        r.ciLowBoot = hlB[min(lo, hlB.size() - 1)];
        r.ciHighBoot = hlB[min(hi, hlB.size() - 1)];
        r.bootOk = true;
      }
    }
    return r;
  }

  bool intervalCovers(const PairedResult& r, Interval iv, double value) {
    switch (iv) {
      case CI_MAD: return r.madOk && r.ciLowMad <= value && r.ciHighMad >= value;
      case CI_WILCOXON: return r.wilOk && r.ciLowWil <= value && r.ciHighWil >= value;
      case CI_BOOTSTRAP: return r.bootOk && r.ciLowBoot <= value && r.ciHighBoot >= value;
    }
    return false;
  }

  bool notWorseThan(const PairedResult& r, double tolerance, Interval iv) {
    switch (iv) {
      case CI_MAD:
        if (!r.madOk) return false;               // fail closed: no interval, no claim
        return r.ciLowMad >= -tolerance;
      case CI_WILCOXON:
        if (!r.wilOk) return false;
        return r.ciLowWil >= -tolerance;
      case CI_BOOTSTRAP:
        if (!r.bootOk) return false;
        return r.ciLowBoot >= -tolerance;
    }
    return false;
  }

  bool equivalentWithin(const PairedResult& r, double tolerance, Interval iv) {
    switch (iv) {
      case CI_MAD:
        if (!r.madOk) return false;
        return r.ciLowMad >= -tolerance && r.ciHighMad <= tolerance;
      case CI_WILCOXON:
        if (!r.wilOk) return false;
        return r.ciLowWil >= -tolerance && r.ciHighWil <= tolerance;
      case CI_BOOTSTRAP:
        if (!r.bootOk) return false;
        return r.ciLowBoot >= -tolerance && r.ciHighBoot <= tolerance;
    }
    return false;
  }

  // ---------------------------------------------------------------- the budget
  Budget allocateReps(const vector<double>& sigmaRel, double tolerance, double alpha,
                      long floorReps, long maxReps) {
    Budget b;
    b.floorReps = floorReps;
    b.totalReps = 0;
    b.anyClampedAtMax = false;
    const double z = normInv(1.0 - alpha / 2.0);
    for (size_t s = 0; s < sigmaRel.size(); s++) {
      long reps;
      if (tolerance <= 0.0 || sigmaRel[s] <= 0.0) { reps = floorReps; }
      else {
        const double need = (z * sigmaRel[s] / tolerance);
        reps = (long)ceil(need * need);
      }
      if (reps < floorReps) reps = floorReps;
      if (reps > maxReps) { reps = maxReps; b.anyClampedAtMax = true; }
      b.repsPerShape.push_back(reps);
      b.totalReps += reps;
    }
    return b;
  }

  // ---------------------------------------------------------------- the self-test
  // Ground truth is INJECTED, so every number below is a MEASUREMENT of the algorithm:
  //   coverage    : how often the nominal 95% interval really contains the true effect
  //   power_at_tol: how often a candidate 1.01x the tolerance worse is refused
  //   power@noise : how often a candidate only noise-level worse is refused (should be LOW)
  //   spikes+-    : whether the verdict agrees with the spike-free verdict
  //   sentinel    : how often an honest A/A comparison is declared broken (should be LOW)
  // and the sweep runs every (interval, pairs) combination, so the choice is made by measurement --
  // the first run's bootstrap collapse is exactly why this loop exists.
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
      double coverage, powerTol, powerNoise, spikesWith, spikesWithout, sentinel, ciWidth;
    };

    template <typename Pred>
    double runRate(size_t trials, size_t pairs, double base, double sigmaRel, double spikeProb,
                   double effect, double alpha, unsigned long long seed, Pred pred, double* widthOut,
                   Interval widthIv) {
      Rng rng(seed);
      vector<double> d;
      size_t hit = 0;
      vector<double> widths;
      if (widthOut != nullptr) widths.reserve(trials);
      for (size_t t = 0; t < trials; t++) {
        synthDifferences(d, pairs, base, sigmaRel, spikeProb, 0.4, 1.0, effect, rng);
        const PairedResult r = pairedSignedRank(d, alpha, seed + 7919ull * (unsigned long long)(t + 1));
        if (pred(r)) hit++;
        if (widthOut != nullptr) {
          // the width of the interval UNDER TEST (reporting another one is how a record starts lying)
          if (widthIv == CI_WILCOXON && r.wilOk) widths.push_back(r.ciHighWil - r.ciLowWil);
          else if (widthIv == CI_BOOTSTRAP && r.bootOk) widths.push_back(r.ciHighBoot - r.ciLowBoot);
          else if (widthIv == CI_MAD && r.madOk) widths.push_back(r.ciHighMad - r.ciLowMad);
        }
      }
      if (widthOut != nullptr) *widthOut = median(widths);
      return (double)hit / (double)trials;
    }

    Rates measure(Interval iv, size_t trials, size_t pairs, double alpha, double tolerance,
                  double sigmaRel, double spikeProb, double base, unsigned long long seed) {
      Rates r;
      const double effectCover = 0.002;
      r.coverage = runRate(trials, pairs, base, sigmaRel, spikeProb, effectCover, alpha, seed,
                           [&](const PairedResult& p) { return intervalCovers(p, iv, effectCover); },
                           &r.ciWidth, iv);
      r.powerTol = runRate(trials, pairs, base, sigmaRel, spikeProb, -1.01 * tolerance, alpha, seed + 11,
                           [&](const PairedResult& p) { return !notWorseThan(p, tolerance, iv); }, nullptr, iv);
      r.powerNoise = runRate(trials, pairs, base, sigmaRel, spikeProb, -sigmaRel, alpha, seed + 23,
                             [&](const PairedResult& p) { return !notWorseThan(p, tolerance, iv); }, nullptr, iv);
      r.spikesWith = runRate(trials, pairs, base, sigmaRel, spikeProb, -0.5 * tolerance, alpha, seed + 37,
                             [&](const PairedResult& p) { return notWorseThan(p, tolerance, iv); }, nullptr, iv);
      r.spikesWithout = runRate(trials, pairs, base, sigmaRel, 0.0, -0.5 * tolerance, alpha, seed + 37,
                                [&](const PairedResult& p) { return notWorseThan(p, tolerance, iv); }, nullptr, iv);
      r.sentinel = runRate(trials, pairs, base, sigmaRel, spikeProb, 0.0, alpha, seed + 53,
                           [&](const PairedResult& p) { return !equivalentWithin(p, tolerance, iv); }, nullptr, iv);
      return r;
    }

    bool passes(const Rates& r) {
      return r.coverage >= 0.90 && r.coverage <= 0.99 && r.powerTol >= 0.80 &&
             r.powerNoise <= 0.20 && fabs(r.spikesWith - r.spikesWithout) <= 0.05 &&
             r.sentinel <= 0.05;
    }
  }

  string selfTestSummary() {
    const double alpha = 0.05;          // DECLARED (the reference makes alpha an explicit parameter too)
    const double tolerance = 0.01;      // 1% : the product tolerance
    const double sigmaRel = 0.001;      // 0.1% : the BLOCK-relative paired scale measured on the 5080
    const double spikeProb = 0.106;     // measured: logs-t3/calib-d-paired.txt
    const double base = 1.0;
    const size_t trials = 300;
    const unsigned long long seed = 0x5EEDC0DEULL;

    const Interval ivs[2] = { CI_MAD, CI_WILCOXON };
    const size_t pairChoices[3] = { 9, 16, 25 };
    Interval pickIv = CI_MAD;
    long pickPairs = -1;
    bool picked = false;

    ostringstream os;
    os << fixed << setprecision(4)
       << "seed=" << seed << " trials=" << trials
       << " alpha=" << setprecision(3) << alpha << setprecision(4)
       << " tol=" << tolerance << " sigma_rel=" << sigmaRel << " spike_p=" << spikeProb << " | sweep";
    for (size_t a = 0; a < 2; a++) {
      for (size_t b = 0; b < 3; b++) {
        const Rates r = measure(ivs[a], trials, pairChoices[b], alpha, tolerance, sigmaRel, spikeProb,
                                base, seed);
        const bool ok = passes(r);
        os << " " << intervalName(ivs[a]) << "/n=" << pairChoices[b]
           << "[cov=" << r.coverage << " pw=" << r.powerTol << " pn=" << r.powerNoise
           << " sp=" << fabs(r.spikesWith - r.spikesWithout)
           << " sn=" << r.sentinel << " w=" << r.ciWidth << (ok ? " OK" : "") << "]";
        if (ok && !picked) { pickIv = ivs[a]; pickPairs = (long)pairChoices[b]; picked = true; }
      }
    }
    const Rates boot = measure(CI_BOOTSTRAP, trials, 9, alpha, tolerance, sigmaRel, spikeProb, base, seed);

    vector<double> sig(1, sigmaRel);
    const Budget bud = allocateReps(sig, tolerance, alpha, 9, 512);

    os << " | picked=" << intervalName(pickIv) << "/n=" << pickPairs
       << " bootstrap_record[sn=" << boot.sentinel << " pn=" << boot.powerNoise << "]"
       << " reps_for_tol=" << (bud.repsPerShape.empty() ? 0 : bud.repsPerShape[0])
       << " | " << (picked ? "ok" : "FAILED (no interval x pairs combination met the acceptance bar)");
    return os.str();
  }
}
