/// numpy / scipy routines used by the pipeline, ported algorithm-for-algorithm
/// so repetition bounds and SPARC match Python.
library;

import 'dart:math' as math;
import 'dart:typed_data';

class _C {
  const _C(this.re, this.im);
  final double re;
  final double im;

  _C operator +(_C o) => _C(re + o.re, im + o.im);
  _C operator -(_C o) => _C(re - o.re, im - o.im);
  _C operator *(_C o) => _C(re * o.re - im * o.im, re * o.im + im * o.re);
  _C operator /(_C o) {
    final d = o.re * o.re + o.im * o.im;
    return _C((re * o.re + im * o.im) / d, (im * o.re - re * o.im) / d);
  }

  _C scale(double k) => _C(re * k, im * k);
}

/// Coefficients of prod(x - r) for complex roots, highest power first. `numpy.poly`
List<_C> _poly(List<_C> roots) {
  var out = <_C>[const _C(1, 0)];
  for (final r in roots) {
    final next = List<_C>.filled(out.length + 1, const _C(0, 0));
    for (var i = 0; i < out.length; i++) {
      next[i] = next[i] + out[i];
      next[i + 1] = next[i + 1] - out[i] * r;
    }
    out = next;
  }
  return out;
}

/// Digital Butterworth low-pass. `scipy.signal.butter(order, wn, btype="low")`
/// [wn] is the cutoff as a fraction of Nyquist
({List<double> b, List<double> a}) butterLowpass(int order, double wn) {
  const fs = 2.0;
  final warped = 2 * fs * math.tan(math.pi * wn / fs);
  // buttap: p = -exp(1j * pi * m / (2N)), m = -N+1, -N+3, ..., N-1
  final poles = <_C>[
    for (var m = -order + 1; m < order; m += 2)
      _C(-math.cos(math.pi * m / (2 * order)), -math.sin(math.pi * m / (2 * order))),
  ];
  // lp2lp_zpk: no zeros, so the gain picks up wo^N
  final analog = [for (final p in poles) p.scale(warped)];
  var k = math.pow(warped, order).toDouble();
  // bilinear_zpk
  const fs2 = _C(2 * fs, 0);
  final digital = [for (final p in analog) (fs2 + p) / (fs2 - p)];
  var denominator = const _C(1, 0);
  for (final p in analog) {
    denominator = denominator * (fs2 - p);
  }
  k = k * (const _C(1, 0) / denominator).re;
  final zeros = List<_C>.filled(order, const _C(-1, 0));
  final b = [for (final c in _poly(zeros)) c.re * k];
  final a = [for (final c in _poly(digital)) c.re];
  return (b: b, a: a);
}

/// Direct-form II transposed IIR filter with initial state. `scipy.signal.lfilter`
List<double> lfilter(List<double> b, List<double> a, List<double> x, List<double> zi) {
  final n = a.length;
  final a0 = a[0];
  final bn = [for (final v in b) v / a0];
  final an = [for (final v in a) v / a0];
  final z = List<double>.from(zi);
  final y = List<double>.filled(x.length, 0);
  for (var t = 0; t < x.length; t++) {
    final xt = x[t];
    final yt = bn[0] * xt + z[0];
    for (var i = 0; i < n - 2; i++) {
      z[i] = bn[i + 1] * xt - an[i + 1] * yt + z[i + 1];
    }
    z[n - 2] = bn[n - 1] * xt - an[n - 1] * yt;
    y[t] = yt;
  }
  return y;
}

/// Steady-state initial conditions for a step input. `scipy.signal.lfilter_zi`
List<double> lfilterZi(List<double> b, List<double> a) {
  final a0 = a[0];
  final bn = [for (final v in b) v / a0];
  final an = [for (final v in a) v / a0];
  final n = an.length - 1;
  // (I - companion(a).T) zi = b[1:] - a[1:] * b[0]
  final m = List.generate(n, (_) => List<double>.filled(n + 1, 0));
  for (var i = 0; i < n; i++) {
    for (var j = 0; j < n; j++) {
      // companion: row 0 = -a[1:], subdiagonal ones; transposed here
      double companionT;
      if (j == 0) {
        companionT = -an[i + 1];
      } else {
        companionT = (i == j - 1) ? 1.0 : 0.0;
      }
      m[i][j] = (i == j ? 1.0 : 0.0) - companionT;
    }
    m[i][n] = bn[i + 1] - an[i + 1] * bn[0];
  }
  return _solve(m);
}

/// Gaussian elimination with partial pivoting on an augmented matrix
List<double> _solve(List<List<double>> m) {
  final n = m.length;
  for (var col = 0; col < n; col++) {
    var pivot = col;
    for (var r = col + 1; r < n; r++) {
      if (m[r][col].abs() > m[pivot][col].abs()) pivot = r;
    }
    final tmp = m[col];
    m[col] = m[pivot];
    m[pivot] = tmp;
    for (var r = col + 1; r < n; r++) {
      final f = m[r][col] / m[col][col];
      for (var c = col; c <= n; c++) {
        m[r][c] -= f * m[col][c];
      }
    }
  }
  final x = List<double>.filled(n, 0);
  for (var r = n - 1; r >= 0; r--) {
    var s = m[r][n];
    for (var c = r + 1; c < n; c++) {
      s -= m[r][c] * x[c];
    }
    x[r] = s / m[r][r];
  }
  return x;
}

/// Zero-phase forward-backward filter, odd padding. `scipy.signal.filtfilt`
List<double> filtfilt(List<double> b, List<double> a, List<double> x, {required int padlen}) {
  final n = x.length;
  final ext = <double>[
    for (var i = padlen; i >= 1; i--) 2 * x[0] - x[i],
    ...x,
    for (var i = n - 2; i >= n - 1 - padlen; i--) 2 * x[n - 1] - x[i],
  ];
  final zi = lfilterZi(b, a);
  var y = lfilter(b, a, ext, [for (final z in zi) z * ext.first]);
  final reversed = y.reversed.toList();
  y = lfilter(b, a, reversed, [for (final z in zi) z * reversed.first]);
  final forward = y.reversed.toList();
  return forward.sublist(padlen, forward.length - padlen);
}

/// Least-squares polynomial of [order] through (positions, values), evaluated at [at]
List<double> _polyfitEval(List<double> values, int order, List<double> at) {
  final n = values.length;
  final centre = (n - 1) / 2.0;
  final size = order + 1;
  final m = List.generate(size, (_) => List<double>.filled(size + 1, 0));
  for (var i = 0; i < n; i++) {
    final t = i - centre;
    final powers = List<double>.generate(2 * size - 1, (p) => math.pow(t, p).toDouble());
    for (var r = 0; r < size; r++) {
      for (var c = 0; c < size; c++) {
        m[r][c] += powers[r + c];
      }
      m[r][size] += powers[r] * values[i];
    }
  }
  final coef = _solve(m);
  return [
    for (final position in at)
      () {
        final t = position - centre;
        var s = 0.0;
        for (var p = size - 1; p >= 0; p--) {
          s = s * t + coef[p];
        }
        return s;
      }(),
  ];
}

/// Savitzky-Golay smoothing, `mode="interp"`. `scipy.signal.savgol_filter`
List<double> savgolFilter(List<double> x, int window, int polyorder) {
  final n = x.length;
  final half = window ~/ 2;
  final y = List<double>.filled(n, 0);
  for (var i = half; i < n - half; i++) {
    y[i] = _polyfitEval(x.sublist(i - half, i + half + 1), polyorder, [half.toDouble()])[0];
  }
  // The edges are fitted by one polynomial over the first / last window
  final left = _polyfitEval(x.sublist(0, window), polyorder,
      [for (var i = 0; i < half; i++) i.toDouble()]);
  final right = _polyfitEval(x.sublist(n - window), polyorder,
      [for (var i = window - half; i < window; i++) i.toDouble()]);
  for (var i = 0; i < half; i++) {
    y[i] = left[i];
    y[n - half + i] = right[i];
  }
  return y;
}

/// Python's `round(x)`: half to even
int pyRound(double x) {
  final floor = x.floorToDouble();
  final diff = x - floor;
  if (diff > 0.5) return floor.toInt() + 1;
  if (diff < 0.5) return floor.toInt();
  final f = floor.toInt();
  return f.isEven ? f : f + 1;
}

/// Python's `round(x, digits)` for the JSON the service returns
double pyRoundTo(double x, int digits) {
  if (!x.isFinite) return x;
  return double.parse(x.toStringAsFixed(digits));
}

/// Zero-phase low-pass for limb motion. `features.kinematics.smooth_signal`
List<double> smoothSignal(List<double> x, double fps, {double cutoffHz = 6.0}) {
  final n = x.length;
  final normalised = cutoffHz / (0.5 * fps);
  if (n >= 18 && normalised > 0.0 && normalised < 0.95) {
    final ba = butterLowpass(4, normalised);
    final padlen = math.min<int>(3 * math.max<int>(ba.a.length, ba.b.length), n - 1);
    return filtfilt(ba.b, ba.a, x, padlen: padlen);
  }
  var window = math.max(5, pyRound(fps / math.max(cutoffHz, 0.5)) | 1);
  if (window >= n) window = n.isOdd ? n : n - 1;
  if (window < 5) return List<double>.from(x);
  return savgolFilter(x, window, 2);
}

/// [smoothSignal] applied to every column of a frame-major matrix
List<List<double>> smoothColumns(List<List<double>> columns, double fps, {double cutoffHz = 6.0}) =>
    [for (final c in columns) smoothSignal(c, fps, cutoffHz: cutoffHz)];

/// Central differences with one-sided ends. `numpy.gradient(x, dx)`
List<double> gradient(List<double> x, double dx) {
  final n = x.length;
  if (n < 2) return List<double>.filled(n, 0);
  final out = List<double>.filled(n, 0);
  out[0] = (x[1] - x[0]) / dx;
  out[n - 1] = (x[n - 1] - x[n - 2]) / dx;
  for (var i = 1; i < n - 1; i++) {
    out[i] = (x[i + 1] - x[i - 1]) / (2 * dx);
  }
  return out;
}

/// Linear-interpolation percentile, `numpy.percentile(x, q)`
double percentile(List<double> x, double q) {
  final sorted = List<double>.from(x)..sort();
  final n = sorted.length;
  if (n == 1) return sorted[0];
  final rank = q / 100.0 * (n - 1);
  final lo = rank.floor();
  final hi = math.min(lo + 1, n - 1);
  final frac = rank - lo;
  final a = sorted[lo];
  final b = sorted[hi];
  // numpy's lerp form, which is exact at both ends
  return frac >= 0.5 ? b - (b - a) * (1 - frac) : a + (b - a) * frac;
}

double median(List<double> x) => percentile(x, 50);

double mean(List<double> x) {
  var s = 0.0;
  for (final v in x) {
    s += v;
  }
  return s / x.length;
}

double maxOf(List<double> x) => x.reduce(math.max);
double minOf(List<double> x) => x.reduce(math.min);

int argmax(List<double> x) {
  var best = 0;
  for (var i = 1; i < x.length; i++) {
    if (x[i] > x[best]) best = i;
  }
  return best;
}

/// `numpy.interp`: clamped at both ends
double interp(double xq, List<double> xp, List<double> fp) {
  if (xq <= xp.first) return fp.first;
  if (xq >= xp.last) return fp.last;
  var lo = 0;
  var hi = xp.length - 1;
  while (hi - lo > 1) {
    final mid = (lo + hi) >> 1;
    if (xp[mid] <= xq) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  final span = xp[hi] - xp[lo];
  if (span == 0) return fp[hi];
  return fp[lo] + (fp[hi] - fp[lo]) * (xq - xp[lo]) / span;
}

/// `numpy.linspace(start, stop, n)`
List<double> linspace(double start, double stop, int n) {
  if (n == 1) return [start];
  final step = (stop - start) / (n - 1);
  return [for (var i = 0; i < n; i++) i == n - 1 ? stop : start + i * step];
}

/// `scipy.signal.find_peaks`: maxima, then height, distance, prominence filters.
/// Flat peaks resolve to their middle sample
List<int> findPeaks(
  List<double> x, {
  double? height,
  double? distance,
  double? prominence,
}) {
  final n = x.length;
  var peaks = <int>[];
  var i = 1;
  final iMax = n - 1;
  while (i < iMax) {
    if (x[i - 1] < x[i]) {
      var ahead = i + 1;
      while (ahead < iMax && x[ahead] == x[i]) {
        ahead++;
      }
      if (x[ahead] < x[i]) {
        peaks.add((i + ahead - 1) ~/ 2);
        i = ahead;
      }
    }
    i++;
  }

  if (height != null) {
    peaks = [for (final p in peaks) if (x[p] >= height) p];
  }

  if (distance != null && peaks.length > 1) {
    final d = distance.ceil();
    final keep = List<bool>.filled(peaks.length, true);
    final order = List<int>.generate(peaks.length, (k) => k)
      ..sort((a, b) {
        final c = x[peaks[a]].compareTo(x[peaks[b]]);
        return c != 0 ? c : a.compareTo(b);
      });
    for (var r = order.length - 1; r >= 0; r--) {
      final j = order[r];
      if (!keep[j]) continue;
      var k = j - 1;
      while (k >= 0 && peaks[j] - peaks[k] < d) {
        keep[k] = false;
        k--;
      }
      k = j + 1;
      while (k < peaks.length && peaks[k] - peaks[j] < d) {
        keep[k] = false;
        k++;
      }
    }
    peaks = [for (var k = 0; k < peaks.length; k++) if (keep[k]) peaks[k]];
  }

  if (prominence != null) {
    peaks = [for (final p in peaks) if (peakProminence(x, p) >= prominence) p];
  }
  return peaks;
}

/// `scipy.signal.peak_prominences` for one peak, no window
double peakProminence(List<double> x, int peak) {
  final top = x[peak];
  var leftMin = top;
  for (var i = peak; i >= 0 && x[i] <= top; i--) {
    if (x[i] < leftMin) leftMin = x[i];
  }
  var rightMin = top;
  for (var i = peak; i < x.length && x[i] <= top; i++) {
    if (x[i] < rightMin) rightMin = x[i];
  }
  return top - math.max(leftMin, rightMin);
}

/// Magnitude of the real FFT, zero-padded to [nFft] (a power of two). `numpy.fft.rfft`
List<double> rfftMagnitude(List<double> x, int nFft) {
  final re = List<double>.filled(nFft, 0);
  final im = List<double>.filled(nFft, 0);
  for (var i = 0; i < x.length && i < nFft; i++) {
    re[i] = x[i];
  }
  // Bit-reversal permutation
  for (var i = 1, j = 0; i < nFft; i++) {
    var bit = nFft >> 1;
    for (; (j & bit) != 0; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      final tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      final ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (var len = 2; len <= nFft; len <<= 1) {
    final angle = -2 * math.pi / len;
    final wr = math.cos(angle);
    final wi = math.sin(angle);
    for (var s = 0; s < nFft; s += len) {
      var cr = 1.0;
      var ci = 0.0;
      for (var k = 0; k < len ~/ 2; k++) {
        final ar = re[s + k], ai = im[s + k];
        final br = re[s + k + len ~/ 2], bi = im[s + k + len ~/ 2];
        final tr = br * cr - bi * ci;
        final ti = br * ci + bi * cr;
        re[s + k] = ar + tr;
        im[s + k] = ai + ti;
        re[s + k + len ~/ 2] = ar - tr;
        im[s + k + len ~/ 2] = ai - ti;
        final nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
  return [
    for (var k = 0; k <= nFft ~/ 2; k++) math.sqrt(re[k] * re[k] + im[k] * im[k]),
  ];
}

/// Rounding to float32 and back, at the points where numpy stores float32
double f32(double v) {
  _f32[0] = v;
  return _f32[0];
}

final Float32List _f32 = Float32List(1);
