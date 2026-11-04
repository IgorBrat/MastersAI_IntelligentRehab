/// Kinematic summaries of one repetition. `features.kinematics`
library;

import 'dart:math' as math;

import '../analysis/live_angles.dart' show V3;
import 'signal.dart';

const double _eps = 1e-12;

/// Tangential speed of a joint path, units per second
List<double> speedProfile(List<V3> path, double fps, {double cutoffHz = 4.0}) {
  var xs = [for (final p in path) p.x];
  var ys = [for (final p in path) p.y];
  var zs = [for (final p in path) p.z];
  if (path.length >= 5) {
    xs = smoothSignal(xs, fps, cutoffHz: cutoffHz);
    ys = smoothSignal(ys, fps, cutoffHz: cutoffHz);
    zs = smoothSignal(zs, fps, cutoffHz: cutoffHz);
  }
  final vx = gradient(xs, 1 / fps);
  final vy = gradient(ys, 1 / fps);
  final vz = gradient(zs, 1 / fps);
  return [
    for (var t = 0; t < path.length; t++)
      math.sqrt(vx[t] * vx[t] + vy[t] * vy[t] + vz[t] * vz[t]),
  ];
}

/// Spectral arc length of a speed profile; NaN when undefined
double sparc(List<double> speed, double fps, {double fcMax = 10.0, double ampThreshold = 0.05}) {
  if (speed.length < 8 || !speed.any((v) => v.abs() > _eps)) return double.nan;
  final target = math.max(speed.length * 4, 64);
  final nFft = math.pow(2, (math.log(target) / math.ln2).ceil()).toInt();
  var spectrum = rfftMagnitude(speed, nFft);
  final peak = maxOf(spectrum);
  if (peak <= _eps) return double.nan;
  spectrum = [for (final s in spectrum) s / peak];
  var freqs = [for (var k = 0; k < spectrum.length; k++) k * fps / nFft];

  final inBand = [for (var k = 0; k < freqs.length; k++) if (freqs[k] <= fcMax) k];
  if (inBand.length < 4) return double.nan;
  freqs = [for (final k in inBand) freqs[k]];
  spectrum = [for (final k in inBand) spectrum[k]];

  var cutoff = spectrum.length - 1;
  for (var k = spectrum.length - 1; k >= 0; k--) {
    if (spectrum[k] >= ampThreshold) {
      cutoff = k;
      break;
    }
    if (k == 0) cutoff = spectrum.length - 1;
  }
  cutoff = math.max(cutoff, 3);
  freqs = freqs.sublist(0, cutoff + 1);
  spectrum = spectrum.sublist(0, cutoff + 1);

  final last = freqs.last;
  var arc = 0.0;
  for (var k = 1; k < freqs.length; k++) {
    final df = freqs[k] / last - freqs[k - 1] / last;
    final dv = spectrum[k] - spectrum[k - 1];
    arc += math.sqrt(df * df + dv * dv);
  }
  return -arc;
}

/// Log dimensionless jerk of a speed profile; NaN when undefined
double ldlj(List<double> speed, double fps) {
  if (speed.length < 9) return double.nan;
  final v = smoothSignal(speed, fps);
  final vPeak = v.map((e) => e.abs()).reduce(math.max);
  final duration = v.length / fps;
  if (vPeak <= _eps || duration <= _eps) return double.nan;
  final jerk = gradient(gradient(v, 1 / fps), 1 / fps);
  // Trapezoid rule over jerk squared
  var integral = 0.0;
  for (var t = 1; t < jerk.length; t++) {
    integral += 0.5 * (jerk[t] * jerk[t] + jerk[t - 1] * jerk[t - 1]) / fps;
  }
  final dlj = (duration * duration * duration / (vPeak * vPeak)) * integral;
  if (dlj <= _eps) return double.nan;
  return -math.log(dlj);
}

/// Velocity peaks, i.e. corrective submovements
int nSubmovements(List<double> speed, double fps, {double prominenceRatio = 0.10}) {
  if (speed.length < 5) return 0;
  final v = smoothSignal(speed, fps);
  final peak = maxOf(v);
  if (peak <= _eps) return 0;
  return findPeaks(
    v,
    prominence: prominenceRatio * peak,
    distance: math.max(1, pyRound(0.15 * fps)).toDouble(),
  ).length;
}

/// Timing decomposition of one repetition. `kinematics.TempoMetrics`
class TempoMetrics {
  const TempoMetrics({
    required this.durationS,
    required this.concentricS,
    required this.eccentricS,
    required this.holdS,
    required this.peakIndex,
    required this.peakValue,
    this.holdRawS = 0.0,
  });

  final double durationS;
  final double concentricS;
  final double eccentricS;

  /// Bias-corrected end-range hold; the value the criteria grade
  final double holdS;
  final int peakIndex;
  final double peakValue;
  final double holdRawS;

  double get concentricEccentricRatio => concentricS / math.max(eccentricS, _eps);
}

/// Splitting a single-repetition excursion into phases. `kinematics.tempo_metrics`
TempoMetrics tempoMetrics(
  List<double> x,
  double fps, {
  double holdBandFrac = 0.08,
  double holdSpeedFrac = 0.30,
}) {
  final n = x.length;
  if (n == 0) {
    return const TempoMetrics(
        durationS: 0, concentricS: 0, eccentricS: 0, holdS: 0, peakIndex: 0, peakValue: double.nan);
  }
  final peakIdx = argmax(x);
  final peakVal = x[peakIdx];
  final baseline = minOf(x);
  final excursion = math.max(peakVal - baseline, _eps);

  final amplitudeOk = [for (final v in x) v >= peakVal - holdBandFrac * excursion];
  List<bool> speedOk;
  if (n >= 3) {
    final speed = [for (final v in gradient(x, 1 / fps)) v.abs()];
    final limit = math.max(holdSpeedFrac * maxOf(speed), _eps);
    speedOk = [for (final s in speed) s <= limit];
  } else {
    speedOk = List<bool>.filled(n, true);
  }
  final stationary = [for (var i = 0; i < n; i++) amplitudeOk[i] && speedOk[i]];

  var lo = peakIdx;
  var hi = peakIdx;
  int holdFrames;
  if (stationary[peakIdx]) {
    while (lo > 0 && stationary[lo - 1]) {
      lo--;
    }
    while (hi < n - 1 && stationary[hi + 1]) {
      hi++;
    }
    holdFrames = hi - lo + 1;
  } else {
    holdFrames = 0;
  }

  final holdRaw = holdFrames / fps;
  final moving = math.max(n / fps - holdRaw, 0.0);
  final hold = math.max(holdRaw - _approachBiasFraction(holdSpeedFrac) * moving, 0.0);
  return TempoMetrics(
    durationS: n / fps,
    concentricS: lo / fps,
    eccentricS: (n - 1 - hi) / fps,
    holdS: hold,
    peakIndex: peakIdx,
    peakValue: peakVal,
    holdRawS: holdRaw,
  );
}

double _approachBiasFraction(double speedFrac) {
  final f = speedFrac.clamp(0.0, 1.0);
  return (1.0 - math.sqrt(1.0 - math.sqrt(f))) / 2.0;
}

/// Start / peak / end of a repetition's primary angle. `kinematics.RomMetrics`
class RomMetrics {
  const RomMetrics({required this.start, required this.peak, required this.end, required this.excursion});

  final double start;
  final double peak;
  final double end;
  final double excursion;

  /// How far short of the starting position the limb was left
  double get returnError => (end - start).abs();
}

RomMetrics romMetrics(List<double> x, {int edgeFrames = 3}) {
  if (x.isEmpty) {
    return const RomMetrics(start: double.nan, peak: double.nan, end: double.nan, excursion: 0);
  }
  final k = math.max(1, math.min(edgeFrames, x.length));
  final start = median(x.sublist(0, k));
  final end = median(x.sublist(x.length - k));
  final peak = maxOf(x);
  return RomMetrics(start: start, peak: peak, end: end, excursion: peak - start);
}
