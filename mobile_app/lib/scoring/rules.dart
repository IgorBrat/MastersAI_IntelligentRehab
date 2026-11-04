/// The rule engine: graded criteria, the quality index, framing gates and
/// repetition segmentation. `rehab_ml.rules` and `rehab_ml.segmentation`
library;

import 'dart:math' as math;

import '../analysis/live_angles.dart' show BodySide;
import 'angles.dart';
import 'exercises.dart';
import 'kinematics.dart';
import 'pose.dart';
import 'signal.dart';
import 'taxonomy.dart';

// ---------------------------------------------------------------- criteria

/// Percentile of the gated frames a deviation is summarised by
const double deviationPercentile = 90.0;

/// Frames at the start of a repetition that define its zero
const int baselineFrames = 5;

/// Low-pass cutoff for the primary clinical channel
const double clinicalCutoffHz = 2.0;

class Finding {
  const Finding({
    required this.error,
    required this.severity,
    required this.measured,
    required this.threshold,
    required this.unit,
  });

  final ErrorType error;
  final Severity severity;
  final double measured;
  final double threshold;
  final String unit;

  bool get triggered => severity != Severity.none;
}

Severity grade(double deviation, Tolerance t) {
  if (deviation <= t.mild) return Severity.none;
  if (deviation <= t.moderate) return Severity.mild;
  if (deviation <= t.severe) return Severity.moderate;
  return Severity.severe;
}

/// `criteria.normalised_penalty`, in [0, 1]
double normalisedPenalty(double deviation, Tolerance t) {
  final span = t.severe - t.mild;
  if (span <= 1e-9) return deviation > t.mild ? 1.0 : 0.0;
  return ((deviation - t.mild) / span).clamp(0.0, 1.0).toDouble();
}

const Set<String> _tlChannels = {
  'shoulder_hike',
  'hand_forward',
  'hand_up',
  'hand_lateral',
  'shoulder_height_working',
  'shoulder_height_other',
};

String channelUnit(String channel) => _tlChannels.contains(channel) ? ' TL' : '°';

List<double> resolveChannel(AngleSeries angles, String channel, ExerciseDefinition exercise) {
  switch (channel) {
    case 'trunk_lean_contralateral':
      return angles.trunkLeanContralateral();
    case 'shoulder_hike':
      return angles.shoulderHike(baselineFrames: baselineFrames);
    case 'plane_deviation':
      final plane = exercise.targetPlaneDeg;
      if (plane == null) return List<double>.filled(angles.nFrames, 0);
      return angles.planeDeviationFrom(plane);
    case 'elbow_flexion_coupling':
      return angles['elbow_flexion'];
    default:
      return angles[channel];
  }
}

List<bool> gateMask(AngleSeries angles, ExerciseDefinition exercise, double gateFrac) {
  final primary = angles[exercise.primaryChannel];
  if (gateFrac <= 0.0) return List<bool>.filled(primary.length, true);
  final k = math.max(1, math.min(baselineFrames, primary.length));
  final baseline = median(primary.sublist(0, k));
  final excursion = (exercise.rom.target - baseline).abs();
  if (excursion < 1e-9) return List<bool>.filled(primary.length, true);
  return [for (final v in primary) (v - baseline).abs() >= gateFrac * excursion];
}

Finding evaluateCompensation(
  AngleSeries angles,
  ExerciseDefinition exercise,
  CompensationLimit limit,
) {
  var series = resolveChannel(angles, limit.channel, exercise);
  if (limit.baselineRelative) {
    final k = math.max(1, math.min(baselineFrames, series.length));
    final zero = median(series.sublist(0, k));
    series = [for (final v in series) v - zero];
  }
  final deviation =
      limit.direction == Direction.positive ? series : [for (final v in series) v.abs()];
  final mask = gateMask(angles, exercise, limit.gatePrimaryFrac);
  final gated = [for (var t = 0; t < deviation.length; t++) if (mask[t]) deviation[t]];
  final measured = math.max(gated.isEmpty ? 0.0 : percentile(gated, deviationPercentile), 0.0);
  return Finding(
    error: limit.error,
    severity: grade(measured, limit.tolerance),
    measured: measured,
    threshold: limit.tolerance.mild,
    unit: channelUnit(limit.channel),
  );
}

// ---------------------------------------------------------------- quality index

class EqiWeights {
  const EqiWeights({
    this.rom = 0.45,
    this.smoothness = 0.20,
    this.tempo = 0.15,
    this.hold = 0.10,
    this.returnToStart = 0.10,
  });

  final double rom;
  final double smoothness;
  final double tempo;
  final double hold;
  final double returnToStart;

  double get total => rom + smoothness + tempo + hold + returnToStart;
}

const Map<ErrorType, double> compensationWeights = {
  ErrorType.trunkForwardFlexion: 1.3,
  ErrorType.trunkLateralLean: 1.2,
  ErrorType.shoulderHiking: 1.0,
  ErrorType.elbowFlexionSynergy: 1.0,
  ErrorType.trunkRotation: 0.8,
  ErrorType.outOfPlaneDrift: 0.7,
};

const double gateFloor = 0.35;

class QualityIndex {
  const QualityIndex({
    required this.value,
    required this.performance,
    required this.gate,
    required this.components,
    required this.penalties,
  });

  final double value;
  final double performance;
  final double gate;
  final Map<String, double> components;
  final Map<String, double> penalties;

  int get grade => eqiToGrade(value);
}

QualityIndex computeEqi({
  required ExerciseDefinition exercise,
  required double peak,
  required double baseline,
  required double sparc,
  required double durationS,
  required double holdS,
  required double returnError,
  required Map<ErrorType, double> compensationPenalties,
  EqiWeights weights = const EqiWeights(),
}) {
  const eps = 1e-9;
  final total = weights.total;

  final targetExcursion = (exercise.rom.target - baseline).abs();
  final romScore = targetExcursion <= eps
      ? 1.0
      : ((peak - baseline).abs() / targetExcursion).clamp(0.0, 1.0).toDouble();

  final severe = exercise.sparcTolerance.severe;
  double smoothnessScore;
  if (!sparc.isFinite) {
    smoothnessScore = 0.5;
  } else if (severe <= eps) {
    smoothnessScore = sparc >= exercise.sparcFloor ? 1.0 : 0.0;
  } else {
    final deficit = math.max(0.0, exercise.sparcFloor - sparc);
    smoothnessScore = (1.0 - deficit / severe).clamp(0.0, 1.0).toDouble();
  }

  final prescribed = exercise.tempo.totalS;
  double tempoScore;
  if (prescribed <= eps) {
    tempoScore = 1.0;
  } else {
    final minimum = prescribed * (1.0 - exercise.tempo.speedTolerance);
    tempoScore = durationS >= minimum
        ? 1.0
        : (durationS / math.max(minimum, eps)).clamp(0.0, 1.0).toDouble();
  }

  final holdScore = exercise.tempo.holdS <= eps
      ? 1.0
      : (holdS / exercise.tempo.holdS).clamp(0.0, 1.0).toDouble();

  final returnSevere = exercise.returnTolerance.severe;
  final returnScore = returnSevere <= eps
      ? 1.0
      : (1.0 - returnError / returnSevere).clamp(0.0, 1.0).toDouble();

  final components = {
    'rom': romScore,
    'smoothness': smoothnessScore,
    'tempo': tempoScore,
    'hold': holdScore,
    'return_to_start': returnScore,
  };
  final performance = romScore * (weights.rom / total) +
      smoothnessScore * (weights.smoothness / total) +
      tempoScore * (weights.tempo / total) +
      holdScore * (weights.hold / total) +
      returnScore * (weights.returnToStart / total);

  final penalties = <String, double>{};
  var weightedSum = 0.0;
  var weightTotal = 0.0;
  compensationPenalties.forEach((error, penalty) {
    final w = compensationWeights[error] ?? 1.0;
    final p = penalty.clamp(0.0, 1.0).toDouble();
    penalties[error.pythonName] = p;
    weightedSum += w * p;
    weightTotal += w;
  });
  final meanPenalty = weightTotal > eps ? weightedSum / weightTotal : 0.0;
  final gate = 1.0 - (1.0 - gateFloor) * meanPenalty;

  return QualityIndex(
    value: (100.0 * performance * gate).clamp(0.0, 100.0).toDouble(),
    performance: performance,
    gate: gate,
    components: components,
    penalties: penalties,
  );
}

// ---------------------------------------------------------------- framing

class FramingVerdict {
  const FramingVerdict({
    required this.viewOk,
    required this.jointsOk,
    required this.detectedView,
    required this.yawDeg,
    this.viewMessageUk = '',
    this.missingJoints = const [],
  });

  final bool viewOk;
  final bool jointsOk;
  final String detectedView;
  final double yawDeg;
  final String viewMessageUk;
  final List<String> missingJoints;

  bool get ok => viewOk && jointsOk;

  /// View first, as `assess_sequence` checks it first
  String get messageUk {
    if (!viewOk) return viewMessageUk;
    if (!jointsOk) {
      return 'Не вдалося надійно розпізнати ключові точки тіла. Переконайтесь, '
          'що вся рука і тулуб у кадрі, а освітлення достатнє.';
    }
    return '';
  }
}

/// Median subject yaw: 0 = facing the camera, 90 = side-on. `pose.view.estimate_yaw`
double medianYaw(PoseSequence seq) => median([
      for (var t = 0; t < seq.nFrames; t++)
        () {
          final line = seq.at(t, Joint.lShoulder) - seq.at(t, Joint.rShoulder);
          return math.atan2(line.z.abs(), math.max(line.x.abs(), 1e-8)) * 180.0 / math.pi;
        }(),
    ]);

FramingVerdict framingVerdict(
  PoseSequence seq,
  ExerciseDefinition exercise,
  BodySide side, {
  double confidenceThreshold = 0.4,
  double frontalMaxYaw = 35.0,
  double sagittalMinYaw = 55.0,
}) {
  final yaw = medianYaw(seq);
  final detected = yaw <= frontalMaxYaw
      ? CameraView.frontal
      : yaw >= sagittalMinYaw
          ? CameraView.sagittal
          : CameraView.any;

  var viewOk = true;
  var message = '';
  final shown = pyRound(yaw);
  switch (exercise.requiredView) {
    case CameraView.any:
      break;
    case CameraView.frontal:
      viewOk = yaw <= frontalMaxYaw;
      if (!viewOk) {
        message = 'Повернітесь обличчям до камери - вправу потрібно записувати спереду '
            '(зараз корпус повернутий приблизно на $shown°).';
      }
    case CameraView.sagittal:
      viewOk = yaw >= sagittalMinYaw;
      if (!viewOk) {
        message = 'Станьте боком до камери - вправу потрібно записувати з боку '
            '(зараз корпус повернутий приблизно на $shown°).';
      }
  }

  final critical = {
    Joint.shoulder(side),
    Joint.elbow(side),
    Joint.wrist(side),
    Joint.lHip,
    Joint.rHip,
  };
  final missing = [
    for (final j in seq.lowConfidenceJoints(threshold: confidenceThreshold))
      if (critical.contains(j)) Joint.names[j],
  ]..sort();

  return FramingVerdict(
    viewOk: viewOk,
    jointsOk: missing.isEmpty,
    detectedView: detected.wire,
    yawDeg: yaw,
    viewMessageUk: message,
    missingJoints: missing,
  );
}

// ---------------------------------------------------------------- segmentation

class Repetition {
  const Repetition({
    required this.index,
    required this.start,
    required this.end,
    required this.peak,
    required this.peakValue,
    required this.baselineValue,
  });

  final int index;
  final int start;
  final int end;
  final int peak;
  final double peakValue;
  final double baselineValue;
}

/// Splitting a continuous angle stream into repetitions. `segment_repetitions`
List<Repetition> segmentRepetitions(
  AngleSeries angles,
  ExerciseDefinition exercise, {
  double minExcursionFrac = 0.35,
  double minPeakDistanceS = 1.0,
  double valleyReturnFrac = 0.5,
  double padS = 0.15,
  double cutoffHz = 3.0,
}) {
  final raw = angles[exercise.primaryChannel];
  final n = raw.length;
  if (n < 5) return const [];
  final signal = smoothSignal(raw, angles.fps, cutoffHz: cutoffHz);

  final baseline = percentile(signal, 10);
  var targetExcursion = (exercise.rom.target - baseline).abs();
  if (targetExcursion < 1e-6) {
    targetExcursion = math.max(percentile(signal, 90) - baseline, 1e-6);
  }
  final minHeight = baseline + minExcursionFrac * targetExcursion;
  final distance = math.max(1, pyRound(minPeakDistanceS * angles.fps));

  var peaks = findPeaks(
    signal,
    height: minHeight,
    distance: distance.toDouble(),
    prominence: 0.25 * minExcursionFrac * targetExcursion,
  );
  if (peaks.isEmpty) return const [];

  final valleyLevel = baseline + valleyReturnFrac * targetExcursion;
  peaks = _mergePeaksWithoutValley(signal, peaks, baseline, valleyReturnFrac);
  final bounds = _valleysBetween(signal, peaks, valleyLevel);

  final pad = pyRound(padS * angles.fps);
  final reps = <Repetition>[];
  for (var i = 0; i < peaks.length; i++) {
    final start = math.max(0, bounds[i] - pad);
    final end = math.min(n, bounds[i + 1] + pad);
    if (end - start < 3) continue;
    reps.add(Repetition(
      index: reps.length,
      start: start,
      end: end,
      peak: peaks[i],
      peakValue: raw[peaks[i]],
      baselineValue: median(raw.sublist(start, math.min(n, start + math.max(1, pad)))),
    ));
  }
  return reps;
}

List<int> _mergePeaksWithoutValley(
  List<double> signal,
  List<int> peaks,
  double baseline,
  double valleyReturnFrac,
) {
  if (peaks.length <= 1) return peaks;
  final kept = [peaks.first];
  for (final peak in peaks.skip(1)) {
    final previous = kept.last;
    final pairPeak = math.min(signal[previous], signal[peak]);
    final threshold = baseline + valleyReturnFrac * (pairPeak - baseline);
    final valley = minOf(signal.sublist(previous, peak + 1));
    if (valley <= threshold) {
      kept.add(peak);
    } else if (signal[peak] > signal[previous]) {
      kept[kept.length - 1] = peak;
    }
  }
  return kept;
}

List<int> _valleysBetween(List<double> signal, List<int> peaks, double valleyLevel) {
  final bounds = <int>[];
  var idx = peaks.first;
  while (idx > 0 && (signal[idx - 1] < signal[idx] || signal[idx] > valleyLevel)) {
    idx--;
  }
  bounds.add(idx);
  for (var k = 0; k < peaks.length - 1; k++) {
    final left = peaks[k];
    final segment = signal.sublist(left, peaks[k + 1] + 1);
    var best = 0;
    for (var i = 1; i < segment.length; i++) {
      if (segment[i] < segment[best]) best = i;
    }
    bounds.add(left + best);
  }
  idx = peaks.last;
  final n = signal.length;
  while (idx < n - 1 && (signal[idx + 1] < signal[idx] || signal[idx] > valleyLevel)) {
    idx++;
  }
  bounds.add(idx + 1);
  return bounds;
}

// ---------------------------------------------------------------- engine

class SmoothnessMetrics {
  const SmoothnessMetrics({required this.sparc, required this.ldlj, required this.nSubmovements});
  final double sparc;
  final double ldlj;
  final int nSubmovements;
}

class RepetitionAssessment {
  RepetitionAssessment({
    required this.repIndex,
    required this.exerciseId,
    required this.side,
    required this.frames,
    required this.findings,
    required this.rom,
    required this.tempo,
    required this.smoothness,
    required this.quality,
  });

  final int repIndex;
  final String exerciseId;
  final BodySide side;
  final (int, int) frames;
  final List<Finding> findings;
  final RomMetrics rom;
  final TempoMetrics tempo;
  final SmoothnessMetrics smoothness;
  final QualityIndex quality;

  List<Finding> get triggered => [for (final f in findings) if (f.triggered) f];
  bool get isCorrect => triggered.isEmpty;

  Severity get worstSeverity => findings.fold(
      Severity.none, (worst, f) => f.severity.index > worst.index ? f.severity : worst);

  Finding? findingFor(ErrorType error) {
    for (final f in findings) {
      if (f.error == error) return f;
    }
    return null;
  }

  /// The service's JSON for one repetition. `RepetitionAssessment.as_dict`
  Map<String, dynamic> asDict() => {
        'rep_index': repIndex,
        'exercise_id': exerciseId,
        'side': side.wire.toUpperCase(),
        'frames': [frames.$1, frames.$2],
        'is_correct': isCorrect,
        'worst_severity': worstSeverity.wire,
        'eqi': pyRoundTo(quality.value, 2),
        'grade': quality.grade,
        'rom': {
          'peak': pyRoundTo(rom.peak, 2),
          'excursion': pyRoundTo(rom.excursion, 2),
          'return_error': pyRoundTo(rom.returnError, 2),
        },
        'tempo': {
          'duration_s': pyRoundTo(tempo.durationS, 2),
          'hold_s': pyRoundTo(tempo.holdS, 2),
        },
        'smoothness': {
          'sparc': smoothness.sparc.isFinite ? pyRoundTo(smoothness.sparc, 3) : null,
          'n_submovements': smoothness.nSubmovements,
        },
        'errors': [
          for (final f in triggered)
            {
              'slug': f.error.slug,
              'severity': f.severity.wire,
              'measured': pyRoundTo(f.measured, 4),
              'threshold': pyRoundTo(f.threshold, 4),
              'unit': f.unit.trim(),
            },
        ],
      };
}

class SessionAssessment {
  SessionAssessment({required this.exerciseId, required this.side});

  final String exerciseId;
  final BodySide side;
  final List<RepetitionAssessment> repetitions = [];
  String? rejectionReasonUk;

  int get nReps => repetitions.length;
  bool get usable => rejectionReasonUk == null && nReps > 0;

  double get meanEqi => mean([for (final r in repetitions) r.quality.value]);
  double get bestEqi => maxOf([for (final r in repetitions) r.quality.value]);
  double get correctRatio => mean([for (final r in repetitions) r.isCorrect ? 1.0 : 0.0]);

  /// Fraction of repetitions in which each watched error triggered
  Map<ErrorType, double> errorFrequency() {
    final counts = <ErrorType, int>{};
    final watched = <ErrorType, int>{};
    for (final rep in repetitions) {
      for (final f in rep.findings) {
        watched[f.error] = (watched[f.error] ?? 0) + 1;
        if (f.triggered) counts[f.error] = (counts[f.error] ?? 0) + 1;
      }
    }
    return {
      for (final e in watched.entries)
        if (e.value > 0) e.key: (counts[e.key] ?? 0) / e.value,
    };
  }

  /// Least-squares slope of EQI across the set, points per repetition
  double fatigueTrend() {
    if (nReps < 3) return 0.0;
    final y = [for (final r in repetitions) r.quality.value];
    final xMean = (y.length - 1) / 2.0;
    final yMean = mean(y);
    var num = 0.0;
    var den = 0.0;
    for (var i = 0; i < y.length; i++) {
      num += (i - xMean) * (y[i] - yMean);
      den += (i - xMean) * (i - xMean);
    }
    return num / den;
  }

  /// The service's JSON for a set. `SessionAssessment.as_dict`
  Map<String, dynamic> asDict() {
    final any = repetitions.isNotEmpty;
    return {
      'exercise_id': exerciseId,
      'side': side.wire.toUpperCase(),
      'usable': usable,
      'rejection_reason_uk': rejectionReasonUk,
      'n_reps': nReps,
      'mean_eqi': any ? pyRoundTo(meanEqi, 2) : null,
      'best_eqi': any ? pyRoundTo(bestEqi, 2) : null,
      'correct_ratio': any ? pyRoundTo(correctRatio, 3) : null,
      'grade': any ? eqiToGrade(meanEqi) : null,
      'fatigue_trend': pyRoundTo(fatigueTrend(), 3),
      'error_frequency': {
        for (final e in errorFrequency().entries) e.key.slug: pyRoundTo(e.value, 3),
      },
      'repetitions': [for (final r in repetitions) r.asDict()],
    };
  }
}

Severity _gradeBands(double deviation, (double, double, double) bands) {
  if (bands.$3 <= 1e-9) return Severity.none;
  return grade(deviation, Tolerance(bands.$1, bands.$2, bands.$3));
}

/// Evaluating repetitions against an exercise's criteria. `rules.engine.RuleEngine`
class RuleEngine {
  RuleEngine({this.excludedErrors = const {}});

  /// Criteria switched off for the sensor; dropped from findings and the EQI
  final Set<ErrorType> excludedErrors;

  EqiWeights get _weights => excludedErrors.contains(ErrorType.jerkyMovement)
      ? const EqiWeights(smoothness: 0.0)
      : const EqiWeights();

  RepetitionAssessment assessRepetition(
    AngleSeries angles,
    ExerciseDefinition exercise, {
    int repIndex = 0,
    (int, int) frames = (0, 0),
    List<double>? handSpeed,
  }) {
    final rawPrimary = angles[exercise.primaryChannel];
    final primary = rawPrimary.length >= 5
        ? smoothSignal(rawPrimary, angles.fps, cutoffHz: clinicalCutoffHz)
        : List<double>.from(rawPrimary);
    final rom = romMetrics(primary);
    final tempo = tempoMetrics(primary, angles.fps);

    final speed = handSpeed ??
        [for (final v in gradient(smoothSignal(primary, angles.fps), 1 / angles.fps)) v.abs()];
    // Smoothness over the concentric phase only
    final concentricEnd = math.min(math.max(tempo.peakIndex + 1, 2), speed.length);
    final concentric = speed.sublist(0, concentricEnd);
    final smooth = SmoothnessMetrics(
      sparc: sparc(concentric, angles.fps),
      ldlj: ldlj(concentric, angles.fps),
      nSubmovements: nSubmovements(concentric, angles.fps),
    );

    final findings = <Finding>[];
    final penalties = <ErrorType, double>{};
    final unit = channelUnit(exercise.primaryChannel);

    final targetExcursion = (exercise.rom.target - rom.start).abs();
    final romDeficit = math.max(0.0, targetExcursion - (rom.peak - rom.start).abs());
    final romBands = (0.10 * targetExcursion, 0.25 * targetExcursion, 0.45 * targetExcursion);
    findings.add(Finding(
      error: ErrorType.insufficientRom,
      severity: _gradeBands(romDeficit, romBands),
      measured: romDeficit,
      threshold: romBands.$1,
      unit: unit,
    ));

    findings.add(Finding(
      error: ErrorType.incompleteReturn,
      severity: grade(rom.returnError, exercise.returnTolerance),
      measured: rom.returnError,
      threshold: exercise.returnTolerance.mild,
      unit: unit,
    ));

    if (exercise.tempo.holdS > 0) {
      final shortfall =
          math.max(0.0, (exercise.tempo.holdS - tempo.holdS) / exercise.tempo.holdS);
      findings.add(Finding(
        error: ErrorType.insufficientHold,
        severity: grade(shortfall, exercise.holdTolerance),
        measured: 100.0 * shortfall,
        threshold: 100.0 * exercise.holdTolerance.mild,
        unit: '%',
      ));
    }

    final speedDeficit = math.max(0.0, exercise.tempo.minAcceptableS - tempo.durationS);
    final total = exercise.tempo.totalS;
    final speedBands = (0.10 * total, 0.25 * total, 0.45 * total);
    findings.add(Finding(
      error: ErrorType.excessiveSpeed,
      severity: _gradeBands(speedDeficit, speedBands),
      measured: speedDeficit,
      threshold: speedBands.$1,
      unit: 's',
    ));

    final sparcDeficit =
        smooth.sparc.isFinite ? math.max(0.0, exercise.sparcFloor - smooth.sparc) : 0.0;
    if (!excludedErrors.contains(ErrorType.jerkyMovement)) {
      findings.add(Finding(
        error: ErrorType.jerkyMovement,
        severity: grade(sparcDeficit, exercise.sparcTolerance),
        measured: sparcDeficit,
        threshold: exercise.sparcTolerance.mild,
        unit: ' SPARC',
      ));
    }

    for (final limit in exercise.compensations) {
      if (excludedErrors.contains(limit.error)) continue;
      final finding = evaluateCompensation(angles, exercise, limit);
      findings.add(finding);
      penalties[limit.error] = normalisedPenalty(finding.measured, limit.tolerance);
    }

    final quality = computeEqi(
      exercise: exercise,
      peak: rom.peak,
      baseline: rom.start,
      sparc: smooth.sparc,
      durationS: tempo.durationS,
      holdS: tempo.holdS,
      returnError: rom.returnError,
      compensationPenalties: penalties,
      weights: _weights,
    );

    return RepetitionAssessment(
      repIndex: repIndex,
      exerciseId: exercise.id,
      side: angles.side,
      frames: frames,
      findings: findings,
      rom: rom,
      tempo: tempo,
      smoothness: smooth,
      quality: quality,
    );
  }

  /// Segmenting and scoring a whole set. `RuleEngine.assess_sequence`
  /// Also returns angles and bounds so the model reuses the same windows
  ({SessionAssessment session, AngleSeries? angles, List<Repetition> reps}) assessSequence(
    PoseSequence seq,
    ExerciseDefinition exercise,
    BodySide side, {
    bool checkView = true,
  }) {
    final session = SessionAssessment(exerciseId: exercise.id, side: side);
    final framing = framingVerdict(seq, exercise, side);
    if (checkView && !framing.viewOk) {
      session.rejectionReasonUk = framing.viewMessageUk;
      return (session: session, angles: null, reps: const <Repetition>[]);
    }
    if (!framing.jointsOk) {
      session.rejectionReasonUk = framing.messageUk;
      return (session: session, angles: null, reps: const <Repetition>[]);
    }

    final angles = computeAngles(seq, side);
    final reps = segmentRepetitions(angles, exercise);
    if (reps.isEmpty) {
      session.rejectionReasonUk = 'Не виявлено жодного повторення. Виконайте вправу з більшою '
          'амплітудою або перевірте, що камера бачить вас повністю.';
      return (session: session, angles: angles, reps: reps);
    }

    final hand = seq.trajectory(Joint.hand(side));
    for (final rep in reps) {
      session.repetitions.add(assessRepetition(
        angles.slice(rep.start, rep.end),
        exercise,
        repIndex: rep.index,
        frames: (rep.start, rep.end),
        handSpeed: speedProfile(hand.sublist(rep.start, rep.end), seq.fps),
      ));
    }
    return (session: session, angles: angles, reps: reps);
  }
}
