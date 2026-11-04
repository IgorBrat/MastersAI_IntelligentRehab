/// The six-exercise clinical protocol and threshold calibration.
/// `rehab_ml.exercises` and `rules.calibration`; values from `stroke_upper_limb.py`
library;

import 'dart:convert';
import 'dart:math' as math;

import 'taxonomy.dart';

enum CameraView { frontal, sagittal, any }

extension CameraViewName on CameraView {
  /// `CameraView.name` on the Python side
  String get wire => name.toUpperCase();
}

/// Which sign of a deviation counts as an error
enum Direction { absolute, positive }

/// A graded threshold. `exercises.base.Tolerance`
class Tolerance {
  const Tolerance(this.mild, this.moderate, this.severe);

  final double mild;
  final double moderate;
  final double severe;

  Tolerance scaled(double k) => Tolerance(mild * k, moderate * k, severe * k);
}

class RomTarget {
  const RomTarget({
    required this.target,
    required this.floor,
    required this.ceiling,
    required this.normative,
  });

  final double target;
  final double floor;
  final double ceiling;
  final double normative;

  RomTarget withTarget(double value) => RomTarget(
        target: math.min(math.max(value, floor), ceiling),
        floor: floor,
        ceiling: ceiling,
        normative: normative,
      );
}

class TempoPrescription {
  const TempoPrescription({
    required this.concentricS,
    required this.holdS,
    required this.eccentricS,
    this.speedTolerance = 0.4,
  });

  final double concentricS;
  final double holdS;
  final double eccentricS;
  final double speedTolerance;

  double get totalS => concentricS + holdS + eccentricS;
  double get minAcceptableS => totalS * (1.0 - speedTolerance);
}

class CompensationLimit {
  const CompensationLimit({
    required this.error,
    required this.channel,
    required this.tolerance,
    this.direction = Direction.absolute,
    this.baselineRelative = false,
    this.gatePrimaryFrac = 0.0,
  });

  final ErrorType error;
  final String channel;
  final Tolerance tolerance;
  final Direction direction;
  final bool baselineRelative;
  final double gatePrimaryFrac;

  CompensationLimit withTolerance(Tolerance t) => CompensationLimit(
        error: error,
        channel: channel,
        tolerance: t,
        direction: direction,
        baselineRelative: baselineRelative,
        gatePrimaryFrac: gatePrimaryFrac,
      );
}

class ExerciseDefinition {
  const ExerciseDefinition({
    required this.id,
    required this.nameUk,
    required this.nameEn,
    required this.primaryChannel,
    required this.rom,
    required this.tempo,
    required this.requiredView,
    required this.compensations,
    this.targetPlaneDeg,
    this.requiredElbowDeg,
    this.returnTolerance = const Tolerance(10.0, 20.0, 35.0),
    this.holdTolerance = const Tolerance(0.2, 0.4, 0.7),
    this.sparcFloor = -1.50,
    this.sparcTolerance = const Tolerance(0.05, 0.20, 0.45),
    this.defaultSets = 2,
    this.defaultReps = 10,
    this.instructionsUk = '',
  });

  final String id;
  final String nameUk;
  final String nameEn;
  final String primaryChannel;
  final RomTarget rom;
  final TempoPrescription tempo;
  final CameraView requiredView;
  final List<CompensationLimit> compensations;
  final double? targetPlaneDeg;
  final double? requiredElbowDeg;
  final Tolerance returnTolerance;
  final Tolerance holdTolerance;
  final double sparcFloor;
  final Tolerance sparcTolerance;
  final int defaultSets;
  final int defaultReps;
  final String instructionsUk;

  ExerciseDefinition copyWith({
    RomTarget? rom,
    TempoPrescription? tempo,
    List<CompensationLimit>? compensations,
    Tolerance? returnTolerance,
    Tolerance? holdTolerance,
    Tolerance? sparcTolerance,
  }) =>
      ExerciseDefinition(
        id: id,
        nameUk: nameUk,
        nameEn: nameEn,
        primaryChannel: primaryChannel,
        rom: rom ?? this.rom,
        tempo: tempo ?? this.tempo,
        requiredView: requiredView,
        compensations: compensations ?? this.compensations,
        targetPlaneDeg: targetPlaneDeg,
        requiredElbowDeg: requiredElbowDeg,
        returnTolerance: returnTolerance ?? this.returnTolerance,
        holdTolerance: holdTolerance ?? this.holdTolerance,
        sparcFloor: sparcFloor,
        sparcTolerance: sparcTolerance ?? this.sparcTolerance,
        defaultSets: defaultSets,
        defaultReps: defaultReps,
        instructionsUk: instructionsUk,
      );
}

const _trunkLean = CompensationLimit(
  error: ErrorType.trunkLateralLean,
  channel: 'trunk_lean_contralateral',
  tolerance: Tolerance(5.0, 10.0, 18.0),
  direction: Direction.positive,
  baselineRelative: true,
  gatePrimaryFrac: 0.25,
);

const _trunkFlexion = CompensationLimit(
  error: ErrorType.trunkForwardFlexion,
  channel: 'trunk_forward_flexion',
  tolerance: Tolerance(8.0, 15.0, 25.0),
  baselineRelative: true,
  gatePrimaryFrac: 0.25,
);

const _trunkRotation = CompensationLimit(
  error: ErrorType.trunkRotation,
  channel: 'trunk_rotation',
  tolerance: Tolerance(8.0, 15.0, 25.0),
  baselineRelative: true,
  gatePrimaryFrac: 0.25,
);

const _shoulderHike = CompensationLimit(
  error: ErrorType.shoulderHiking,
  channel: 'shoulder_hike',
  tolerance: Tolerance(0.04, 0.08, 0.14),
  direction: Direction.positive,
  baselineRelative: true,
  gatePrimaryFrac: 0.33,
);

const _elbowSynergy = CompensationLimit(
  error: ErrorType.elbowFlexionSynergy,
  channel: 'elbow_flexion_coupling',
  tolerance: Tolerance(15.0, 30.0, 50.0),
  direction: Direction.positive,
  baselineRelative: true,
  gatePrimaryFrac: 0.33,
);

const _planeDrift = CompensationLimit(
  error: ErrorType.outOfPlaneDrift,
  channel: 'plane_deviation',
  tolerance: Tolerance(15.0, 25.0, 40.0),
  gatePrimaryFrac: 0.28,
);

const List<ExerciseDefinition> protocol = [
  ExerciseDefinition(
    id: 'shoulder_abduction_seated',
    nameUk: 'Відведення руки в сторону (сидячи)',
    nameEn: 'Seated shoulder abduction',
    primaryChannel: 'shoulder_abduction',
    rom: RomTarget(target: 90.0, floor: 45.0, ceiling: 150.0, normative: 180.0),
    tempo: TempoPrescription(concentricS: 3.0, holdS: 2.0, eccentricS: 3.0),
    requiredView: CameraView.frontal,
    targetPlaneDeg: 90.0,
    requiredElbowDeg: 0.0,
    compensations: [_trunkLean, _shoulderHike, _elbowSynergy, _planeDrift, _trunkRotation],
    instructionsUk: 'Сядьте прямо, спина притиснута до спинки стільця. Повільно підніміть '
        'пряму руку в сторону до рівня плеча, затримайтесь на 2 секунди і так само '
        'повільно опустіть. Плече не піднімайте до вуха, тулуб тримайте рівно.',
  ),
  ExerciseDefinition(
    id: 'shoulder_flexion_seated',
    nameUk: 'Підйом руки вперед (сидячи)',
    nameEn: 'Seated shoulder flexion',
    primaryChannel: 'shoulder_flexion',
    rom: RomTarget(target: 90.0, floor: 45.0, ceiling: 160.0, normative: 180.0),
    tempo: TempoPrescription(concentricS: 3.0, holdS: 2.0, eccentricS: 3.0),
    requiredView: CameraView.sagittal,
    requiredElbowDeg: 0.0,
    compensations: [_trunkFlexion, _shoulderHike, _elbowSynergy],
    instructionsUk: 'Сядьте боком до камери, спина пряма. Повільно підніміть пряму руку '
        'вперед і вгору до рівня плеча, затримайтесь на 2 секунди, повільно '
        'опустіть. Не відхиляйтеся назад і не нахиляйтеся вперед.',
  ),
  ExerciseDefinition(
    id: 'elbow_flexion_extension',
    nameUk: 'Згинання та розгинання в лікті',
    nameEn: 'Elbow flexion / extension',
    primaryChannel: 'elbow_flexion',
    rom: RomTarget(target: 110.0, floor: 60.0, ceiling: 145.0, normative: 150.0),
    tempo: TempoPrescription(concentricS: 2.0, holdS: 1.0, eccentricS: 3.0),
    requiredView: CameraView.sagittal,
    compensations: [_shoulderHike, _trunkFlexion],
    instructionsUk: 'Притисніть лікоть до тулуба. Повільно зігніть руку в лікті, підводячи '
        'кисть до плеча, потім повністю випряміть. Плече залишається нерухомим.',
  ),
  ExerciseDefinition(
    id: 'shoulder_external_rotation',
    nameUk: 'Зовнішня ротація плеча (лікоть 90°)',
    nameEn: 'Shoulder external rotation at 90 degrees elbow flexion',
    primaryChannel: 'shoulder_rotation',
    rom: RomTarget(target: 45.0, floor: 15.0, ceiling: 80.0, normative: 90.0),
    tempo: TempoPrescription(concentricS: 3.0, holdS: 2.0, eccentricS: 3.0),
    requiredView: CameraView.frontal,
    requiredElbowDeg: 90.0,
    compensations: [_trunkRotation, _shoulderHike, _trunkLean],
    instructionsUk: 'Притисніть лікоть до тулуба і зігніть його під прямим кутом. Повільно '
        'відведіть передпліччя назовні, не відриваючи лікоть від тулуба. '
        'Затримайтесь на 2 секунди і повернітесь.',
  ),
  ExerciseDefinition(
    id: 'forward_reach_to_target',
    nameUk: 'Дотягування вперед до цілі',
    nameEn: 'Seated forward reach to target',
    primaryChannel: 'hand_forward',
    rom: RomTarget(target: 0.55, floor: 0.25, ceiling: 0.85, normative: 0.90),
    tempo: TempoPrescription(concentricS: 2.0, holdS: 1.0, eccentricS: 2.0),
    requiredView: CameraView.sagittal,
    compensations: [_trunkFlexion, _trunkRotation, _shoulderHike],
    returnTolerance: Tolerance(0.08, 0.15, 0.25),
    instructionsUk: 'Сядьте боком до камери. Тягніться рукою вперед, намагаючись дістати '
        'якнайдалі, але НЕ нахиляючи тулуб. Спина залишається притиснутою до '
        'спинки стільця.',
  ),
  ExerciseDefinition(
    id: 'hand_to_mouth',
    nameUk: 'Піднесення руки до рота',
    nameEn: 'Hand to mouth',
    primaryChannel: 'hand_up',
    rom: RomTarget(target: 0.45, floor: 0.20, ceiling: 0.60, normative: 0.65),
    tempo: TempoPrescription(concentricS: 2.0, holdS: 1.0, eccentricS: 2.0),
    requiredView: CameraView.frontal,
    compensations: [_trunkFlexion, _trunkLean, _shoulderHike],
    returnTolerance: Tolerance(0.08, 0.15, 0.25),
    instructionsUk: 'Повільно піднесіть кисть до рота, ніби збираєтесь випити води. '
        'Затримайтесь на секунду і повільно опустіть руку. Голову не нахиляйте до руки.',
  ),
];

/// Easiest to hardest. `stroke_upper_limb.PROGRESSION_ORDER`
const List<String> progressionOrder = [
  'elbow_flexion_extension',
  'shoulder_abduction_seated',
  'shoulder_flexion_seated',
  'shoulder_external_rotation',
  'hand_to_mouth',
  'forward_reach_to_target',
];

ExerciseDefinition getExercise(String id) {
  for (final e in protocol) {
    if (e.id == id) return e;
  }
  throw ArgumentError('unknown exercise $id');
}

/// Dropping the prescribed tempo and hold. `registry.as_self_paced`
ExerciseDefinition asSelfPaced(ExerciseDefinition e) => e.copyWith(
      tempo: TempoPrescription(
        concentricS: e.tempo.concentricS,
        holdS: 0.0,
        eccentricS: e.tempo.eccentricS,
        speedTolerance: 1.0,
      ),
    );

/// Criteria whose thresholds are fitted. `calibration.CALIBRATABLE`
const Set<String> calibratable = {
  'trunk_lateral_lean',
  'trunk_forward_flexion',
  'trunk_rotation',
  'shoulder_hiking',
  'elbow_flexion_synergy',
  'out_of_plane_drift',
  'incomplete_return',
  'jerky_movement',
};

/// One exercise's fitted thresholds, keyed by error slug
class Calibration {
  const Calibration(this.tolerances);
  final Map<String, Tolerance> tolerances;
}

/// A `rehab-ml/thresholds@1` file: the fitted tolerances plus their provenance
class ThresholdFile {
  ThresholdFile({required this.calibrations, required this.provenance, required this.source});

  final Map<String, Calibration> calibrations;
  final Map<String, dynamic> provenance;

  /// Asset path, reported where the service reported its `--thresholds`
  final String source;

  factory ThresholdFile.parse(String text, {required String source}) {
    final payload = jsonDecode(text) as Map<String, dynamic>;
    if (payload['format'] != 'rehab-ml/thresholds@1') {
      throw FormatException('$source is not a rehab-ml threshold file');
    }
    final exercises = (payload['exercises'] as Map).cast<String, dynamic>();
    return ThresholdFile(
      source: source,
      provenance: (payload['provenance'] as Map? ?? {}).cast<String, dynamic>(),
      calibrations: {
        for (final e in exercises.entries)
          e.key: Calibration({
            for (final t in ((e.value as Map)['tolerances'] as Map).entries)
              t.key as String: () {
                final bands = (t.value as List).cast<num>();
                return Tolerance(bands[0].toDouble(), bands[1].toDouble(), bands[2].toDouble());
              }(),
          }),
      },
    );
  }
}

/// A copy of [exercise] carrying the fitted thresholds. `calibration.apply_to_exercise`
ExerciseDefinition applyCalibration(ExerciseDefinition exercise, Calibration calibration) {
  final t = calibration.tolerances;
  return exercise.copyWith(
    compensations: [
      for (final limit in exercise.compensations)
        t.containsKey(limit.error.slug) ? limit.withTolerance(t[limit.error.slug]!) : limit,
    ],
    returnTolerance: t[ErrorType.incompleteReturn.slug],
    sparcTolerance: t[ErrorType.jerkyMovement.slug],
  );
}

/// One patient's current parameters for one exercise. `registry.Prescription`
class Prescription {
  const Prescription({
    required this.exerciseId,
    required this.sideWire,
    required this.romTarget,
    required this.sets,
    required this.reps,
    this.toleranceScale = 1.0,
  });

  final String exerciseId;

  /// 'left' or 'right'
  final String sideWire;
  final double romTarget;
  final int sets;
  final int reps;
  final double toleranceScale;

  factory Prescription.defaultFor(String exerciseId, String sideWire) {
    final e = getExercise(exerciseId);
    return Prescription(
      exerciseId: exerciseId,
      sideWire: sideWire,
      romTarget: e.rom.target,
      sets: e.defaultSets,
      reps: e.defaultReps,
    );
  }

  Prescription copyWith({double? romTarget, int? sets, int? reps}) => Prescription(
        exerciseId: exerciseId,
        sideWire: sideWire,
        romTarget: romTarget ?? this.romTarget,
        sets: sets ?? this.sets,
        reps: reps ?? this.reps,
        toleranceScale: toleranceScale,
      );
}
