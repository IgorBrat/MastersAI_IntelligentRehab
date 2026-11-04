/// On-device scoring with the same JSON as the rehab-ml service endpoints.
/// [ScoringCore] is the synchronous pipeline; [RehabEngine] runs it on an isolate.
library;

import 'dart:isolate';

import 'package:flutter/services.dart' show AssetBundle, rootBundle;

import '../analysis/live_angles.dart' show BodySide;
import '../analysis/rep_detector.dart' show effectiveFps;
import '../api/client.dart'
    show Assessment, CoachAdvice, ExerciseInfo, Readiness;
import '../pose/pose_stream.dart';
import 'adaptation.dart';
import 'angles.dart';
import 'exercises.dart';
import 'model.dart';
import 'pose.dart';
import 'rules.dart';
import 'taxonomy.dart';

const String thresholdsAsset = 'assets/thresholds/thresholds_irds_patients.json';
const String weightsAsset = 'assets/model/hybrid.bin';
const String manifestAsset = 'assets/model/hybrid.json';

BodySide parseSide(String raw) {
  switch (raw.trim().toLowerCase()) {
    case 'l':
    case 'left':
    case 'ліва':
      return BodySide.left;
    case 'r':
    case 'right':
    case 'права':
      return BodySide.right;
  }
  throw ArgumentError("side must be 'left' or 'right', got $raw");
}

/// The scoring configuration and the synchronous pipeline
class ScoringCore {
  ScoringCore({
    this.thresholds,
    HybridWeights? weights,
    this.selfPaced = true,
    Set<ErrorType> excluded = const {},
  })  : runner = weights == null ? null : ModelRunner(weights),
        engine = RuleEngine(excludedErrors: excluded);

  final ThresholdFile? thresholds;
  final ModelRunner? runner;
  final bool selfPaced;
  final RuleEngine engine;

  /// An exercise with the process-wide configuration applied. `service.resolve`
  ExerciseDefinition resolve(String exerciseId) {
    var definition = getExercise(exerciseId);
    if (selfPaced) definition = asSelfPaced(definition);
    final calibration = thresholds?.calibrations[exerciseId];
    if (calibration != null) definition = applyCalibration(definition, calibration);
    return definition;
  }

  List<String> get warnings => [
        if (thresholds == null)
          'No threshold file: the protocol bands are in force, and three of them '
              'are known to be several times too tight for real movement.',
        if (runner != null && runner!.selfPaced != selfPaced)
          'The model was fitted with self_paced=${runner!.selfPaced}; scoring runs '
              'with self_paced=$selfPaced. The hold_ratio feature differs between them.',
      ];

  /// `GET /health`
  Map<String, dynamic> health() => {
        'status': 'ok',
        'runtime': 'on-device',
        'self_paced': selfPaced,
        'thresholds': thresholds?.source,
        'thresholds_provenance': thresholds?.provenance,
        'model': runner?.info(),
        'excluded_criteria': [for (final e in engine.excludedErrors) e.slug]..sort(),
        'warnings': warnings,
      };

  /// `GET /exercises`
  List<Map<String, dynamic>> exercises() => [
        for (final base in protocol)
          () {
            final e = resolve(base.id);
            return {
              'id': e.id,
              'name_uk': e.nameUk,
              'name_en': e.nameEn,
              'primary_channel': e.primaryChannel,
              'required_view': e.requiredView.wire,
              'target_deg': e.rom.target,
              'floor_deg': e.rom.floor,
              'ceiling_deg': e.rom.ceiling,
              'concentric_s': e.tempo.concentricS,
              'hold_s': e.tempo.holdS,
              'eccentric_s': e.tempo.eccentricS,
              'default_reps': e.defaultReps,
              'instructions_uk': e.instructionsUk,
              'watched_errors': [
                for (final limit in e.compensations)
                  {
                    'slug': limit.error.slug,
                    'cue_uk': limit.error.patientUk,
                    'safety_relevant': limit.error.safetyRelevant,
                  },
              ],
            };
          }(),
      ];

  /// A posted window as a preprocessed canonical sequence. `service._to_sequence`
  PoseSequence toSequence(Map<String, dynamic> wire) {
    final landmarks = wire['landmarks'] as List;
    final visibility = wire['visibility'] as List?;
    List<double> flat(Object frame) => [
          for (final p in frame as List)
            for (final v in p as List) (v as num).toDouble(),
        ];
    final raw = sequenceFromLandmarks(
      [for (final f in landmarks) flat(f as Object)],
      visibility == null
          ? null
          : [for (final f in visibility) [for (final v in f as List) (v as num).toDouble()]],
      (wire['fps'] as num).toDouble(),
    );
    return preprocess(raw);
  }

  /// `POST /check/readiness`
  Map<String, dynamic> readiness(Map<String, dynamic> wire) {
    final definition = resolve(wire['exercise_id'] as String);
    final side = parseSide(wire['side'] as String);
    final verdict = framingVerdict(toSequence(wire), definition, side);
    return {
      'ready': verdict.ok,
      'view_ok': verdict.viewOk,
      'joints_ok': verdict.jointsOk,
      'required_view': definition.requiredView.wire,
      'detected_view': verdict.detectedView,
      'yaw_deg': double.parse(verdict.yawDeg.toStringAsFixed(1)),
      'message_uk': verdict.messageUk.isEmpty ? 'Готово - можна починати' : verdict.messageUk,
      'missing_joints': verdict.missingJoints,
    };
  }

  Map<String, dynamic> _skipped(ExerciseDefinition definition) => {
        'skipped': 'the model was not fitted on ${definition.id}; '
            'it was fitted on ${runner!.fittedExercises.join(', ')}',
      };

  /// `POST /assess/session`
  Map<String, dynamic> assessSession(Map<String, dynamic> wire, {bool checkView = true}) {
    final definition = resolve(wire['exercise_id'] as String);
    final side = parseSide(wire['side'] as String);
    final sequence = toSequence(wire);
    final result = engine.assessSequence(sequence, definition, side, checkView: checkView);
    final payload = result.session.asDict();

    final runner = this.runner;
    final angles = result.angles;
    if (runner == null || angles == null || result.session.repetitions.isEmpty) return payload;
    if (!runner.appliesTo(definition.id)) {
      payload['model'] = _skipped(definition);
      return payload;
    }

    final repPayloads = (payload['repetitions'] as List).cast<Map<String, dynamic>>();
    final predictions = <Map<String, dynamic>>[];
    for (var i = 0; i < result.session.repetitions.length; i++) {
      final rep = result.session.repetitions[i];
      final prediction = runner.predict(
        angles.slice(rep.frames.$1, rep.frames.$2),
        definition,
        ruleEqi: rep.quality.value,
      );
      repPayloads[i]['model'] = prediction;
      predictions.add(prediction);
    }
    double meanOf(String key) =>
        predictions.map((p) => p[key] as double).reduce((a, b) => a + b) / predictions.length;
    payload['model'] = {
      'artifact': runner.info()['path'],
      'exercise_id': runner.exerciseId,
      'mean_eqi': double.parse(meanOf('eqi').toStringAsFixed(2)),
      'mean_correct_prob': double.parse(meanOf('correct_prob').toStringAsFixed(3)),
      'mean_eqi_correction': double.parse(meanOf('eqi_correction').toStringAsFixed(2)),
      'validated_on_phone': false,
    };
    return payload;
  }

  /// `POST /assess/repetition`, one already-segmented window
  Map<String, dynamic> assessRepetition(Map<String, dynamic> wire) {
    final definition = resolve(wire['exercise_id'] as String);
    final side = parseSide(wire['side'] as String);
    final sequence = toSequence(wire);
    final angles = computeAngles(sequence, side);
    final assessment =
        engine.assessRepetition(angles, definition, frames: (0, sequence.nFrames));
    final payload = assessment.asDict();
    final runner = this.runner;
    if (runner != null) {
      payload['model'] = runner.appliesTo(definition.id)
          ? runner.predict(angles, definition, ruleEqi: assessment.quality.value)
          : _skipped(definition);
    }
    return payload;
  }

  /// `POST /coach` decision, without its phrasing
  (Decision, ExerciseDefinition, CoachSession?) decideFor(Map<String, dynamic> body) {
    final exerciseId = body['exercise_id'] as String;
    final definition = resolve(exerciseId);
    var current = Prescription.defaultFor(exerciseId, parseSide(body['side'] as String).wire);
    final romTarget = (body['rom_target'] as num?)?.toDouble();
    final reps = (body['reps'] as num?)?.toInt();
    final sets = (body['sets'] as num?)?.toInt();
    if (romTarget != null || reps != null || sets != null) {
      current = current.copyWith(romTarget: romTarget, reps: reps, sets: sets);
      current = current.copyWith(romTarget: definition.rom.withTarget(current.romTarget).target);
    }
    final history = [
      for (final item in (body['history'] as List? ?? const []))
        if ('${(item as Map)['exercise_id'] ?? ''}'.isNotEmpty)
          CoachSession.fromJson(item.cast<String, dynamic>()),
    ];
    final decision = decide(
      history,
      current,
      unlocked: {for (final u in (body['unlocked'] as List? ?? const [])) '$u'},
    );
    CoachSession? last;
    for (final s in history.reversed) {
      if (s.exerciseId == exerciseId) {
        last = s;
        break;
      }
    }
    return (decision, definition, last);
  }
}

/// [ScoringCore] behind the interface the screens already used for the service
class RehabEngine {
  RehabEngine(this.core);

  final ScoringCore core;

  /// Loading bundled thresholds and weights; a missing model leaves rules only
  static Future<RehabEngine> load({AssetBundle? bundle}) async {
    final assets = bundle ?? rootBundle;
    final thresholds = ThresholdFile.parse(
      await assets.loadString(thresholdsAsset),
      source: thresholdsAsset,
    );
    HybridWeights? weights;
    String? modelError;
    try {
      weights = HybridWeights.fromBytes(
        await assets.load(weightsAsset),
        await assets.loadString(manifestAsset),
      );
    } catch (e) {
      modelError = '$e';
    }
    final engine = RehabEngine(ScoringCore(
      thresholds: thresholds,
      weights: weights,
      excluded: phoneExcludedCriteria,
    ));
    engine._modelError = modelError;
    return engine;
  }

  /// jerky_movement is off on the phone: its SPARC band is Kinect-fitted.
  /// Same as `serve --exclude-criteria`
  static const Set<ErrorType> phoneExcludedCriteria = {ErrorType.jerkyMovement};

  String? _modelError;

  Map<String, dynamic> health() => {
        ...core.health(),
        if (_modelError != null) 'model_error': _modelError,
      };

  List<ExerciseInfo> exercises() => [for (final e in core.exercises()) ExerciseInfo.fromJson(e)];

  /// Wire body for a window, stored with a recording for re-scoring.
  /// fps is measured from timestamps, since CameraX drops frames
  Future<Map<String, Object>> recordingPayload({
    required String exerciseId,
    required BodySide side,
    required List<PoseFrame> frames,
  }) async =>
      wire(exerciseId, side, frames);

  static Map<String, Object> wire(String exerciseId, BodySide side, List<PoseFrame> frames) => {
        'exercise_id': exerciseId,
        'side': side.wire,
        'fps': effectiveFps([for (final f in frames) f.timestampMs]),
        'landmarks': [for (final f in frames) f.toNestedWorld()],
        'visibility': [for (final f in frames) f.toVisibilityList()],
      };

  Future<T> _run<T>(T Function(ScoringCore core) job) {
    final core = this.core;
    return Isolate.run(() => job(core));
  }

  Future<Readiness> checkReadiness({
    required String exerciseId,
    required BodySide side,
    required List<PoseFrame> frames,
  }) async {
    if (frames.isEmpty) throw ScoringException('no frames to check');
    final body = wire(exerciseId, side, frames);
    return Readiness.fromJson(await _guard(() => _run((c) => c.readiness(body))));
  }

  Future<Assessment> assessRepetition({
    required String exerciseId,
    required BodySide side,
    required List<PoseFrame> frames,
  }) async {
    if (frames.isEmpty) throw ScoringException('no frames to score');
    final body = wire(exerciseId, side, frames);
    return Assessment(await _guard(() => _run((c) => c.assessRepetition(body))));
  }

  Future<Assessment> assessSession({
    required String exerciseId,
    required BodySide side,
    required List<PoseFrame> frames,
  }) async {
    if (frames.isEmpty) throw ScoringException('no frames to score');
    final body = wire(exerciseId, side, frames);
    return Assessment(await _guard(() => _run((c) => c.assessSession(body))));
  }

  /// Re-scoring a stored recording payload exactly as it was first scored
  Future<Assessment> rescore(Map<String, dynamic> payload) async =>
      Assessment(await _guard(() => _run((c) => c.assessSession(payload))));

  /// Next dose, phrased by Gemini when a key is built in, else by the template
  Future<CoachAdvice> coach({
    required String exerciseId,
    required BodySide side,
    required List<Map<String, dynamic>> history,
    double? romTarget,
    int? reps,
    int? sets,
    List<String> unlocked = const [],
  }) async {
    final (decision, definition, last) = core.decideFor({
      'exercise_id': exerciseId,
      'side': side.wire,
      'history': history,
      'rom_target': romTarget,
      'reps': reps,
      'sets': sets,
      'unlocked': unlocked,
    });
    final (text, source, error) = await phrase(decision, definition, last);
    return CoachAdvice.fromJson({
      ...decision.asDict(),
      'text_uk': text,
      'text_source': source,
      'text_error': error,
    });
  }

  /// Uniformly thinning an overlong window, keeping the late (fatigued) reps
  static List<PoseFrame> fitToCap(List<PoseFrame> frames, {int cap = 5400}) {
    if (frames.length <= cap) return frames;
    final stride = (frames.length / cap).ceil();
    return [for (var i = 0; i < frames.length; i += stride) frames[i]];
  }

  Future<Map<String, dynamic>> _guard(Future<Map<String, dynamic>> Function() job) async {
    try {
      return await job();
    } catch (e) {
      throw ScoringException('$e');
    }
  }
}

class ScoringException implements Exception {
  ScoringException(this.message);
  final String message;
  @override
  String toString() => message;
}
