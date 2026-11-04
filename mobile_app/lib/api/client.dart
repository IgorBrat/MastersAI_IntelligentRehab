import 'dart:convert';

import 'package:flutter/foundation.dart' show compute;
import 'package:http/http.dart' as http;

import '../analysis/live_angles.dart';
import '../analysis/rep_detector.dart' show effectiveFps;
import '../pose/pose_stream.dart';

/// Protocol data for an exercise. Mirrors `api.schemas.ExerciseInfo`
class ExerciseInfo {
  ExerciseInfo({
    required this.id,
    required this.nameUk,
    required this.requiredView,
    required this.primaryChannel,
    required this.targetDeg,
    required this.ceilingDeg,
    required this.holdSeconds,
    this.concentricSeconds = 0.0,
    this.eccentricSeconds = 0.0,
    required this.defaultReps,
    required this.instructionsUk,
    required this.watchedErrors,
  });

  final String id;
  final String nameUk;
  final String requiredView;

  /// `AngleSeries` channel the repetition excursion is measured on
  final String primaryChannel;

  final double targetDeg;

  /// Highest target adaptive progression may set
  final double ceilingDeg;

  /// Prescribed end-range hold, seconds; zero means self-paced
  final double holdSeconds;

  /// Prescribed lifting and lowering time, seconds
  final double concentricSeconds;
  final double eccentricSeconds;

  final int defaultReps;
  final String instructionsUk;

  /// slug -> Ukrainian patient-facing cue
  final Map<String, String> watchedErrors;

  factory ExerciseInfo.fromJson(Map<String, dynamic> json) => ExerciseInfo(
        id: json['id'] as String,
        nameUk: json['name_uk'] as String,
        requiredView: json['required_view'] as String,
        primaryChannel: json['primary_channel'] as String? ?? 'shoulder_elevation',
        targetDeg: (json['target_deg'] as num).toDouble(),
        // Missing ceiling falls back to 1.5x the target
        ceilingDeg: (json['ceiling_deg'] as num?)?.toDouble() ??
            (json['target_deg'] as num).toDouble() * 1.5,
        holdSeconds: (json['hold_s'] as num?)?.toDouble() ?? 0.0,
        concentricSeconds: (json['concentric_s'] as num?)?.toDouble() ?? 0.0,
        eccentricSeconds: (json['eccentric_s'] as num?)?.toDouble() ?? 0.0,
        defaultReps: (json['default_reps'] as num).toInt(),
        instructionsUk: json['instructions_uk'] as String? ?? '',
        watchedErrors: {
          for (final e in (json['watched_errors'] as List? ?? []))
            (e as Map)['slug'] as String: e['cue_uk'] as String? ?? '',
        },
      );
}

/// A scored repetition or session, kept as raw JSON
class Assessment {
  Assessment(this.json);
  final Map<String, dynamic> json;

  double? get eqi => (json['eqi'] as num?)?.toDouble();
  double? get meanEqi => (json['mean_eqi'] as num?)?.toDouble();
  int get nReps => (json['n_reps'] as num?)?.toInt() ?? 0;
  bool get usable => json['usable'] as bool? ?? true;
  String? get rejectionUk => json['rejection_reason_uk'] as String?;
  int? get grade => (json['grade'] as num?)?.toInt();

  /// Triggered errors on a repetition: slug -> severity name
  Map<String, String> get errors => {
        for (final e in (json['errors'] as List? ?? []))
          (e as Map)['slug'] as String: e['severity'] as String,
      };

  /// Session-level error frequency: slug -> fraction of repetitions
  Map<String, double> get errorFrequency => {
        for (final entry in (json['error_frequency'] as Map? ?? {}).entries)
          entry.key as String: (entry.value as num).toDouble(),
      };

  List<Assessment> get repetitions => [
        for (final r in (json['repetitions'] as List? ?? []))
          Assessment((r as Map).cast<String, dynamic>()),
      ];
}

/// Whether the camera framing is good enough to start. Mirrors `api.schemas.ReadinessInfo`
class Readiness {
  Readiness({
    required this.ready,
    required this.viewOk,
    required this.jointsOk,
    required this.requiredView,
    required this.detectedView,
    required this.yawDeg,
    required this.messageUk,
    required this.missingJoints,
  });

  final bool ready;
  final bool viewOk;
  final bool jointsOk;
  final String requiredView;
  final String detectedView;

  /// 0 = facing the camera, 90 = fully side-on
  final double yawDeg;

  final String messageUk;
  final List<String> missingJoints;

  factory Readiness.fromJson(Map<String, dynamic> json) => Readiness(
        ready: json['ready'] as bool? ?? false,
        viewOk: json['view_ok'] as bool? ?? false,
        jointsOk: json['joints_ok'] as bool? ?? false,
        requiredView: json['required_view'] as String? ?? '',
        detectedView: json['detected_view'] as String? ?? '',
        yawDeg: (json['yaw_deg'] as num?)?.toDouble() ?? 0,
        messageUk: json['message_uk'] as String? ?? '',
        missingJoints: [
          for (final j in (json['missing_joints'] as List? ?? [])) j as String,
        ],
      );
}

/// One pre-recorded clip stored on the laptop
class VideoInfo {
  VideoInfo({required this.name, required this.relativePath, required this.sizeMb});

  final String name;
  final String relativePath;
  final double sizeMb;

  factory VideoInfo.fromJson(Map<String, dynamic> json) => VideoInfo(
        name: json['name'] as String,
        relativePath: json['relative_path'] as String,
        sizeMb: (json['size_mb'] as num?)?.toDouble() ?? 0,
      );
}


/// A progression decision from the coach, with its Ukrainian phrasing
class CoachAdvice {
  CoachAdvice({
    required this.action,
    required this.reason,
    required this.changed,
    required this.textUk,
    required this.textSource,
    required this.romTarget,
    required this.reps,
    required this.sets,
    this.unlockedExerciseId,
    this.textError,
    this.evidence = const {},
  });

  final String action;
  final String reason;
  final bool changed;
  final String textUk;

  /// 'gemini' or 'template'
  final String textSource;
  final String? textError;

  final double romTarget;
  final int reps;
  final int sets;
  final String? unlockedExerciseId;

  /// The numbers the decision was made from, for the "why" expander
  final Map<String, dynamic> evidence;

  bool get generated => textSource == 'gemini';

  factory CoachAdvice.fromJson(Map<String, dynamic> json) {
    final p = (json['prescription'] as Map).cast<String, dynamic>();
    return CoachAdvice(
      action: json['action'] as String,
      reason: json['reason'] as String? ?? '',
      changed: json['changed'] as bool? ?? false,
      textUk: json['text_uk'] as String? ?? '',
      textSource: json['text_source'] as String? ?? 'template',
      textError: json['text_error'] as String?,
      romTarget: (p['rom_target'] as num).toDouble(),
      reps: (p['reps'] as num).toInt(),
      sets: (p['sets'] as num).toInt(),
      unlockedExerciseId: json['unlocked_exercise_id'] as String?,
      evidence: (json['evidence'] as Map? ?? {}).cast<String, dynamic>(),
    );
  }
}

class ApiException implements Exception {
  ApiException(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Client for the rehab-ml service, used only for the laptop video mode
class RehabApi {
  RehabApi(this.baseUrl, {http.Client? client})
      : _client = client ?? http.Client();

  final String baseUrl;
  final http.Client _client;

  static const Duration _timeout = Duration(seconds: 20);

  Future<Map<String, dynamic>> health() async {
    final body = await _get('/health');
    return (body as Map).cast<String, dynamic>();
  }

  /// Wire body for a window, so a recording can be stored and re-scored as is
  Future<Map<String, Object>> recordingPayload({
    required String exerciseId,
    required BodySide side,
    required List<PoseFrame> frames,
  }) =>
      _wire(exerciseId, side, frames);

  Future<List<ExerciseInfo>> exercises() async {
    final body = await _get('/exercises') as List;
    return [
      for (final e in body) ExerciseInfo.fromJson((e as Map).cast<String, dynamic>()),
    ];
  }

  /// Clips in the service's video directory, if it has one
  Future<List<VideoInfo>> videos() async {
    final body = await _get('/videos') as List;
    return [
      for (final v in body) VideoInfo.fromJson((v as Map).cast<String, dynamic>()),
    ];
  }

  /// Assesses a clip stored on the laptop, not on the phone
  Future<Assessment> assessVideo({
    required String exerciseId,
    required BodySide side,
    required String relativePath,
  }) async {
    final body = await _post(
      '/assess/video',
      {
        'exercise_id': exerciseId,
        'side': side.wire,
        'relative_path': relativePath,
      },
      timeout: const Duration(minutes: 5),
    );
    return Assessment((body as Map).cast<String, dynamic>());
  }

  /// Checks the framing before the set starts
  Future<Readiness> checkReadiness({
    required String exerciseId,
    required BodySide side,
    required List<PoseFrame> frames,
  }) async {
    if (frames.isEmpty) throw ApiException('no frames to check');
    final body = await _post('/check/readiness', await _wire(exerciseId, side, frames));
    return Readiness.fromJson((body as Map).cast<String, dynamic>());
  }

  /// Posts one already-segmented repetition
  Future<Assessment> assessRepetition({
    required String exerciseId,
    required BodySide side,
    required List<PoseFrame> frames,
  }) =>
      _assess('/assess/repetition', exerciseId, side, frames);

  /// Posts a whole set, segmented into repetitions on the other side
  Future<Assessment> assessSession({
    required String exerciseId,
    required BodySide side,
    required List<PoseFrame> frames,
  }) =>
      _assess('/assess/session', exerciseId, side, frames);

  /// Builds the wire body. The frame rate is measured from the timestamps, never assumed
  Future<Map<String, Object>> _wire(
    String exerciseId,
    BodySide side,
    List<PoseFrame> frames,
  ) async {
    return {
      'exercise_id': exerciseId,
      'side': side.wire,
      'fps': effectiveFps([for (final f in frames) f.timestampMs]),
      'landmarks': [for (final f in frames) f.toNestedWorld()],
      'visibility': [for (final f in frames) f.toVisibilityList()],
    };
  }

  Future<Assessment> _assess(
    String path,
    String exerciseId,
    BodySide side,
    List<PoseFrame> frames,
  ) async {
    if (frames.isEmpty) throw ApiException('no frames to score');
    final body = await _post(
      path,
      await _wire(exerciseId, side, frames),
      offloadEncode: frames.length > _offloadAboveFrames,
      // Whole sets take longer than single repetitions
      timeout: frames.length > _offloadAboveFrames
          ? const Duration(seconds: 60)
          : null,
    );
    return Assessment((body as Map).cast<String, dynamic>());
  }

  /// Uniformly thins a window to fit `api.schemas.MAX_FRAMES`, keeping the tail
  static List<PoseFrame> fitToCap(List<PoseFrame> frames, {int cap = 5400}) {
    if (frames.length <= cap) return frames;
    final stride = (frames.length / cap).ceil();
    return [
      for (var i = 0; i < frames.length; i += stride) frames[i],
    ];
  }


  /// Requests the next dose from the stored history
  Future<CoachAdvice> coach({
    required String exerciseId,
    required BodySide side,
    required List<Map<String, dynamic>> history,
    double? romTarget,
    int? reps,
    int? sets,
    List<String> unlocked = const [],
  }) async {
    final body = await _post('/coach', {
      'exercise_id': exerciseId,
      'side': side.wire,
      'history': history,
      if (romTarget != null) 'rom_target': romTarget,
      if (reps != null) 'reps': reps,
      if (sets != null) 'sets': sets,
      'unlocked': unlocked,
    });
    return CoachAdvice.fromJson((body as Map).cast<String, dynamic>());
  }

  Future<Object?> _get(String path) async {
    try {
      final response =
          await _client.get(Uri.parse('$baseUrl$path')).timeout(_timeout);
      return _decode(response);
    } on ApiException {
      rethrow;
    } catch (e) {
      throw ApiException('cannot reach $baseUrl$path - $e');
    }
  }

  /// Above this many frames, JSON is encoded off the UI isolate
  static const int _offloadAboveFrames = 400;

  Future<Object?> _post(
    String path,
    Object payload, {
    Duration? timeout,
    bool offloadEncode = false,
  }) async {
    try {
      final encoded =
          offloadEncode ? await compute(jsonEncode, payload) : jsonEncode(payload);
      final response = await _client
          .post(
            Uri.parse('$baseUrl$path'),
            headers: {'Content-Type': 'application/json'},
            body: encoded,
          )
          .timeout(timeout ?? _timeout);
      return _decode(response);
    } on ApiException {
      rethrow;
    } catch (e) {
      throw ApiException('cannot reach $baseUrl$path - $e');
    }
  }

  Object? _decode(http.Response response) {
    if (response.statusCode >= 400) {
      throw ApiException('HTTP ${response.statusCode}: ${response.body}');
    }
    return jsonDecode(utf8.decode(response.bodyBytes));
  }

  void close() => _client.close();
}
