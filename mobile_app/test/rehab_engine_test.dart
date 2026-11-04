import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:intelligent_rehab/analysis/live_angles.dart';
import 'package:intelligent_rehab/pose/pose_stream.dart';
import 'package:intelligent_rehab/scoring/rehab_engine.dart';

/// Testing the engine as the screens use it: bundled assets, PoseFrames, the isolate
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final fixture = jsonDecode(File('test/golden/video_test_ex_right.json').readAsStringSync())
      as Map<String, dynamic>;
  final input = (fixture['input'] as Map).cast<String, dynamic>();
  final fps = (input['fps'] as num).toDouble();

  List<PoseFrame> frames() {
    final landmarks = input['landmarks'] as List;
    final visibility = input['visibility'] as List;
    return [
      for (var t = 0; t < landmarks.length; t++)
        PoseFrame(
          world: Float32List.fromList([
            for (final p in landmarks[t] as List)
              for (final v in p as List) (v as num).toDouble(),
          ]),
          visibility: Float32List.fromList(
              [for (final v in visibility[t] as List) (v as num).toDouble()]),
          screen: Float32List(66),
          timestampMs: (t * 1000 / fps).round(),
          frameWidth: 720,
          frameHeight: 1280,
        ),
    ];
  }

  test('loads the bundled thresholds and model', () async {
    final engine = await RehabEngine.load();
    final health = engine.health();
    expect(health['thresholds'], thresholdsAsset);
    expect(health['model'], isNotNull, reason: '${health['model_error']}');
    expect((health['model'] as Map)['mirror_augment'], false);
    expect((health['model'] as Map)['fitted_on_exercises'],
        containsAll(['elbow_flexion_extension', 'shoulder_abduction_seated', 'shoulder_flexion_seated']));
    expect(engine.exercises(), hasLength(6));
  });

  test('a set scored through the isolate matches the expected session', () async {
    final engine = await RehabEngine.load();
    final watch = Stopwatch()..start();
    final assessment = await engine.assessSession(
      exerciseId: input['exercise_id'] as String,
      side: BodySide.right,
      frames: frames(),
    );
    watch.stop();
    // ignore: avoid_print
    print('scored ${frames().length} frames in ${watch.elapsedMilliseconds} ms');
    // Phone config, jerky_movement excluded; the clip passes the view gate
    final want = (fixture['expected'] as Map)['session_no_view_check_no_jerky'] as Map<String, dynamic>;
    expect((engine.health()['excluded_criteria'] as List), ['jerky_movement']);
    expect(assessment.nReps, want['n_reps']);
    expect(assessment.usable, want['usable']);
    // Millisecond timestamps shift the measured rate slightly
    expect(assessment.meanEqi!, closeTo((want['mean_eqi'] as num).toDouble(), 0.05));
    expect(assessment.repetitions.first.json['model'], isNotNull);
  });

  test('an exercise the model never saw is scored by the rules alone', () async {
    final engine = await RehabEngine.load();
    final assessment = await engine.assessRepetition(
      exerciseId: 'hand_to_mouth',
      side: BodySide.right,
      frames: frames().sublist(0, 120),
    );
    expect(assessment.eqi, isNotNull);
    expect((assessment.json['model'] as Map)['skipped'], isA<String>());
  });

  test('the coach decides without a network', () async {
    final engine = await RehabEngine.load();
    final advice = await engine.coach(
      exerciseId: 'elbow_flexion_extension',
      side: BodySide.left,
      history: [
        for (var i = 0; i < 3; i++)
          {
            'exercise_id': 'elbow_flexion_extension',
            'mean_eqi': 92.0,
            'n_reps': 10,
            'correct_ratio': 0.9,
            'fatigue_trend': 0.0,
            'error_frequency': <String, double>{},
          },
      ],
      unlocked: ['elbow_flexion_extension'],
    );
    expect(advice.action, 'advance_range');
    expect(advice.romTarget, greaterThan(110));
    // No API key in tests, so the template is used
    expect(advice.textSource, 'template');
    expect(advice.textUk, isNotEmpty);
  });

  test('live girdle hike subtracts the obligatory scapular rise', () {
    const angles = LiveAngles(
      channel: LiveChannel.shoulderAbduction,
      primaryDeg: 90,
      compensationDeg: 0,
      elevationDeg: 90,
      girdleHeightTl: 0.70,
    );
    // 0.00071 TL per degree above 30 deg
    expect(angles.girdleHikeTl(0.60), closeTo(0.10 - 0.00071 * 60, 1e-12));
  });
}
