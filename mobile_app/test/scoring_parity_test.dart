import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:intelligent_rehab/scoring/adaptation.dart';
import 'package:intelligent_rehab/scoring/angles.dart';
import 'package:intelligent_rehab/scoring/exercises.dart';
import 'package:intelligent_rehab/scoring/kinematics.dart';
import 'package:intelligent_rehab/scoring/model.dart';
import 'package:intelligent_rehab/scoring/rehab_engine.dart';
import 'package:intelligent_rehab/scoring/rules.dart';
import 'package:intelligent_rehab/scoring/signal.dart';

/// Comparing the Dart scoring pipeline against the Python fixtures in `test/golden/`
///
/// Fixtures come from `ml/scripts/export_dart.py`. Continuous values use float32
/// tolerances; discrete values must match exactly.
void main() {
  final golden = Directory('test/golden');
  Map<String, dynamic> load(String name) =>
      jsonDecode(File('${golden.path}/$name').readAsStringSync()) as Map<String, dynamic>;

  final core = ScoringCore(
    thresholds: ThresholdFile.parse(
      File('assets/thresholds/thresholds_irds_patients.json').readAsStringSync(),
      source: 'thresholds_irds_patients.json',
    ),
    weights: HybridWeights.fromBytes(
      ByteData.sublistView(File('assets/model/hybrid.bin').readAsBytesSync()),
      File('assets/model/hybrid.json').readAsStringSync(),
    ),
  );

  final phoneCore = ScoringCore(
    thresholds: core.thresholds,
    excluded: RehabEngine.phoneExcludedCriteria,
  );

  group('signal routines match scipy', () {
    final s = load('signal.json');
    List<double> v(Object? x) => [for (final e in x as List) (e as num).toDouble()];

    test('butter', () {
      for (final c in (s['butter'] as List).cast<Map>()) {
        final ba = butterLowpass(4, (c['wn'] as num).toDouble());
        expectClose(ba.b, v(c['b']), 1e-10, 'b');
        expectClose(ba.a, v(c['a']), 1e-10, 'a');
      }
    });

    test('filtfilt', () {
      for (final c in (s['filtfilt'] as List).cast<Map>()) {
        final ba = butterLowpass(4, (c['wn'] as num).toDouble());
        expectClose(filtfilt(ba.b, ba.a, v(c['x']), padlen: 15), v(c['y']), 1e-6, 'filtfilt');
      }
    });

    test('savgol', () {
      for (final c in (s['savgol'] as List).cast<Map>()) {
        expectClose(savgolFilter(v(c['x']), c['window'] as int, 2), v(c['y']), 1e-6, 'savgol');
      }
    });

    test('smooth_signal', () {
      for (final c in (s['smooth'] as List).cast<Map>()) {
        final y = smoothSignal(v(c['x']), (c['fps'] as num).toDouble(),
            cutoffHz: (c['cutoff'] as num).toDouble());
        expectClose(y, v(c['y']), 1e-6, 'smooth fps=${c['fps']} n=${(c['x'] as List).length}');
      }
    });

    test('find_peaks', () {
      for (final c in (s['find_peaks'] as List).cast<Map>()) {
        final k = (c['kwargs'] as Map).cast<String, dynamic>();
        final peaks = findPeaks(
          v(c['x']),
          height: (k['height'] as num?)?.toDouble(),
          distance: (k['distance'] as num?)?.toDouble(),
          prominence: (k['prominence'] as num?)?.toDouble(),
        );
        expect(peaks, (c['peaks'] as List).cast<int>(), reason: '$k');
      }
    });

    test('kinematics', () {
      for (final c in (s['kinematics'] as List).cast<Map>()) {
        final speed = v(c['speed']);
        expect(sparc(speed, 30), closeTo((c['sparc'] as num).toDouble(), 1e-6));
        expect(ldlj(speed, 30), closeTo((c['ldlj'] as num).toDouble(), 1e-6));
        expect(nSubmovements(speed, 30), c['n_submovements']);
        var cum = 0.0;
        final position = [for (final x in speed) (cum += x) / 30.0];
        final tempo = tempoMetrics(position, 30);
        final t = (c['tempo'] as Map).cast<String, dynamic>();
        expect(tempo.peakIndex, t['peak_index']);
        expect(tempo.holdS, closeTo((t['hold_s'] as num).toDouble(), 1e-9));
        expect(tempo.concentricS, closeTo((t['concentric_s'] as num).toDouble(), 1e-9));
      }
    });
  });

  final fixtures = golden
      .listSync()
      .whereType<File>()
      .map((f) => f.uri.pathSegments.last)
      .where((n) => n.endsWith('.json') && n != 'signal.json' && n != 'coach.json')
      .toList()
    ..sort();

  for (final name in fixtures) {
    group(name, () {
      final fixture = load(name);
      final wire = (fixture['input'] as Map).cast<String, dynamic>();
      final expected = (fixture['expected'] as Map).cast<String, dynamic>();
      final definition = core.resolve(wire['exercise_id'] as String);
      final side = parseSide(wire['side'] as String);
      final sequence = core.toSequence(wire);

      test('preprocess', () {
        final want = [
          for (final f in expected['preprocessed_joints'] as List)
            for (final j in f as List)
              for (final x in j as List) (x as num).toDouble(),
        ];
        expectClose([for (final x in sequence.joints) x.toDouble()], want, 1e-4, 'joints');
        final conf = [
          for (final f in expected['preprocessed_confidence'] as List)
            for (final x in f as List) (x as num).toDouble(),
        ];
        expectClose([for (final x in sequence.confidence) x.toDouble()], conf, 1e-3, 'confidence');
      });

      test('angles', () {
        final angles = computeAngles(sequence, side);
        final want = (expected['angles'] as Map).cast<String, dynamic>();
        for (final channel in angleChannels) {
          final tl = channel.startsWith('shoulder_height') || channel.startsWith('hand_');
          expectClose(angles[channel], [for (final x in want[channel] as List) (x as num).toDouble()],
              tl ? 1e-4 : 2e-3, channel);
        }
      });

      test('framing', () {
        final verdict = framingVerdict(sequence, definition, side);
        final want = (expected['framing'] as Map).cast<String, dynamic>();
        expect(verdict.viewOk, want['view_ok']);
        expect(verdict.jointsOk, want['joints_ok']);
        expect(verdict.detectedView, want['detected_view']);
        expect(verdict.yawDeg, closeTo((want['yaw_deg'] as num).toDouble(), 1e-3));
        expect(verdict.missingJoints, (want['missing_joints'] as List).cast<String>());
      });

      test('segmentation', () {
        final reps = segmentRepetitions(computeAngles(sequence, side), definition);
        expect([for (final r in reps) [r.start, r.end, r.peak]],
            [for (final s in expected['segments'] as List) (s as List).cast<int>()]);
      });

      /// The rule-engine part of a session; the model is compared separately
      Map<String, dynamic> withoutModel(Map<String, dynamic> session) {
        for (final r in (session['repetitions'] as List).cast<Map<String, dynamic>>()) {
          r.remove('model');
        }
        return session..remove('model');
      }

      test('session, as the app sees it', () {
        expectJsonClose(withoutModel(core.assessSession(wire)), expected['session'], 'session');
      });

      test('session without the view gate', () {
        expectJsonClose(withoutModel(core.assessSession(wire, checkView: false)),
            expected['session_no_view_check'], 'session_no_view_check');
      });

      test('session with jerky_movement excluded, as the phone runs', () {
        final want = expected['session_no_view_check_no_jerky'];
        if (want == null) return;
        expectJsonClose(withoutModel(phoneCore.assessSession(wire, checkView: false)), want,
            'session_no_jerky');
      });

      test('session model summary', () {
        final got = core.assessSession(wire, checkView: false);
        final reps = (got['repetitions'] as List).cast<Map<String, dynamic>>();
        final perRep = (expected['repetitions'] as List).cast<Map<String, dynamic>>();
        for (var i = 0; i < perRep.length; i++) {
          final want = perRep[i]['model'] as Map<String, dynamic>?;
          if (want == null) continue;
          final model = reps[i]['model'] as Map<String, dynamic>;
          expect(model['correct_prob'] as double, closeTo((want['correct_prob'] as num).toDouble(), 2e-3));
        }
      });

      test('model features and outputs', () {
        final angles = computeAngles(sequence, side);
        final scored = core.engine.assessSequence(sequence, definition, side, checkView: false);
        final perRep = (expected['repetitions'] as List).cast<Map<String, dynamic>>();
        expect(scored.session.repetitions.length, perRep.length);
        for (var i = 0; i < perRep.length; i++) {
          final want = perRep[i];
          final window = angles.slice(want['start'] as int, want['end'] as int);
          final seqF = sequenceFeatures(window);
          expectClose([for (final x in seqF) x.toDouble()],
              [for (final row in want['sequence_features'] as List) for (final x in row as List) (x as num).toDouble()],
              2e-3, 'sequence_features rep $i');
          expectClose([for (final x in summaryFeatures(window, definition)) x.toDouble()],
              [for (final x in want['summary_features'] as List) (x as num).toDouble()],
              2e-3, 'summary_features rep $i', relative: true);
          final model = want['model'] as Map<String, dynamic>?;
          if (model == null) continue;
          final got = core.runner!.predict(window, definition,
              ruleEqi: scored.session.repetitions[i].quality.value);
          expect(got['correct_prob'] as double, closeTo((model['correct_prob'] as num).toDouble(), 2e-3),
              reason: 'correct_prob rep $i');
          expect(got['eqi'] as double, closeTo((model['eqi'] as num).toDouble(), 0.05), reason: 'eqi rep $i');
          final errors = (model['errors'] as Map).cast<String, dynamic>();
          final gotErrors = (got['errors'] as Map).cast<String, double>();
          for (final e in errors.entries) {
            expect(gotErrors[e.key]!, closeTo((e.value as num).toDouble(), 2e-3), reason: '${e.key} rep $i');
          }
        }
      });

      test('single repetition, the live path', () {
        final single = expected['single_repetition'] as Map<String, dynamic>?;
        if (single == null) return;
        final start = single['start'] as int;
        final end = single['end'] as int;
        final window = {
          ...wire,
          'landmarks': (wire['landmarks'] as List).sublist(start, end),
          'visibility': (wire['visibility'] as List).sublist(start, end),
        };
        final got = core.assessRepetition(window);
        final want = (single['assessment'] as Map).cast<String, dynamic>();
        final gotModel = got.remove('model') as Map<String, dynamic>?;
        final wantModel = want.remove('model') as Map<String, dynamic>?;
        expectJsonClose(got, want, 'single');
        if (wantModel != null) {
          expect(gotModel!['correct_prob'] as double,
              closeTo((wantModel['correct_prob'] as num).toDouble(), 2e-3));
          expect(gotModel['eqi'] as double, closeTo((wantModel['eqi'] as num).toDouble(), 0.05));
        }
      });
    });
  }

  group('coach decisions match adaptation.decide', () {
    final cases = (jsonDecode(File('${golden.path}/coach.json').readAsStringSync()) as List)
        .cast<Map<String, dynamic>>();
    for (final c in cases) {
      test(c['name'] as String, () {
        final (decision, _, last) = core.decideFor((c['request'] as Map).cast<String, dynamic>());
        expectJsonClose(decision.asDict(), c['decision'], c['name'] as String);
        expect(templateText(decision, last), c['template_text']);
      });
    }
  });
}

void expectClose(List<double> got, List<double> want, double tol, String what, {bool relative = false}) {
  expect(got.length, want.length, reason: '$what length');
  var worst = 0.0;
  var at = -1;
  for (var i = 0; i < got.length; i++) {
    final scale = relative ? math.max(1.0, want[i].abs()) : 1.0;
    final d = (got[i] - want[i]).abs() / scale;
    if (d > worst || d.isNaN) {
      worst = d;
      at = i;
    }
  }
  expect(worst <= tol, isTrue,
      reason: '$what: worst deviation $worst at $at (got ${at >= 0 ? got[at] : '-'}, '
          'want ${at >= 0 ? want[at] : '-'}), tolerance $tol');
}

/// Numbers within the rounding of the service's JSON, everything else exact
void expectJsonClose(Object? got, Object? want, String path) {
  if (want is num && got is num) {
    final tol = math.max(0.0115, 1e-4 * want.abs());
    expect((got - want).abs() <= tol, isTrue, reason: '$path: got $got, want $want');
  } else if (want is Map) {
    expect(got, isA<Map>(), reason: path);
    final g = got as Map;
    expect(g.keys.map((k) => '$k').toSet(), want.keys.map((k) => '$k').toSet(), reason: '$path keys');
    for (final k in want.keys) {
      expectJsonClose(g[k], want[k], '$path.$k');
    }
  } else if (want is List) {
    expect(got, isA<List>(), reason: path);
    final g = got as List;
    expect(g.length, want.length, reason: '$path length');
    for (var i = 0; i < want.length; i++) {
      expectJsonClose(g[i], want[i], '$path[$i]');
    }
  } else {
    expect(got, want, reason: path);
  }
}
