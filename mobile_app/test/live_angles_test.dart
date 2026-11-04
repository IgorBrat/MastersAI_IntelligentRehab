import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:intelligent_rehab/analysis/live_angles.dart';
import 'package:intelligent_rehab/pose/pose_stream.dart';

/// Validating the Dart angles against `rehab_ml.features.angles` on synthetic poses
///
/// `preprocess` is not applied on either side: it is zero-phase and cannot run live.
void main() {
  final file = File('test/live_angles_reference.json');
  final cases = (jsonDecode(file.readAsStringSync()) as List)
      .cast<Map<String, dynamic>>();

  PoseFrame frameFrom(List<dynamic> landmarks) {
    final world = Float32List(33 * 3);
    for (var i = 0; i < 33; i++) {
      final p = (landmarks[i] as List).cast<num>();
      world[i * 3] = p[0].toDouble();
      world[i * 3 + 1] = p[1].toDouble();
      world[i * 3 + 2] = p[2].toDouble();
    }
    return PoseFrame(
      world: world,
      visibility: Float32List.fromList(List.filled(33, 1.0)),
      screen: Float32List(66),
      timestampMs: 0,
      frameWidth: 720,
      frameHeight: 1280,
    );
  }

  /// Reading one channel from the call that selects it
  double channel(PoseFrame frame, BodySide side, String name) {
    switch (name) {
      case 'flexion':
        return LiveAngles.fromFrame(frame, side,
                channel: LiveChannel.shoulderFlexion)
            .primaryDeg;
      case 'abduction':
        return LiveAngles.fromFrame(frame, side,
                channel: LiveChannel.shoulderAbduction)
            .primaryDeg;
      case 'elevation':
        return LiveAngles.fromFrame(frame, side,
                channel: LiveChannel.shoulderElevation)
            .primaryDeg;
      case 'elbow':
        return LiveAngles.fromFrame(frame, side,
                channel: LiveChannel.elbowFlexion)
            .primaryDeg;
      case 'trunk_forward':
        // A magnitude, compared against the reference's absolute value
        return LiveAngles.fromFrame(frame, side,
                channel: LiveChannel.shoulderFlexion)
            .compensationDeg;
      case 'trunk_tilt':
        return LiveAngles.fromFrame(frame, side,
                channel: LiveChannel.shoulderAbduction)
            .compensationDeg;
      default:
        throw ArgumentError('unknown channel $name');
    }
  }

  group('LiveAngles matches rehab_ml.features.angles', () {
    test('the reference fixture is present and non-trivial', () {
      expect(cases.length, greaterThanOrEqualTo(10));
    });

    for (final testCase in cases) {
      final label = testCase['label'] as String;
      final side =
          testCase['side'] == 'right' ? BodySide.right : BodySide.left;
      final expected = (testCase['expected'] as Map).cast<String, dynamic>();
      final assertable =
          (testCase['assert_channels'] as List).cast<String>();

      test('$label (${side.wire})', () {
        final frame = frameFrom(testCase['landmarks'] as List);
        for (final name in assertable) {
          final want = (expected[name] as num).toDouble();
          final got = channel(frame, side, name);
          // The compensation channels are magnitudes on the Dart side
          final target = (name == 'trunk_forward' || name == 'trunk_tilt')
              ? want.abs()
              : want;
          expect(
            got,
            closeTo(target, 0.02),
            reason: '$label: $name expected $target, port gave $got',
          );
        }
      });
    }
  });

  group('LiveChannel', () {
    test('maps every protocol primary channel it claims to handle', () {
      expect(LiveChannel.fromPrimary('shoulder_flexion'),
          LiveChannel.shoulderFlexion);
      expect(LiveChannel.fromPrimary('shoulder_abduction'),
          LiveChannel.shoulderAbduction);
      expect(LiveChannel.fromPrimary('elbow_flexion'),
          LiveChannel.elbowFlexion);
    });

    test('falls back to elevation for channels not derived on device', () {
      // hand_up, hand_forward and shoulder_rotation are not ported
      for (final name in ['hand_up', 'hand_forward', 'shoulder_rotation', '']) {
        expect(LiveChannel.fromPrimary(name), LiveChannel.shoulderElevation,
            reason: name);
      }
    });

    test('every channel has its own Ukrainian cues', () {
      final starts = {for (final c in LiveChannel.values) c.startCueUk};
      expect(starts.length, LiveChannel.values.length,
          reason: 'a shared cue would tell the patient the wrong movement');
      for (final c in LiveChannel.values) {
        expect(c.compensationCueUk, isNotEmpty);
      }
    });
  });

  group('geometry primitives', () {
    test('signedAngle carries the sign the unsigned one loses', () {
      const a = V3(0, 1, 0);
      const b = V3(0, 0, 1);
      const normal = V3(1, 0, 0);
      expect(signedAngle(a, b, normal), closeTo(90.0, 1e-9));
      expect(signedAngle(b, a, normal), closeTo(-90.0, 1e-9));
      expect(angleBetween(a, b), closeTo(90.0, 1e-9));
      expect(angleBetween(b, a), closeTo(90.0, 1e-9));
    });

    test('projectOntoPlane removes exactly the normal component', () {
      const v = V3(1, 2, 3);
      const normal = V3(0, 1, 0);
      final projected = projectOntoPlane(v, normal);
      expect(projected.y, closeTo(0.0, 1e-9));
      expect(projected.x, closeTo(1.0, 1e-9));
      expect(projected.z, closeTo(3.0, 1e-9));
    });

    test('unit of a degenerate vector does not divide by zero', () {
      expect(V3.zero.unit.length, closeTo(0.0, 1e-6));
    });
  });

  group('Smoother', () {
    test('passes the first sample through unchanged', () {
      expect(Smoother().add(42.0), closeTo(42.0, 1e-9));
    });

    test('lags rather than looking ahead', () {
      final s = Smoother(alpha: 0.5);
      s.add(0.0);
      expect(s.add(10.0), closeTo(5.0, 1e-9));
      expect(s.add(10.0), closeTo(7.5, 1e-9));
    });

    test('reset forgets the previous session', () {
      final s = Smoother();
      s.add(90.0);
      s.reset();
      expect(s.value, isNull);
      expect(s.add(10.0), closeTo(10.0, 1e-9));
    });
  });
}
