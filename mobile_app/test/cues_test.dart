import 'package:flutter_test/flutter_test.dart';
import 'package:intelligent_rehab/analysis/cues.dart';

/// Shoulder abduction, seated: target 90, ceiling 150 (from the protocol).
RepPhase phaseAt(
  double angle, {
  bool rising = true,
  bool calibrating = false,
  bool inRep = true,
}) =>
    repPhase(
      angleDeg: angle,
      target: 90.0,
      ceiling: 150.0,
      rising: rising,
      calibrating: calibrating,
      detectorInRep: inRep,
    );

void main() {
  group('repPhase', () {
    test('says raise from rest, not "higher"', () {
      expect(phaseAt(8, inRep: false), RepPhase.resting);
    });

    test('cues higher only while below target', () {
      expect(phaseAt(40), RepPhase.raising);
      expect(phaseAt(70), RepPhase.raising);
    });

    test('stops cueing higher at the target', () {
      expect(phaseAt(90), RepPhase.atTarget);
    });

    /// The reported bug: the arm raised well above the head still read as "higher"
    /// because the old cue was keyed on the detector's hysteresis flag, which stays
    /// true for the whole repetition.
    test('an arm above the head is never told to go higher', () {
      for (final angle in [95.0, 120.0, 150.0, 170.0, 179.0]) {
        final phase = phaseAt(angle);
        expect(phase, isNot(RepPhase.raising), reason: '$angle deg read as raising');
      }
    });

    test('past the ceiling it says lower, not higher', () {
      expect(phaseAt(165), RepPhase.overshoot);
    });

    test('tolerates jitter just below the target', () {
      // 5 deg band: a patient holding at 86 must not be nagged upward.
      expect(phaseAt(86), RepPhase.atTarget);
      expect(phaseAt(84), RepPhase.raising);
    });

    test('descending below target reads as lowering', () {
      expect(phaseAt(60, rising: false), RepPhase.lowering);
    });

    test('calibrating wins over everything', () {
      expect(phaseAt(140, calibrating: true), RepPhase.calibrating);
    });
  });

  group('DirectionTracker', () {
    test('reports rising on a monotonic lift', () {
      final t = DirectionTracker();
      for (final v in [10.0, 20.0, 35.0, 50.0, 70.0]) {
        t.add(v);
      }
      expect(t.rising, isTrue);
    });

    test('flips on a descent', () {
      final t = DirectionTracker();
      for (final v in [10.0, 40.0, 90.0, 70.0, 40.0]) {
        t.add(v);
      }
      expect(t.rising, isFalse);
    });

    /// The reason the deadband exists: holding at end range produces small
    /// alternating deltas, and a naive sign test flips the cue every frame.
    test('sub-deadband jitter does not flip direction', () {
      final t = DirectionTracker();
      t.add(10.0);
      t.add(90.0); // clear rise
      for (final v in [90.3, 89.8, 90.2, 89.9, 90.1]) {
        t.add(v);
      }
      expect(t.rising, isTrue);
    });

    /// Drift smaller than the deadband must still accumulate, or a slow lowering
    /// would never be detected at all.
    test('slow drift below the deadband still accumulates', () {
      final t = DirectionTracker(deadbandDeg: 0.8);
      t.add(90.0);
      // Five 0.3-degree steps down: no single step crosses 0.8, the total is 1.5.
      for (final v in [89.7, 89.4, 89.1, 88.8, 88.5]) {
        t.add(v);
      }
      expect(t.rising, isFalse);
    });
  });
}
