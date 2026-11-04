import 'package:flutter_test/flutter_test.dart';
import 'package:intelligent_rehab/analysis/cues.dart';

/// Seated shoulder abduction: target 90, ceiling 150
RepPhase phaseAt(
  double angle, {
  bool rising = true,
  bool calibrating = false,
  bool inRep = true,
  double heldFor = 0.0,
  double holdSeconds = 2.0,
}) =>
    repPhase(
      angleDeg: angle,
      target: 90.0,
      ceiling: 150.0,
      rising: rising,
      calibrating: calibrating,
      detectorInRep: inRep,
      secondsAtTarget: heldFor,
      holdSeconds: holdSeconds,
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
      // 5 deg band below the target
      expect(phaseAt(86), RepPhase.atTarget);
      expect(phaseAt(84), RepPhase.raising);
    });

    test('descending below target reads as lowering', () {
      expect(phaseAt(60, rising: false), RepPhase.lowering);
    });

    test('calibrating wins over everything', () {
      expect(phaseAt(140, calibrating: true), RepPhase.calibrating);
    });

    test('the hold ends in an instruction to come down', () {
      expect(phaseAt(95, heldFor: 0.0), RepPhase.atTarget);
      expect(phaseAt(95, heldFor: 1.9), RepPhase.atTarget);
      expect(phaseAt(95, heldFor: 2.0), RepPhase.lowerNow);
      expect(phaseAt(95, heldFor: 6.0), RepPhase.lowerNow,
          reason: 'holding too long must not revert to "hold"');
    });

    /// The caller substitutes a cueing default for a zero hold, see `_holdCueSeconds`
    test('a zero hold is the callers problem, not this functions', () {
      expect(phaseAt(95, heldFor: 0.0, holdSeconds: 0.0), RepPhase.lowerNow);
    });

    test('lowerNow is an instruction, lowering is an observation', () {
      expect(phaseAt(95, heldFor: 3.0, rising: false), RepPhase.lowerNow,
          reason: 'still at the top, whatever the direction of travel');
      expect(phaseAt(60, rising: false, heldFor: 3.0), RepPhase.lowering);
    });

    /// The caller starts the hold clock on band entry, so 84 deg has no dwell
    test('the hold is only counted inside the band', () {
      expect(phaseAt(84, heldFor: 9.0), RepPhase.raising);
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

    /// Holding at end range gives small alternating deltas
    test('sub-deadband jitter does not flip direction', () {
      final t = DirectionTracker();
      t.add(10.0);
      t.add(90.0); // clear rise
      for (final v in [90.3, 89.8, 90.2, 89.9, 90.1]) {
        t.add(v);
      }
      expect(t.rising, isTrue);
    });

    test('slow drift below the deadband still accumulates', () {
      final t = DirectionTracker(deadbandDeg: 0.8);
      t.add(90.0);
      // Five 0.3 deg steps: none crosses 0.8, the total is 1.5
      for (final v in [89.7, 89.4, 89.1, 88.8, 88.5]) {
        t.add(v);
      }
      expect(t.rising, isFalse);
    });
  });
}
