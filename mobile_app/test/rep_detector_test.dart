import 'package:flutter_test/flutter_test.dart';
import 'package:intelligent_rehab/analysis/rep_detector.dart';

/// Feeding a rest, one lift and one lower into the detector
List<DetectedRep> drive(
  RepDetector detector, {
  required double fps,
  double peak = 100.0,
  double baseline = 5.0,
  double restSeconds = 0.6,
  double phaseSeconds = 1.2,
  int startMs = 0,
}) {
  final step = 1000 / fps;
  final reps = <DetectedRep>[];
  var t = startMs.toDouble();

  void feed(double value) {
    final rep = detector.update(value, t.round());
    if (rep != null) reps.add(rep);
    t += step;
  }

  final restFrames = (restSeconds * fps).round();
  final phaseFrames = (phaseSeconds * fps).round();

  for (var i = 0; i < restFrames; i++) {
    feed(baseline);
  }
  for (var i = 0; i <= phaseFrames; i++) {
    feed(baseline + (peak - baseline) * i / phaseFrames);
  }
  for (var i = phaseFrames; i >= 0; i--) {
    feed(baseline + (peak - baseline) * i / phaseFrames);
  }
  return reps;
}

void main() {
  group('effectiveFps', () {
    test('measures the rate from the span between first and last frame', () {
      final stamps = [for (var i = 0; i < 31; i++) i * 50]; // 20 fps
      expect(effectiveFps(stamps), closeTo(20.0, 1e-9));
    });

    test('reports the real rate when the camera drops frames', () {
      // 30 fps nominal, every third frame delivered
      final stamps = [for (var i = 0; i < 20; i++) i * 100];
      expect(effectiveFps(stamps), closeTo(10.0, 1e-9));
    });

    test('falls back rather than returning an implausible rate', () {
      expect(effectiveFps([]), 30.0);
      expect(effectiveFps([100]), 30.0);
      expect(effectiveFps([100, 100, 100]), 30.0, reason: 'no time elapsed');
      expect(effectiveFps([100, 50]), 30.0, reason: 'timestamps go backwards');
      expect(effectiveFps([0, 1, 2, 3]), 30.0, reason: '1000 fps is not a camera');
      expect(effectiveFps([0, 1000000]), 30.0, reason: 'one frame per 1000 s');
    });

    test('honours a caller-supplied fallback', () {
      expect(effectiveFps([], fallback: 24.0), 24.0);
    });
  });

  group('RepDetector', () {
    test('detects one repetition from a lift and lower', () {
      final detector = RepDetector(targetDeg: 90.0);
      final reps = drive(detector, fps: 30.0);
      expect(reps.length, 1);
      expect(detector.completedReps, 1);
      expect(reps.single.peakValue, greaterThan(90.0));
    });

    test('counts the same repetition whatever the real frame rate', () {
      for (final fps in [10.0, 15.0, 24.0, 30.0, 60.0]) {
        final reps = drive(RepDetector(targetDeg: 90.0), fps: fps);
        expect(reps.length, 1, reason: 'at $fps fps');
      }
    });

    test('measures duration in wall-clock, not frames', () {
      final slow = drive(RepDetector(targetDeg: 90.0), fps: 10.0).single;
      final fast = drive(RepDetector(targetDeg: 90.0), fps: 60.0).single;
      // Same movement, same seconds, six times the frames
      expect(slow.duration.inMilliseconds,
          closeTo(fast.duration.inMilliseconds, 250));
    });

    test('rejects a twitch shorter than the minimum duration', () {
      // 0.3 s total, under minDurationSeconds = 0.8
      final reps = drive(RepDetector(targetDeg: 90.0), fps: 60.0, phaseSeconds: 0.15);
      expect(reps, isEmpty);
    });

    test('learns the baseline from the resting pose, taking the minimum', () {
      final detector = RepDetector(targetDeg: 90.0);
      drive(detector, fps: 30.0, baseline: 7.0);
      expect(detector.baseline, closeTo(7.0, 1e-6));
    });

    test('is calibrating until the baseline window has elapsed', () {
      final detector = RepDetector(targetDeg: 90.0);
      expect(detector.calibrating, isTrue, reason: 'before any frame');
      detector.update(5.0, 0);
      expect(detector.calibrating, isTrue);
      detector.update(5.0, 200);
      expect(detector.calibrating, isTrue, reason: '0.2 s of a 0.5 s window');
      detector.update(5.0, 600);
      expect(detector.calibrating, isFalse);
    });

    test('reset clears every field so one detector can start a new session', () {
      final detector = RepDetector(targetDeg: 90.0);
      drive(detector, fps: 30.0);
      expect(detector.completedReps, 1);
      detector.reset();
      expect(detector.completedReps, 0);
      expect(detector.baseline, isNull);
      expect(detector.inRepetition, isFalse);
      expect(detector.calibrating, isTrue);
    });

    /// Timestamps come from the camera and need not start at zero
    test('tolerates a stream that starts at an arbitrary clock', () {
      final reps = drive(RepDetector(targetDeg: 90.0), fps: 30.0, startMs: 1738000000000);
      expect(reps.length, 1);
    });
  });

  /// The window bounds decide which frames get scored
  group('RepDetector window bounds', () {
    /// Replaying the same ramp as [drive] while recording the timeline
    ({List<DetectedRep> reps, int riseCrossedMs, int returnedMs}) traced({
      double fps = 30.0,
      double peak = 100.0,
      double baseline = 5.0,
      double target = 90.0,
      double restSeconds = 0.6,
      double phaseSeconds = 1.2,
    }) {
      final detector = RepDetector(targetDeg: target);
      final excursion = (target - baseline).abs();
      final riseLevel = baseline + 0.55 * excursion;
      final restLevel = baseline + 0.15 * excursion;

      final step = 1000 / fps;
      final reps = <DetectedRep>[];
      var t = 0.0;
      var riseCrossedMs = -1;
      var returnedMs = -1;
      var wasUp = false;

      void feed(double value) {
        final now = t.round();
        if (riseCrossedMs < 0 && value >= riseLevel) riseCrossedMs = now;
        if (value >= riseLevel) wasUp = true;
        if (wasUp && returnedMs < 0 && value <= restLevel) returnedMs = now;
        final rep = detector.update(value, now);
        if (rep != null) reps.add(rep);
        t += step;
      }

      final restFrames = (restSeconds * fps).round();
      final phaseFrames = (phaseSeconds * fps).round();
      for (var i = 0; i < restFrames; i++) {
        feed(baseline);
      }
      for (var i = 0; i <= phaseFrames; i++) {
        feed(baseline + (peak - baseline) * i / phaseFrames);
      }
      for (var i = phaseFrames; i >= 0; i--) {
        feed(baseline + (peak - baseline) * i / phaseFrames);
      }
      return (reps: reps, riseCrossedMs: riseCrossedMs, returnedMs: returnedMs);
    }

    test('the window starts before the rise threshold was crossed', () {
      final run = traced();
      final rep = run.reps.single;
      // The detector fires half way up; the window must include the lift before it
      expect(rep.startMs, lessThan(run.riseCrossedMs));
    });

    test('the window covers the return to rest, not just the 30% crossing', () {
      final run = traced();
      final rep = run.reps.single;
      expect(run.returnedMs, greaterThan(0), reason: 'the ramp does return');
      expect(rep.endMs, greaterThanOrEqualTo(run.returnedMs));
      expect(rep.returnedToRest, isTrue);
    });

    test('the window spans most of the movement, not a fragment of it', () {
      final run = traced(restSeconds: 0.6, phaseSeconds: 1.2);
      // 1.2 s up plus 1.2 s down
      expect(run.reps.single.duration.inMilliseconds, greaterThan(2000));
    });

    test('a descent that stalls above rest still yields a scored repetition', () {
      // Counted with returnedToRest false, so `incomplete_return` can grade it
      final detector = RepDetector(targetDeg: 90.0);
      final reps = <DetectedRep>[];
      var t = 0;
      void feed(double v) {
        final rep = detector.update(v, t);
        if (rep != null) reps.add(rep);
        t += 33;
      }

      for (var i = 0; i < 20; i++) {
        feed(5.0);
      }
      for (var i = 0; i <= 40; i++) {
        feed(5.0 + 90.0 * i / 40);
      }
      // 28 deg: below the 30% fall level, above the 15% rest band
      for (var i = 0; i < 60; i++) {
        feed(28.0);
      }

      expect(reps.length, 1);
      expect(reps.single.returnedToRest, isFalse);
    });

    test('a hold at the top cannot stall the repetition forever', () {
      final detector = RepDetector(targetDeg: 90.0, maxDurationSeconds: 3.0);
      final reps = <DetectedRep>[];
      var t = 0;
      void feed(double v) {
        final rep = detector.update(v, t);
        if (rep != null) reps.add(rep);
        t += 33;
      }

      for (var i = 0; i < 20; i++) {
        feed(5.0);
      }
      for (var i = 0; i < 200; i++) {
        feed(95.0);
      }

      expect(reps.length, 1, reason: 'force-emitted at maxDurationSeconds');
      expect(reps.single.returnedToRest, isFalse);
    });

    test('a wobble during the descent is one repetition, not two', () {
      final detector = RepDetector(targetDeg: 90.0);
      final reps = <DetectedRep>[];
      var t = 0;
      void feed(double v) {
        final rep = detector.update(v, t);
        if (rep != null) reps.add(rep);
        t += 33;
      }

      for (var i = 0; i < 20; i++) {
        feed(5.0);
      }
      for (var i = 0; i <= 30; i++) {
        feed(5.0 + 90.0 * i / 30);
      }
      // Dips into the fall band, goes back up, then comes all the way down
      for (var i = 0; i < 8; i++) {
        feed(28.0);
      }
      for (var i = 0; i < 10; i++) {
        feed(80.0);
      }
      for (var i = 0; i < 20; i++) {
        feed(5.0);
      }

      expect(reps.length, 1);
      expect(reps.single.peakValue, greaterThan(90.0));
    });
  });

  _registerDriftedRestTests();
}

/// Lifts that return only to the fall band, above the baseline learned at the start
void driveDriftedRest(RepDetector detector, int lifts) {
  const fps = 30.0;
  const step = 1000 / fps;
  var t = 0.0;

  void feed(double value) {
    detector.update(value, t.round());
    t += step;
  }

  // Arm pressed to the body, baseline learned low
  for (var i = 0; i < (0.6 * fps).round(); i++) {
    feed(5.0);
  }

  // Returns to 28 deg: above rest (17.75), below fall (30.5)
  for (var rep = 0; rep < lifts; rep++) {
    final frames = (1.2 * fps).round();
    for (var i = 0; i <= frames; i++) {
      feed(28.0 + (100.0 - 28.0) * i / frames);
    }
    for (var i = frames; i >= 0; i--) {
      feed(28.0 + (100.0 - 28.0) * i / frames);
    }
    for (var i = 0; i < (1.0 * fps).round(); i++) {
      feed(28.0);
    }
  }
}

void _registerDriftedRestTests() {
  group('rest angle above the learned baseline', () {
    test('counts every repetition instead of latching on the first', () {
      final detector = RepDetector(targetDeg: 90.0);
      driveDriftedRest(detector, 8);
      expect(detector.completedReps, 8);
    });

    test('an arm simply held up still mints no repetitions', () {
      final detector = RepDetector(targetDeg: 90.0);
      const fps = 30.0;
      var t = 0.0;
      void feed(double v) {
        detector.update(v, t.round());
        t += 1000 / fps;
      }

      for (var i = 0; i < (0.6 * fps).round(); i++) {
        feed(5.0);
      }
      // Up once, then held above the fall band for a minute
      final frames = (1.2 * fps).round();
      for (var i = 0; i <= frames; i++) {
        feed(5.0 + 95.0 * i / frames);
      }
      for (var i = 0; i < (60 * fps).round(); i++) {
        feed(100.0);
      }
      // One forced emit at maxDurationSeconds is allowed
      expect(detector.completedReps, lessThanOrEqualTo(1));
    });
  });
}
