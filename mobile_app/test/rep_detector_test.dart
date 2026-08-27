import 'package:flutter_test/flutter_test.dart';
import 'package:intelligent_rehab/analysis/rep_detector.dart';

/// Repetition detection and the measured frame rate
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
      // 30 fps nominal, every third frame delivered: 10 fps in reality.
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
      // 0.15 s up and 0.15 s down is well inside minDurationSeconds = 0.8.
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
}
