/// Holding one detected repetition and the peak it reached
class DetectedRep {
  const DetectedRep({
    required this.index,
    required this.startMs,
    required this.endMs,
    required this.peakValue,
    required this.baselineValue,
    required this.returnedToRest,
  });

  final int index;

  /// When the movement started, not when the detector fired
  final int startMs;

  /// When the arm returned to rest, or when the detector stopped waiting
  final int endMs;

  final double peakValue;
  final double baselineValue;

  /// Whether the angle actually came back down to the resting band
  final bool returnedToRest;

  Duration get duration => Duration(milliseconds: endMs - startMs);
}

/// Detecting repetitions online from the primary angle, using frame timestamps
class RepDetector {
  RepDetector({
    required this.targetDeg,
    this.riseFraction = 0.55,
    this.fallFraction = 0.30,
    this.restFraction = 0.15,
    this.minDurationSeconds = 0.8,
    this.maxDurationSeconds = 20.0,
    this.settleSeconds = 0.7,
    this.baselineSeconds = 0.5,
  });

  /// Prescribed excursion for the exercise, degrees
  final double targetDeg;

  /// Fraction of the excursion that starts a repetition
  final double riseFraction;

  /// Fraction below which the descent counts as under way
  final double fallFraction;

  /// Fraction of the excursion that bounds the resting band
  final double restFraction;

  final double minDurationSeconds;

  /// Force-emit after this long
  final double maxDurationSeconds;

  /// How long to wait during the descent for a return to rest before emitting
  final double settleSeconds;

  /// The resting pose is learned from this much of the start of the session
  final double baselineSeconds;

  double? _baseline;
  int? _firstMs;
  int _lastMs = 0;
  bool _inRep = false;
  int _repStartMs = 0;

  /// Last time the angle was inside the resting band while not in a repetition
  int? _restingMs;

  /// Last time the angle was below [fallFraction]; fallback start point
  int? _lowMs;

  /// When the descent first crossed [fallFraction] during this repetition
  int? _fallCrossedMs;

  /// Set when a repetition ended with the arm still up, so holding it up
  /// does not start a new repetition
  bool _awaitingRest = false;

  double _peak = double.negativeInfinity;
  int _completed = 0;

  int get completedReps => _completed;
  bool get inRepetition => _inRep;
  double? get baseline => _baseline;

  /// Reporting whether the resting baseline is still being learned
  bool get calibrating {
    final first = _firstMs;
    if (first == null) return true;
    return (_lastMs - first) < (baselineSeconds * 1000).round();
  }

  /// Clearing all state so the same detector can start a new session
  void reset() {
    _baseline = null;
    _firstMs = null;
    _lastMs = 0;
    _inRep = false;
    _repStartMs = 0;
    _restingMs = null;
    _lowMs = null;
    _fallCrossedMs = null;
    _awaitingRest = false;
    _peak = double.negativeInfinity;
    _completed = 0;
  }

  /// Feeding one frame's angle and timestamp, returning a repetition when one completes
  DetectedRep? update(double value, int timestampMs) {
    _firstMs ??= timestampMs;
    _lastMs = timestampMs;

    // Baseline is the minimum over the first half-second, robust to an early lift
    if (_baseline == null || calibrating) {
      _baseline = _baseline == null ? value : (value < _baseline! ? value : _baseline!);
      return null;
    }

    final excursion = (targetDeg - _baseline!).abs();
    if (excursion < 1e-6) return null;

    final rise = _baseline! + riseFraction * excursion;
    final fall = _baseline! + fallFraction * excursion;
    final rest = _baseline! + restFraction * excursion;

    if (!_inRep) {
      if (value <= fall) {
        // Release the guard below [fallFraction], since some patients rest above the baseline
        _awaitingRest = false;
        _lowMs = timestampMs;
      }
      if (value <= rest) _restingMs = timestampMs;
      if (!_awaitingRest && value >= rise) {
        _inRep = true;
        // The movement began when the angle last left the low band
        _repStartMs = _restingMs ?? _lowMs ?? timestampMs;
        _restingMs = null;
        _lowMs = null;
        _fallCrossedMs = null;
        _peak = value;
      }
      return null;
    }

    if (value > _peak) _peak = value;

    final elapsed = timestampMs - _repStartMs;
    final longEnough = elapsed >= (minDurationSeconds * 1000).round();
    final tooLong = elapsed >= (maxDurationSeconds * 1000).round();

    if (value <= rest) {
      // Clean return to rest
      if (longEnough) return _emit(timestampMs, returnedToRest: true);
    } else if (value <= fall) {
      _fallCrossedMs ??= timestampMs;
      // Descent stalled above rest: emit once settled so `incomplete_return` can grade it
      final settled =
          (timestampMs - _fallCrossedMs!) >= (settleSeconds * 1000).round();
      if (longEnough && settled) {
        return _emit(timestampMs, returnedToRest: false);
      }
    } else {
      // Went back up: one repetition with a wobble
      _fallCrossedMs = null;
    }

    if (tooLong) return _emit(timestampMs, returnedToRest: value <= rest);
    return null;
  }

  DetectedRep? _emit(int timestampMs, {required bool returnedToRest}) {
    final rep = DetectedRep(
      index: _completed,
      startMs: _repStartMs,
      endMs: timestampMs,
      peakValue: _peak,
      baselineValue: _baseline!,
      returnedToRest: returnedToRest,
    );
    _inRep = false;
    _peak = double.negativeInfinity;
    _fallCrossedMs = null;
    _restingMs = null;
    _lowMs = null;
    _awaitingRest = !returnedToRest;
    _completed += 1;
    return rep;
  }
}

/// Measuring the frame rate a window of frames was actually captured at
double effectiveFps(
  List<int> timestampsMs, {
  double fallback = 30.0,
  double minPlausible = 5.0,
  double maxPlausible = 120.0,
}) {
  if (timestampsMs.length < 2) return fallback;
  final span = timestampsMs.last - timestampsMs.first;
  if (span <= 0) return fallback;
  final rate = (timestampsMs.length - 1) * 1000.0 / span;
  if (rate < minPlausible || rate > maxPlausible) return fallback;
  return rate;
}
