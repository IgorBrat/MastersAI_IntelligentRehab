/// Which phase of a repetition the patient is in, and therefore what to tell them
enum RepPhase {
  /// Still learning the resting pose
  calibrating,

  /// Arm down, nothing happening yet
  resting,

  /// Below target and still going up
  raising,

  /// At or past the prescribed excursion
  atTarget,

  /// Coming back down
  lowering,

  overshoot,
}

/// Deciding the phase from the current angle and the direction of travel
RepPhase repPhase({
  required double angleDeg,
  required double target,
  required double ceiling,
  required bool rising,
  required bool calibrating,
  required bool detectorInRep,
  double tolerance = 5.0,
}) {
  if (calibrating) return RepPhase.calibrating;
  if (angleDeg > ceiling) return RepPhase.overshoot;
  if (angleDeg >= target - tolerance) return RepPhase.atTarget;
  if (!detectorInRep) return RepPhase.resting;
  return rising ? RepPhase.raising : RepPhase.lowering;
}

/// Tracks whether the angle is rising, with a deadband so jitter does not flip it
class DirectionTracker {
  DirectionTracker({this.deadbandDeg = 0.8});

  final double deadbandDeg;
  double? _previous;
  bool _rising = true;

  bool get rising => _rising;

  void reset() {
    _previous = null;
    _rising = true;
  }

  bool add(double sample) {
    final previous = _previous;
    if (previous != null) {
      final delta = sample - previous;
      if (delta.abs() >= deadbandDeg) {
        _rising = delta > 0;
        _previous = sample;
      }
      // Inside the deadband the previous value is deliberately not updated, so a slow
      // drift still accumulates to a decision instead of being swallowed frame by frame
    } else {
      _previous = sample;
    }
    return _rising;
  }
}
