/// Which phase of a repetition the patient is in, and so what to tell them
enum RepPhase {
  /// Still learning the resting pose
  calibrating,
  resting,

  /// Below target and still going up
  raising,

  /// At target, hold still running
  atTarget,

  /// Hold done; the cue to start coming down
  lowerNow,

  /// Descent already under way
  lowering,

  overshoot,
}

/// How close to the target still counts as "at target", degrees.
/// Shared so the caller's hold timer enters the band on the same frame
const double kTargetToleranceDeg = 5.0;

/// Deciding the phase from the angle, direction of travel and time at the top
RepPhase repPhase({
  required double angleDeg,
  required double target,
  required double ceiling,
  required bool rising,
  required bool calibrating,
  required bool detectorInRep,
  double secondsAtTarget = 0.0,
  double holdSeconds = 0.0,
  double tolerance = kTargetToleranceDeg,
}) {
  if (calibrating) return RepPhase.calibrating;
  if (angleDeg > ceiling) return RepPhase.overshoot;
  if (angleDeg >= target - tolerance) {
    // Checked before direction, so a small dip mid-hold still counts as holding
    return secondsAtTarget >= holdSeconds ? RepPhase.lowerNow : RepPhase.atTarget;
  }
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
      // Previous is kept inside the deadband so a slow drift still accumulates
    } else {
      _previous = sample;
    }
    return _rising;
  }
}
