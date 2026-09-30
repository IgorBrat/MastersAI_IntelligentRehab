/// Which phase of a repetition the patient is in, and therefore what to tell them
enum RepPhase {
  /// Still learning the resting pose
  calibrating,

  /// Arm down, nothing happening yet
  resting,

  /// Below target and still going up
  raising,

  /// At or past the prescribed excursion, and the hold is still running
  atTarget,

  /// The hold is done -- this is the cue to START coming down
  lowerNow,

  /// Coming back down
  lowering,

  overshoot,
}

/// How close to [target] still counts as "at target"
///
/// Shared with the caller because the caller has to time the hold, and it can
/// only time it if it enters the band on exactly the same frame this function
/// does.
const double kTargetToleranceDeg = 5.0;

/// Deciding the phase from the current angle, the direction of travel and how
/// long the patient has been at the top
///
/// [secondsAtTarget] is measured from the frame the angle first entered the
/// target band, and [holdSeconds] is how long it should stay there. Once the
/// hold is satisfied the phase becomes [RepPhase.lowerNow] -- the INSTRUCTION to
/// come down, as opposed to [RepPhase.lowering], which is an observation that
/// the patient already is
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
      // Inside the deadband the previous value is deliberately not updated, so a slow
      // drift still accumulates to a decision instead of being swallowed frame by frame
    } else {
      _previous = sample;
    }
    return _rising;
  }
}
