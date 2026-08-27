import 'dart:math' as math;

import '../pose/pose_stream.dart';

/// Body side. Matches `rehab_ml.schema.Side`
enum BodySide {
  left('left'),
  right('right');

  const BodySide(this.wire);
  final String wire;
}

/// MediaPipe landmark indices. Same numbering as `rehab_ml.pose.mediapipe_estimator`
class Mp {
  static const int lShoulder = 11, rShoulder = 12;
  static const int lElbow = 13, rElbow = 14;
  static const int lHip = 23, rHip = 24;
}

/// The two scalars computed on-device
class LiveAngles {
  const LiveAngles({required this.abductionDeg, required this.trunkLeanDeg});

  /// Shoulder abduction: the angle between the humerus and the trunk's downward axis, in
  /// the frontal plane. 0 = arm at the side, 90 = horizontal
  final double abductionDeg;

  /// Contralateral trunk lean, degrees away from vertical, positive when leaning *away*
  /// from the working arm - the dominant post-stroke substitution
  final double trunkLeanDeg;

  /// Deriving both from one frame's world landmarks
  static LiveAngles fromFrame(PoseFrame frame, BodySide side) {
    final shoulderIdx = side == BodySide.left ? Mp.lShoulder : Mp.rShoulder;
    final elbowIdx = side == BodySide.left ? Mp.lElbow : Mp.rElbow;

    final shoulder = frame.point(shoulderIdx);
    final elbow = frame.point(elbowIdx);
    final lHip = frame.point(Mp.lHip);
    final rHip = frame.point(Mp.rHip);
    final lSh = frame.point(Mp.lShoulder);
    final rSh = frame.point(Mp.rShoulder);

    final pelvis = _mid(lHip, rHip);
    final neck = _mid(lSh, rSh);

    // Trunk axis, pelvis -> neck. In MediaPipe's frame this points in -y.
    final trunk = _unit(_sub(neck, pelvis));

    // Abduction is measured against the trunk, not against gravity
    final humerus = _unit(_sub(elbow, shoulder));
    final down = (-trunk.$1, -trunk.$2, -trunk.$3);
    final abduction = _angleBetween(humerus, down);

    // Lateral lean: trunk tilt away from vertical in the frontal (x-y) plane
    final leanMagnitude =
        _degrees(math.atan2(trunk.$1, -trunk.$2));
    final lean = side == BodySide.left ? -leanMagnitude : leanMagnitude;

    return LiveAngles(
      abductionDeg: abduction,
      trunkLeanDeg: lean,
    );
  }

  static (double, double, double) _mid(
      (double, double, double) a, (double, double, double) b) =>
      ((a.$1 + b.$1) / 2, (a.$2 + b.$2) / 2, (a.$3 + b.$3) / 2);

  static (double, double, double) _sub(
      (double, double, double) a, (double, double, double) b) =>
      (a.$1 - b.$1, a.$2 - b.$2, a.$3 - b.$3);

  static (double, double, double) _unit((double, double, double) v) {
    final n = math.sqrt(v.$1 * v.$1 + v.$2 * v.$2 + v.$3 * v.$3);
    if (n < 1e-9) return (0, 0, 0);
    return (v.$1 / n, v.$2 / n, v.$3 / n);
  }

  static double _angleBetween(
      (double, double, double) a, (double, double, double) b) {
    final dot = (a.$1 * b.$1 + a.$2 * b.$2 + a.$3 * b.$3).clamp(-1.0, 1.0);
    return _degrees(math.acos(dot));
  }

  static double _degrees(double radians) => radians * 180.0 / math.pi;
}

/// Exponential smoothing, so the gauge and the cues do not flicker on jitter
class Smoother {
  Smoother({this.alpha = 0.35});

  final double alpha;
  double? _value;

  double add(double sample) {
    final previous = _value;
    final next = previous == null ? sample : alpha * sample + (1 - alpha) * previous;
    _value = next;
    return next;
  }

  double? get value => _value;

  void reset() => _value = null;
}
