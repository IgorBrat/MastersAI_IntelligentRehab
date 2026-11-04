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
  static const int lWrist = 15, rWrist = 16;
  static const int lHip = 23, rHip = 24;
}

/// A 3-vector, in the canonical frame unless said otherwise
class V3 {
  const V3(this.x, this.y, this.z);

  final double x;
  final double y;
  final double z;

  static const V3 zero = V3(0, 0, 0);

  /// World 'up'. `gravity_up` in `features/geometry.py`
  static const V3 up = V3(0, 1, 0);

  V3 operator +(V3 o) => V3(x + o.x, y + o.y, z + o.z);
  V3 operator -(V3 o) => V3(x - o.x, y - o.y, z - o.z);
  V3 operator *(double k) => V3(x * k, y * k, z * k);
  V3 operator -() => V3(-x, -y, -z);

  double dot(V3 o) => x * o.x + y * o.y + z * o.z;

  V3 cross(V3 o) => V3(
        y * o.z - z * o.y,
        z * o.x - x * o.z,
        x * o.y - y * o.x,
      );

  double get length => math.sqrt(x * x + y * y + z * z);

  /// Matches `geometry.unit`, including its epsilon floor
  V3 get unit {
    final n = math.max(length, 1e-8);
    return V3(x / n, y / n, z / n);
  }

  V3 scaled(double k) => this * k;

  V3 mid(V3 o) => (this + o) * 0.5;
}

/// Removing the component of [v] along [normal]. `geometry.project_onto_plane`
V3 projectOntoPlane(V3 v, V3 normal) {
  final n = normal.unit;
  return v - n * v.dot(n);
}

/// Unsigned angle in degrees. `geometry.angle_between`
double angleBetween(V3 a, V3 b) =>
    _degrees(math.atan2(a.cross(b).length, a.dot(b)));

/// Signed angle from [a] to [b] about [normal], degrees. `geometry.signed_angle`
double signedAngle(V3 a, V3 b, V3 normal) {
  final n = normal.unit;
  return _degrees(math.atan2(a.cross(b).dot(n), a.dot(b)));
}

double _degrees(double radians) => radians * 180.0 / math.pi;

/// Per-frame anatomical basis of pelvis and trunk. Port of `geometry.BodyFrame`
class BodyFrame {
  BodyFrame._({
    required this.lateral,
    required this.longitudinal,
    required this.anterior,
  });

  final V3 lateral;
  final V3 longitudinal;
  final V3 anterior;

  factory BodyFrame.fromJoints({
    required V3 pelvis,
    required V3 neck,
    required V3 lHip,
    required V3 rHip,
  }) {
    final longitudinal = (neck - pelvis).unit;
    final lateral = projectOntoPlane(lHip - rHip, longitudinal).unit;
    final anterior = lateral.cross(longitudinal).unit;
    return BodyFrame._(
      lateral: lateral,
      longitudinal: longitudinal,
      anterior: anterior,
    );
  }

  /// The direction the arm hangs when at rest
  V3 get down => -longitudinal;
}

/// Live channel for the exercise, chosen from its primary channel
enum LiveChannel {
  shoulderFlexion,
  shoulderAbduction,
  shoulderElevation,
  elbowFlexion;

  static LiveChannel fromPrimary(String primaryChannel) {
    switch (primaryChannel) {
      case 'shoulder_flexion':
        return LiveChannel.shoulderFlexion;
      case 'shoulder_abduction':
        return LiveChannel.shoulderAbduction;
      case 'elbow_flexion':
        return LiveChannel.elbowFlexion;
      default:
        // Channels not derived live fall back to plain arm elevation
        return LiveChannel.shoulderElevation;
    }
  }

  /// Ukrainian phrasing for "start the movement"
  String get startCueUk => switch (this) {
        LiveChannel.shoulderFlexion => 'Піднімайте руку вперед',
        LiveChannel.shoulderAbduction => 'Піднімайте руку в сторону',
        LiveChannel.elbowFlexion => 'Згинайте руку в лікті',
        LiveChannel.shoulderElevation => 'Піднімайте руку',
      };

  /// Ukrainian phrasing for the compensation this exercise watches most closely
  String get compensationCueUk => switch (this) {
        LiveChannel.shoulderFlexion => 'Тримайте спину прямо, не нахиляйтеся вперед',
        LiveChannel.shoulderAbduction => 'Тулуб рівно, не нахиляйтеся в бік',
        LiveChannel.elbowFlexion => 'Тримайте спину прямо',
        LiveChannel.shoulderElevation => 'Тулуб рівно',
      };
}

/// Unfiltered per-frame angles for live cues only; scores come from `scoring/`
class LiveAngles {
  const LiveAngles({
    required this.channel,
    required this.primaryDeg,
    required this.compensationDeg,
    this.elevationDeg = 0.0,
    this.girdleHeightTl = 0.0,
  });

  final LiveChannel channel;

  /// Excursion the repetition is counted on, degrees
  final double primaryDeg;

  /// Trunk lean magnitude, degrees: forward for sagittal, lateral for frontal
  final double compensationDeg;

  /// Humeral elevation from the hanging position, degrees
  final double elevationDeg;

  /// Shoulder height above the pelvis, torso lengths. `shoulder_height_working`
  final double girdleHeightTl;

  /// Shoulder hike above [baselineTl] minus the obligatory scapular rise.
  /// `AngleSeries.shoulder_hike`
  double girdleHikeTl(double baselineTl) =>
      girdleHeightTl -
      baselineTl -
      scapulohumeralRisePerDeg * math.max(0.0, elevationDeg - scapulohumeralOnsetDeg);

  /// `features.angles.SCAPULOHUMERAL_RISE_PER_DEG` and its onset
  static const double scapulohumeralRisePerDeg = 0.00071;
  static const double scapulohumeralOnsetDeg = 30.0;

  /// Computes the angles from one frame's world landmarks
  static LiveAngles fromFrame(
    PoseFrame frame,
    BodySide side, {
    LiveChannel channel = LiveChannel.shoulderAbduction,
  }) {
    // MediaPipe y is down, z away from camera; remap to canonical (x, -y, -z)
    V3 at(int i) {
      final (x, y, z) = frame.point(i);
      return V3(x, -y, -z);
    }

    final lHip = at(Mp.lHip);
    final rHip = at(Mp.rHip);
    final lShoulder = at(Mp.lShoulder);
    final rShoulder = at(Mp.rShoulder);

    // Same as `landmarks_to_canonical`
    final pelvis = lHip.mid(rHip);
    final neck = lShoulder.mid(rShoulder);

    final body = BodyFrame.fromJoints(
      pelvis: pelvis,
      neck: neck,
      lHip: lHip,
      rHip: rHip,
    );

    final isLeft = side == BodySide.left;
    final shoulder = isLeft ? lShoulder : rShoulder;
    final elbow = at(isLeft ? Mp.lElbow : Mp.rElbow);
    final wrist = at(isLeft ? Mp.lWrist : Mp.rWrist);
    final humerus = elbow - shoulder;
    // `_side_sign` in features/angles.py
    final sign = isLeft ? 1.0 : -1.0;

    final primary = switch (channel) {
      LiveChannel.shoulderFlexion =>
        -signedAngle(body.down, projectOntoPlane(humerus, body.lateral), body.lateral),
      LiveChannel.shoulderAbduction => sign *
          signedAngle(
              body.down, projectOntoPlane(humerus, body.anterior), body.anterior),
      LiveChannel.shoulderElevation => angleBetween(humerus, body.down),
      // 180 - interior angle, so 0 is a straight arm
      LiveChannel.elbowFlexion =>
        180.0 - angleBetween(shoulder - elbow, wrist - elbow),
    };

    // Trunk lean is measured against gravity in a pelvis frame, not the trunk frame
    final pelvisLateral = projectOntoPlane(lHip - rHip, V3.up).unit;
    final pelvisAnterior = pelvisLateral.cross(V3.up).unit;
    final trunkAxis = body.longitudinal;

    final trunkForward = signedAngle(
      projectOntoPlane(V3.up, pelvisLateral),
      projectOntoPlane(trunkAxis, pelvisLateral),
      pelvisLateral,
    );
    final trunkTilt = -signedAngle(
      projectOntoPlane(V3.up, pelvisAnterior),
      projectOntoPlane(trunkAxis, pelvisAnterior),
      pelvisAnterior,
    );

    final compensation = switch (channel) {
      LiveChannel.shoulderFlexion => trunkForward.abs(),
      LiveChannel.elbowFlexion => trunkForward.abs(),
      LiveChannel.shoulderAbduction => trunkTilt.abs(),
      LiveChannel.shoulderElevation => trunkTilt.abs(),
    };

    final torso = (neck - pelvis).length;
    return LiveAngles(
      channel: channel,
      primaryDeg: primary,
      compensationDeg: compensation,
      elevationDeg: angleBetween(humerus, body.down),
      girdleHeightTl: torso > 1e-6 ? (shoulder - pelvis).dot(body.longitudinal) / torso : 0.0,
    );
  }
}

/// Causal exponential smoothing so the gauge and cues do not flicker
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
