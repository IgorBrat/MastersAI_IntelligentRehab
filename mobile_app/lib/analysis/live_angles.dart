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

/// Angle from [a] to [b], signed by the right-hand rule about [normal]
///
/// `geometry.signed_angle`. Using atan2 of the cross and dot terms rather than
/// acos of the dot: acos loses the sign, and the sign is the whole point for
/// flexion and for trunk lean.
double signedAngle(V3 a, V3 b, V3 normal) {
  final n = normal.unit;
  return _degrees(math.atan2(a.cross(b).dot(n), a.dot(b)));
}

double _degrees(double radians) => radians * 180.0 / math.pi;

/// Per-frame anatomical basis attached to the pelvis and trunk
///
/// A Dart port of `features/geometry.py:BodyFrame`. Angles measured in this
/// frame are the same whether the patient sits square to the phone or 20 deg off
/// axis, which is the property the whole measurement design rests on.
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

/// Which pair of numbers to derive for the exercise in progress
///
/// The primary channel comes from the service's `/exercises`, so this follows
/// the protocol rather than restating it. Before this existed the screen always
/// computed shoulder abduction, whatever exercise was running -- correct only
/// for one of the six.
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
        // hand_up, hand_forward and shoulder_rotation are not derived on device.
        // Elevation is the honest fallback: it is the excursion the patient is
        // being asked for in every raise, just not the plane-resolved one.
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

/// The two scalars computed on-device, for the exercise in progress
///
/// #### Why anything is computed in Dart at all
/// Intra-repetition cueing has a hard latency budget: a "straighten up" prompt
/// is useless if it arrives 200 ms after the lean. A network round trip per
/// frame cannot meet that, so these two numbers are derived locally.
///
/// #### Why only two
/// The full pipeline is 14 angle channels, Butterworth filtering, SPARC over an
/// FFT, and eleven graded criteria. These two drive *cues* only. Every **score**
/// comes from the service, which runs the real pipeline. If the two ever
/// disagree, the service is right.
///
/// #### How close they are to the service's numbers
/// The geometry is a faithful port: same anatomical frame, same signed-angle
/// convention, same gravity reference. What is missing is everything
/// `pose.normalize.preprocess` does first -- gap filling, the zero-phase
/// Butterworth, rigid bone lengths. Those cannot run causally, so the on-device
/// value is the unfiltered one and differs by the noise the filter would have
/// removed.
class LiveAngles {
  const LiveAngles({
    required this.channel,
    required this.primaryDeg,
    required this.compensationDeg,
  });

  final LiveChannel channel;

  /// The excursion the repetition is counted on, in the channel's own convention
  final double primaryDeg;

  /// The compensation this exercise watches, in degrees away from upright
  ///
  /// Trunk FORWARD flexion for the sagittal exercises, trunk LATERAL tilt for
  /// the frontal ones. Reported as a magnitude: the scored criteria are
  /// direction-aware, this is a cue and a lean either way is worth saying.
  final double compensationDeg;

  /// Deriving both from one frame's world landmarks
  static LiveAngles fromFrame(
    PoseFrame frame,
    BodySide side, {
    LiveChannel channel = LiveChannel.shoulderAbduction,
  }) {
    // MediaPipe world landmarks are metres with y pointing DOWN and z away from
    // the camera. `pose.mediapipe_estimator.landmarks_to_canonical` remaps them
    // to (x, -y, -z), and every formula below assumes it has been done.
    V3 at(int i) {
      final (x, y, z) = frame.point(i);
      return V3(x, -y, -z);
    }

    final lHip = at(Mp.lHip);
    final rHip = at(Mp.rHip);
    final lShoulder = at(Mp.lShoulder);
    final rShoulder = at(Mp.rShoulder);

    // Synthesised exactly as landmarks_to_canonical does
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

    // Trunk posture is measured against GRAVITY in a pelvis-attached frame
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

    return LiveAngles(
      channel: channel,
      primaryDeg: primary,
      compensationDeg: compensation,
    );
  }
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
