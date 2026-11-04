import 'dart:math' as math;

import '../analysis/live_angles.dart'
    show BodySide, V3, angleBetween, projectOntoPlane, signedAngle;
import 'pose.dart';
import 'signal.dart';

/// Elevation band in which the plane of elevation is defined
const double planeOfElevationMinDeg = 40.0;
const double planeOfElevationMaxDeg = 140.0;

/// Acromion rise per degree of elevation above onset, torso lengths.
/// `SCAPULOHUMERAL_RISE_PER_DEG`, fitted on IntelliRehabDS correct reps
const double scapulohumeralRisePerDeg = 0.00071;
const double scapulohumeralOnsetDeg = 30.0;

/// Channel names in declaration order. `features.angles.ANGLE_CHANNELS`
const List<String> angleChannels = [
  'shoulder_elevation',
  'shoulder_flexion',
  'shoulder_abduction',
  'plane_of_elevation',
  'shoulder_rotation',
  'elbow_flexion',
  'trunk_forward_flexion',
  'trunk_lateral_tilt',
  'trunk_rotation',
  'shoulder_height_working',
  'shoulder_height_other',
  'hand_forward',
  'hand_up',
  'hand_lateral',
];

double sideSign(BodySide side) => side == BodySide.left ? 1.0 : -1.0;

/// Per-frame clinical angles for one working limb. `features.angles.AngleSeries`
class AngleSeries {
  AngleSeries({required this.side, required this.fps, required this.channels});

  final BodySide side;
  final double fps;

  /// Name -> `(T,)` values, every name in [angleChannels]
  final Map<String, List<double>> channels;

  int get nFrames => channels['shoulder_elevation']!.length;

  List<double> operator [](String name) {
    final values = channels[name];
    if (values == null) throw ArgumentError('$name is not an angle channel');
    return values;
  }

  /// Lateral lean away from the working arm (+ = compensating)
  List<double> trunkLeanContralateral() {
    final s = -sideSign(side);
    return [for (final v in this['trunk_lateral_tilt']) s * v];
  }

  /// Girdle elevation in excess of the obligatory scapular contribution
  List<double> shoulderHike({int baselineFrames = 5, bool scapularCorrection = true}) {
    final height = this['shoulder_height_working'];
    final n = math.max(1, math.min(baselineFrames, nFrames));
    final baseline = median(height.sublist(0, n));
    final elevation = this['shoulder_elevation'];
    return [
      for (var t = 0; t < height.length; t++)
        (height[t] - baseline) -
            (scapularCorrection
                ? scapulohumeralRisePerDeg *
                    math.max(0.0, elevation[t] - scapulohumeralOnsetDeg)
                : 0.0),
    ];
  }

  /// Absolute departure from a prescribed plane of elevation, gated on elevation
  List<double> planeDeviationFrom(double targetPlaneDeg) {
    final plane = this['plane_of_elevation'];
    final elevation = this['shoulder_elevation'];
    return [
      for (var t = 0; t < plane.length; t++)
        (elevation[t] >= planeOfElevationMinDeg && elevation[t] <= planeOfElevationMaxDeg)
            // Dart's % on doubles is non-negative for a positive divisor, as Python's is
            ? ((plane[t] - targetPlaneDeg + 180.0) % 360.0 - 180.0).abs()
            : 0.0,
    ];
  }

  AngleSeries slice(int start, int stop) => AngleSeries(
        side: side,
        fps: fps,
        channels: {
          for (final e in channels.entries) e.key: e.value.sublist(start, stop),
        },
      );
}

/// The full clinical angle set for [side]. `features.angles.compute_angles`
AngleSeries computeAngles(PoseSequence seq, BodySide side) {
  final n = seq.nFrames;
  final sign = sideSign(side);
  final out = {for (final name in angleChannels) name: List<double>.filled(n, 0)};
  final scale = math.max(seq.torsoLength(), 1e-3);
  const up = V3.up;
  final shoulderIdx = Joint.shoulder(side);
  final otherShoulderIdx = Joint.shoulder(otherSide(side));

  for (var t = 0; t < n; t++) {
    final pelvis = seq.at(t, Joint.pelvis);
    final neck = seq.at(t, Joint.neck);
    final lHip = seq.at(t, Joint.lHip);
    final rHip = seq.at(t, Joint.rHip);

    // BodyFrame
    final longitudinal = (neck - pelvis).unit;
    final lateral = projectOntoPlane(lHip - rHip, longitudinal).unit;
    final anterior = lateral.cross(longitudinal).unit;
    final down = -longitudinal;

    final shoulder = seq.at(t, shoulderIdx);
    final elbow = seq.at(t, Joint.elbow(side));
    final wrist = seq.at(t, Joint.wrist(side));
    final hand = seq.at(t, Joint.hand(side));
    final humerus = elbow - shoulder;
    final forearm = wrist - elbow;

    out['shoulder_elevation']![t] = angleBetween(humerus, down);
    out['shoulder_flexion']![t] =
        -signedAngle(down, projectOntoPlane(humerus, lateral), lateral);
    out['shoulder_abduction']![t] =
        sign * signedAngle(down, projectOntoPlane(humerus, anterior), anterior);
    out['plane_of_elevation']![t] =
        sign * signedAngle(anterior, projectOntoPlane(humerus, longitudinal), longitudinal);

    final humAxis = humerus.unit;
    final forearmPerp = projectOntoPlane(forearm, humAxis);
    final reference = projectOntoPlane(anterior, humAxis).unit;
    out['shoulder_rotation']![t] = -sign * signedAngle(reference, forearmPerp, humAxis);

    out['elbow_flexion']![t] = 180.0 - angleBetween(shoulder - elbow, wrist - elbow);

    // Trunk against gravity, in a pelvis-attached frame
    final pelvisLat = projectOntoPlane(lHip - rHip, up).unit;
    final pelvisAnt = pelvisLat.cross(up).unit;
    out['trunk_forward_flexion']![t] = signedAngle(
      projectOntoPlane(up, pelvisLat),
      projectOntoPlane(longitudinal, pelvisLat),
      pelvisLat,
    );
    out['trunk_lateral_tilt']![t] = -signedAngle(
      projectOntoPlane(up, pelvisAnt),
      projectOntoPlane(longitudinal, pelvisAnt),
      pelvisAnt,
    );

    final shoulderLine = seq.at(t, Joint.lShoulder) - seq.at(t, Joint.rShoulder);
    out['trunk_rotation']![t] = signedAngle(
      projectOntoPlane(lHip - rHip, up),
      projectOntoPlane(shoulderLine, up),
      up,
    );

    // Girdle heights and end-effector in body-frame torso lengths
    double local(V3 p, V3 axis) => (p - pelvis).dot(axis) / scale;
    out['shoulder_height_working']![t] = local(shoulder, longitudinal);
    out['shoulder_height_other']![t] = local(seq.at(t, otherShoulderIdx), longitudinal);
    out['hand_forward']![t] = local(hand, anterior);
    out['hand_up']![t] = local(hand, longitudinal);
    out['hand_lateral']![t] = sign * local(hand, lateral);
  }

  // Stored as float32, as `compute_angles` does
  for (final values in out.values) {
    for (var t = 0; t < n; t++) {
      values[t] = f32(values[t]);
    }
  }
  return AngleSeries(side: side, fps: seq.fps, channels: out);
}
