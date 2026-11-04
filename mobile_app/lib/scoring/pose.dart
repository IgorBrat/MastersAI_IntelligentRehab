import 'dart:math' as math;
import 'dart:typed_data';

import '../analysis/live_angles.dart' show BodySide, V3;
import 'signal.dart';

/// Canonical joint indices. Same order as `rehab_ml.schema.Joint`
class Joint {
  static const int pelvis = 0;
  static const int spineMid = 1;
  static const int neck = 2;
  static const int head = 3;
  static const int lShoulder = 4;
  static const int lElbow = 5;
  static const int lWrist = 6;
  static const int lHand = 7;
  static const int rShoulder = 8;
  static const int rElbow = 9;
  static const int rWrist = 10;
  static const int rHand = 11;
  static const int lHip = 12;
  static const int lKnee = 13;
  static const int lAnkle = 14;
  static const int rHip = 15;
  static const int rKnee = 16;
  static const int rAnkle = 17;

  static const int count = 18;

  static const List<String> names = [
    'PELVIS', 'SPINE_MID', 'NECK', 'HEAD',
    'L_SHOULDER', 'L_ELBOW', 'L_WRIST', 'L_HAND',
    'R_SHOULDER', 'R_ELBOW', 'R_WRIST', 'R_HAND',
    'L_HIP', 'L_KNEE', 'L_ANKLE',
    'R_HIP', 'R_KNEE', 'R_ANKLE',
  ];

  /// Parent of each joint, `schema.SKELETON_PARENTS`
  static const List<int> parents = [
    0, 0, 1, 2, //
    2, 4, 5, 6, //
    2, 8, 9, 10, //
    0, 12, 13, //
    0, 15, 16,
  ];

  static int shoulder(BodySide side) => side == BodySide.left ? lShoulder : rShoulder;
  static int elbow(BodySide side) => side == BodySide.left ? lElbow : rElbow;
  static int wrist(BodySide side) => side == BodySide.left ? lWrist : rWrist;
  static int hand(BodySide side) => side == BodySide.left ? lHand : rHand;
}

BodySide otherSide(BodySide side) =>
    side == BodySide.left ? BodySide.right : BodySide.left;

/// A time-ordered sequence of canonical skeletons. `schema.PoseSequence`
/// Stored as float32, as numpy stores it, so rounding matches through the filters
class PoseSequence {
  PoseSequence({
    required this.joints,
    required this.confidence,
    required this.fps,
  })  : nFrames = confidence.length ~/ Joint.count,
        assert(joints.length == confidence.length * 3);

  /// `(T * 18 * 3)` flattened frame-major positions
  final Float32List joints;

  /// `(T * 18)` per-joint confidence
  final Float32List confidence;

  final double fps;
  final int nFrames;

  V3 at(int t, int joint) {
    final o = (t * Joint.count + joint) * 3;
    return V3(joints[o], joints[o + 1], joints[o + 2]);
  }

  double conf(int t, int joint) => confidence[t * Joint.count + joint];

  /// `(T,)` path of one joint
  List<V3> trajectory(int joint) => [for (var t = 0; t < nFrames; t++) at(t, joint)];

  /// Median pelvis-to-neck distance; the subject's scale unit
  double torsoLength() =>
      median([for (var t = 0; t < nFrames; t++) (at(t, Joint.neck) - at(t, Joint.pelvis)).length]);

  /// Joints whose median confidence is below [threshold]
  List<int> lowConfidenceJoints({double threshold = 0.5}) => [
        for (var j = 0; j < Joint.count; j++)
          if (median([for (var t = 0; t < nFrames; t++) conf(t, j)]) < threshold) j,
      ];

  PoseSequence withJoints(List<double> values) => PoseSequence(
        joints: Float32List.fromList(values),
        confidence: confidence,
        fps: fps,
      );
}

// MediaPipe landmark indices consumed by `landmarks_to_canonical`
const int _mpNose = 0;
const int _mpLShoulder = 11, _mpRShoulder = 12;
const int _mpLElbow = 13, _mpRElbow = 14;
const int _mpLWrist = 15, _mpRWrist = 16;
const int _mpLPinky = 17, _mpRPinky = 18;
const int _mpLIndex = 19, _mpRIndex = 20;
const int _mpLHip = 23, _mpRHip = 24;
const int _mpLKnee = 25, _mpRKnee = 26;
const int _mpLAnkle = 27, _mpRAnkle = 28;

const Map<int, int> _direct = {
  Joint.head: _mpNose,
  Joint.lShoulder: _mpLShoulder,
  Joint.rShoulder: _mpRShoulder,
  Joint.lElbow: _mpLElbow,
  Joint.rElbow: _mpRElbow,
  Joint.lWrist: _mpLWrist,
  Joint.rWrist: _mpRWrist,
  Joint.lHip: _mpLHip,
  Joint.rHip: _mpRHip,
  Joint.lKnee: _mpLKnee,
  Joint.rKnee: _mpRKnee,
  Joint.lAnkle: _mpLAnkle,
  Joint.rAnkle: _mpRAnkle,
};

/// One frame of 33 MediaPipe landmarks as the 18 canonical joints, written at [t].
/// `mediapipe_estimator.landmarks_to_canonical`: y and z flipped, midpoints synthesised
void landmarksToCanonical(
  List<double> world,
  List<double> visibility,
  Float32List joints,
  Float32List confidence,
  int t,
) {
  V3 pt(int i) => V3(world[i * 3], -world[i * 3 + 1], -world[i * 3 + 2]);
  void put(int joint, V3 p, double c) {
    final o = (t * Joint.count + joint) * 3;
    joints[o] = p.x;
    joints[o + 1] = p.y;
    joints[o + 2] = p.z;
    confidence[t * Joint.count + joint] = c;
  }

  for (var j = 0; j < Joint.count; j++) {
    put(j, V3.zero, 0);
  }
  _direct.forEach((joint, idx) => put(joint, pt(idx), visibility[idx]));

  final pelvis = (pt(_mpLHip) + pt(_mpRHip)) * 0.5;
  final neck = (pt(_mpLShoulder) + pt(_mpRShoulder)) * 0.5;
  final pelvisConf = math.min(visibility[_mpLHip], visibility[_mpRHip]);
  final neckConf = math.min(visibility[_mpLShoulder], visibility[_mpRShoulder]);
  put(Joint.pelvis, pelvis, pelvisConf);
  put(Joint.neck, neck, neckConf);
  put(Joint.spineMid, (pelvis + neck) * 0.5, math.min(pelvisConf, neckConf));
  put(Joint.lHand, (pt(_mpLIndex) + pt(_mpLPinky)) * 0.5,
      math.min(visibility[_mpLIndex], visibility[_mpLPinky]));
  put(Joint.rHand, (pt(_mpRIndex) + pt(_mpRPinky)) * 0.5,
      math.min(visibility[_mpRIndex], visibility[_mpRPinky]));
}

/// A posted window of `(T, 33, 3)` landmarks and `(T, 33)` visibility as a sequence
PoseSequence sequenceFromLandmarks(
  List<List<double>> worldPerFrame,
  List<List<double>>? visibilityPerFrame,
  double fps,
) {
  final n = worldPerFrame.length;
  final joints = Float32List(n * Joint.count * 3);
  final confidence = Float32List(n * Joint.count);
  final ones = List<double>.filled(33, 1.0);
  for (var t = 0; t < n; t++) {
    landmarksToCanonical(
      worldPerFrame[t],
      visibilityPerFrame?[t] ?? ones,
      joints,
      confidence,
      t,
    );
  }
  return PoseSequence(joints: joints, confidence: confidence, fps: fps);
}

/// Half-open `[start, stop)` ranges of consecutive true values
List<(int, int)> _runs(List<bool> mask) {
  final out = <(int, int)>[];
  int? start;
  for (var i = 0; i < mask.length; i++) {
    if (mask[i] && start == null) start = i;
    if (!mask[i] && start != null) {
      out.add((start, i));
      start = null;
    }
  }
  if (start != null) out.add((start, mask.length));
  return out;
}

/// Linearly interpolating low-confidence or non-finite joints. `normalize.fill_gaps`
PoseSequence fillGaps(
  PoseSequence seq, {
  double confidenceThreshold = 0.3,
  double maxGapSeconds = 0.5,
}) {
  final n = seq.nFrames;
  final joints = List<double>.from(seq.joints);
  final conf = Float32List.fromList(seq.confidence);
  final maxGap = math.max(1, pyRound(maxGapSeconds * seq.fps));
  final time = [for (var t = 0; t < n; t++) t.toDouble()];

  for (var j = 0; j < Joint.count; j++) {
    final bad = <bool>[
      for (var t = 0; t < n; t++)
        conf[t * Joint.count + j] < confidenceThreshold ||
            !joints[(t * Joint.count + j) * 3].isFinite ||
            !joints[(t * Joint.count + j) * 3 + 1].isFinite ||
            !joints[(t * Joint.count + j) * 3 + 2].isFinite,
    ];
    if (!bad.contains(true)) continue;
    final goodIdx = [for (var t = 0; t < n; t++) if (!bad[t]) t];
    if (goodIdx.length < 2) {
      // Nothing to interpolate from: hold the joint at the pelvis
      for (var t = 0; t < n; t++) {
        for (var a = 0; a < 3; a++) {
          joints[(t * Joint.count + j) * 3 + a] = joints[(t * Joint.count + Joint.pelvis) * 3 + a];
        }
        conf[t * Joint.count + j] = 0;
      }
      continue;
    }
    final xp = [for (final t in goodIdx) time[t]];
    for (var a = 0; a < 3; a++) {
      final fp = [for (final t in goodIdx) joints[(t * Joint.count + j) * 3 + a]];
      for (var t = 0; t < n; t++) {
        if (bad[t]) joints[(t * Joint.count + j) * 3 + a] = interp(time[t], xp, fp);
      }
    }
    for (final (start, stop) in _runs(bad)) {
      if (stop - start > maxGap) {
        for (var t = start; t < stop; t++) {
          conf[t * Joint.count + j] = 0;
        }
      }
    }
  }
  return PoseSequence(joints: Float32List.fromList(joints), confidence: conf, fps: seq.fps);
}

/// Low-passing every coordinate independently. `normalize.denoise`
PoseSequence denoise(PoseSequence seq, {double cutoffHz = 6.0}) {
  final n = seq.nFrames;
  if (n < 5) return seq;
  final out = List<double>.filled(seq.joints.length, 0);
  for (var c = 0; c < Joint.count * 3; c++) {
    final column = [for (var t = 0; t < n; t++) seq.joints[t * Joint.count * 3 + c].toDouble()];
    final smoothed = smoothSignal(column, seq.fps, cutoffHz: cutoffHz);
    for (var t = 0; t < n; t++) {
      out[t * Joint.count * 3 + c] = smoothed[t];
    }
  }
  return seq.withJoints(out);
}

/// Projecting joints so each bone keeps its median length. `normalize.enforce_bone_lengths`
PoseSequence enforceBoneLengths(PoseSequence seq) {
  final n = seq.nFrames;
  final joints = List<double>.from(seq.joints);
  V3 get(int t, int j) {
    final o = (t * Joint.count + j) * 3;
    return V3(joints[o], joints[o + 1], joints[o + 2]);
  }

  // Every parent has a lower index, so index order is a topological order
  for (var child = 0; child < Joint.count; child++) {
    final parent = Joint.parents[child];
    if (parent == child) continue;
    final lengths = [for (var t = 0; t < n; t++) (get(t, child) - get(t, parent)).length];
    final target = median(lengths);
    if (target <= 1e-6) continue;
    for (var t = 0; t < n; t++) {
      final p = get(t, parent);
      final vec = get(t, child) - p;
      final corrected = p + vec * (target / math.max(lengths[t], 1e-8));
      final o = (t * Joint.count + child) * 3;
      joints[o] = corrected.x;
      joints[o + 1] = corrected.y;
      joints[o + 2] = corrected.z;
    }
  }
  return seq.withJoints(joints);
}

/// Moving the pelvis to the origin and rescaling to torso lengths. `normalize.center_and_scale`
PoseSequence centerAndScale(PoseSequence seq) {
  final n = seq.nFrames;
  final scale = math.max(seq.torsoLength(), 1e-3);
  final out = List<double>.filled(seq.joints.length, 0);
  for (var t = 0; t < n; t++) {
    final pelvis = seq.at(t, Joint.pelvis);
    for (var j = 0; j < Joint.count; j++) {
      final o = (t * Joint.count + j) * 3;
      // Divided rather than multiplied by the reciprocal, as numpy does
      out[o] = (seq.joints[o] - pelvis.x) / scale;
      out[o + 1] = (seq.joints[o + 1] - pelvis.y) / scale;
      out[o + 2] = (seq.joints[o + 2] - pelvis.z) / scale;
    }
  }
  return seq.withJoints(out);
}

/// Fill gaps, denoise, rigidify, centre and scale. `normalize.preprocess`
PoseSequence preprocess(PoseSequence seq) {
  var out = fillGaps(seq);
  out = denoise(out);
  out = enforceBoneLengths(out);
  return centerAndScale(out);
}
