import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/services.dart';

/// One frame of MediaPipe output, as it arrives from the Kotlin side
class PoseFrame {
  const PoseFrame({
    required this.world,
    required this.visibility,
    required this.screen,
    required this.timestampMs,
    required this.frameWidth,
    required this.frameHeight,
  });

  /// (33 * 3) flattened world landmarks, metres
  final Float32List world;

  /// (33,) per-landmark visibility in [0, 1]
  final Float32List visibility;

  /// (33 * 2) flattened normalised image coordinates, for drawing
  final Float32List screen;

  final int timestampMs;

  /// Size of the upright frame the landmarks were computed from
  final int frameWidth;
  final int frameHeight;

  static const int landmarkCount = 33;

  /// World landmark [i] as (x, y, z)
  (double, double, double) point(int i) =>
      (world[i * 3], world[i * 3 + 1], world[i * 3 + 2]);

  /// Nested (33, 3) form, which is what the service's wire format expects
  List<List<double>> toNestedWorld() => List.generate(
        landmarkCount,
        (i) => [world[i * 3], world[i * 3 + 1], world[i * 3 + 2]],
        growable: false,
      );

  List<double> toVisibilityList() => List<double>.from(visibility);
}

/// One item from the native stream: a pose, a no-detection marker, or a preview frame
class PoseEvent {
  const PoseEvent.pose(this.frame) : previewJpeg = null;
  const PoseEvent.noDetection()
      : frame = null,
        previewJpeg = null;
  const PoseEvent.preview(this.previewJpeg) : frame = null;

  /// Landmarks, or `null` for a no-detection marker or a preview frame
  final PoseFrame? frame;

  /// A downscaled JPEG of the same upright frame the landmarks came from
  final Uint8List? previewJpeg;

  bool get isPreview => previewJpeg != null;
}

/// The camera + pose pipeline running in Kotlin
class PoseStream {
  static const EventChannel _events =
      EventChannel('ua.lpnu.intelligent_rehab/pose_stream');
  static const MethodChannel _control =
      MethodChannel('ua.lpnu.intelligent_rehab/pose_control');

  /// Setting the lens and model before subscribing
  static Future<void> configure({
    bool frontCamera = true,
    String model = 'pose_landmarker_lite.task',
  }) =>
      _control.invokeMethod<void>('configure', {
        'frontCamera': frontCamera,
        'model': model,
      });

  /// Everything the native side emits: pose frames, no-detection markers, and preview
  /// JPEGs. One broadcast stream, because `receiveBroadcastStream` can only be listened to
  /// once per channel without re-invoking `onListen` natively
  static Stream<PoseEvent> events() {
    return _events.receiveBroadcastStream().map((dynamic event) {
      final map = (event as Map).cast<Object?, Object?>();
      final preview = map['preview'];
      if (preview != null) {
        return PoseEvent.preview(Uint8List.fromList((preview as List).cast<int>()));
      }
      if (map['detected'] != true) return const PoseEvent.noDetection();
      return PoseEvent.pose(PoseFrame(
        world: _asFloat32(map['world']),
        visibility: _asFloat32(map['visibility']),
        screen: _asFloat32(map['screen']),
        timestampMs: (map['timestampMs'] as num).toInt(),
        frameWidth: (map['frameWidth'] as num?)?.toInt() ?? 0,
        frameHeight: (map['frameHeight'] as num?)?.toInt() ?? 0,
      ));
    });
  }

  /// Pose frames only, or `null` where no person was detected
  static Stream<PoseFrame?> frames() => events()
      .where((e) => !e.isPreview)
      .map((e) => e.frame);

  /// Kotlin's `FloatArray` arrives as a `Float32List` under the standard codec, but a plain
  /// `List<double>` if the payload is ever rebuilt by hand
  static Float32List _asFloat32(Object? value) => value is Float32List
      ? value
      : Float32List.fromList((value as List).cast<num>().map((n) => n.toDouble()).toList());
}
