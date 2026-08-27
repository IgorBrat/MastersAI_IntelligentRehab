import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../pose/pose_stream.dart';

/// The camera frame with the detected skeleton drawn on top
class SkeletonOverlay extends StatelessWidget {
  const SkeletonOverlay({
    super.key,
    required this.frame,
    required this.image,
    required this.mirror,
  });

  final PoseFrame? frame;

  /// Decoded preview image, or `null` before the first one arrives
  final ui.Image? image;

  /// The front camera is mirrored, so the patient sees themselves as in a mirror
  final bool mirror;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Transform.scale(
      scaleX: mirror ? -1.0 : 1.0,
      child: CustomPaint(
        painter: _ScenePainter(
          frame: frame,
          image: image,
          color: scheme.primary,
          background: scheme.surfaceContainerLowest,
        ),
        child: const SizedBox.expand(),
      ),
    );
  }
}

/// BlazePose bone pairs, upper body plus hips. Legs are omitted
const List<List<int>> _bones = [
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // trunk sides
  [23, 24], // hips
];

const List<int> _joints = [11, 12, 13, 14, 15, 16, 23, 24];

class _ScenePainter extends CustomPainter {
  _ScenePainter({
    required this.frame,
    required this.image,
    required this.color,
    required this.background,
  });

  final PoseFrame? frame;
  final ui.Image? image;
  final Color color;
  final Color background;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = background);

    final source = image;
    final aspectW = (source?.width ?? frame?.frameWidth ?? 0).toDouble();
    final aspectH = (source?.height ?? frame?.frameHeight ?? 0).toDouble();
    if (aspectW <= 0 || aspectH <= 0) return;

    final box = _fit(aspectW / aspectH, size);

    if (source != null) {
      canvas.drawImageRect(
        source,
        Rect.fromLTWH(0, 0, source.width.toDouble(), source.height.toDouble()),
        box,
        Paint()..filterQuality = FilterQuality.low,
      );
    }

    final current = frame;
    if (current == null || current.screen.length < 33 * 2) return;

    Offset at(int index) => Offset(
          box.left + current.screen[index * 2] * box.width,
          box.top + current.screen[index * 2 + 1] * box.height,
        );

    final bonePaint = Paint()
      ..color = color
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;
    final jointPaint = Paint()..color = color;

    for (final bone in _bones) {
      canvas.drawLine(at(bone[0]), at(bone[1]), bonePaint);
    }
    for (final index in _joints) {
      canvas.drawCircle(at(index), 6, jointPaint);
    }
  }

  /// The largest rect of the given aspect ratio that fits inside [size], centred
  Rect _fit(double aspect, Size size) {
    var width = size.width;
    var height = width / aspect;
    if (height > size.height) {
      height = size.height;
      width = height * aspect;
    }
    return Rect.fromLTWH(
      (size.width - width) / 2,
      (size.height - height) / 2,
      width,
      height,
    );
  }

  @override
  bool shouldRepaint(_ScenePainter old) =>
      old.frame != frame || old.image != image || old.color != color;
}
