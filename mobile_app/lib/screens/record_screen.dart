import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Uint8List;
import 'package:permission_handler/permission_handler.dart';

import '../analysis/cues.dart';
import '../analysis/live_angles.dart';
import '../analysis/rep_detector.dart';
import '../api/client.dart';
import '../config.dart';
import '../pose/pose_stream.dart';
import '../storage/session_record.dart';
import '../widgets/skeleton_overlay.dart';
import 'summary_screen.dart';

/// The recording screen: live skeleton, live angle, rep counter, per-rep score
class RecordScreen extends StatefulWidget {
  const RecordScreen({
    super.key,
    required this.baseUrl,
    required this.exercise,
    required this.side,
    this.romTarget,
    this.onSessionComplete,
  });

  final String baseUrl;
  final ExerciseInfo exercise;
  final BodySide side;

  /// The coach's current target, when one has been applied
  final double? romTarget;

  final Future<void> Function(SessionRecord)? onSessionComplete;

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  /// Above this instantaneous lean the cue shows. This is a **UI heuristic**, not the
  /// scored criterion: the engine grades the 90th percentile of lean over gated frames
  /// against a calibrated band (~4.4° mild), which is not comparable to a
  static const double _leanCueDeg = 12.0;

  late final RehabApi _api = RehabApi(widget.baseUrl);
  /// The target actually in force for this patient
  double get _target => widget.romTarget ?? widget.exercise.targetDeg;

  late final RepDetector _detector = RepDetector(targetDeg: _target);
  final Smoother _abduction = Smoother();
  final Smoother _lean = Smoother();
  final DirectionTracker _direction = DirectionTracker();

  StreamSubscription<PoseEvent>? _subscription;

  /// Latest decoded preview frame
  ui.Image? _preview;
  bool _decoding = false;

  /// Frames of the repetition currently in progress, for the per-rep POST
  final List<PoseFrame> _repWindow = [];

  /// Every frame of the session, for the final POST
  final List<PoseFrame> _sessionFrames = [];

  PoseFrame? _latest;
  bool _detected = false;
  String? _error;
  bool _scoring = false;
  bool _finishing = false;

  final List<Assessment> _repScores = [];

  @override
  void initState() {
    super.initState();
    _begin();
  }

  @override
  void dispose() {
    _subscription?.cancel();
    _preview?.dispose();
    _api.close();
    super.dispose();
  }

  Future<void> _begin() async {
    final granted = await Permission.camera.request();
    if (!granted.isGranted) {
      setState(() => _error = 'Немає доступу до камери');
      return;
    }
    await PoseStream.configure(frontCamera: true);
    _subscription = PoseStream.events().listen(
      _onEvent,
      onError: (Object e) => setState(() => _error = '$e'),
    );
  }

  void _onEvent(PoseEvent event) {
    if (event.isPreview) {
      _decodePreview(event.previewJpeg!);
      return;
    }
    _onFrame(event.frame);
  }

  /// Decoding a preview JPEG, dropping frames that arrive while one is in flight
  Future<void> _decodePreview(Uint8List jpeg) async {
    if (_decoding) return;
    _decoding = true;
    try {
      final decoded = await decodeImageFromList(jpeg);
      if (!mounted) {
        decoded.dispose();
        return;
      }
      setState(() {
        _preview?.dispose();
        _preview = decoded;
      });
    } catch (_) {
      // A corrupt preview frame is not worth interrupting the session for.
    } finally {
      _decoding = false;
    }
  }

  void _onFrame(PoseFrame? frame) {
    if (frame == null) {
      if (mounted) setState(() => _detected = false);
      return;
    }

    final angles = LiveAngles.fromFrame(frame, widget.side);
    final abduction = _abduction.add(angles.abductionDeg);
    _lean.add(angles.trunkLeanDeg);
    // Direction is taken from the SMOOTHED angle, not the raw one
    _direction.add(abduction);

    _sessionFrames.add(frame);
    _repWindow.add(frame);
    if (_repWindow.length > Config.maxFramesPerRep) {
      // The detector never fired - the patient is not reaching the rise threshold.
      // Drop the oldest frame rather than growing without bound.
      _repWindow.removeAt(0);
    }

    final rep = _detector.update(abduction, frame.timestampMs);
    if (mounted) {
      setState(() {
        _latest = frame;
        _detected = true;
      });
    }
    if (rep != null) _scoreRepetition();
  }

  Future<void> _scoreRepetition() async {
    final window = List<PoseFrame>.from(_repWindow);
    _repWindow.clear();
    if (window.length < 10) return;

    setState(() => _scoring = true);
    try {
      final assessment = await _api.assessRepetition(
        exerciseId: widget.exercise.id,
        side: widget.side,
        frames: window,
      );
      if (mounted) setState(() => _repScores.add(assessment));
    } on ApiException catch (e) {
      // A failed rep score must not abort the session - the patient keeps going and
      // the final session POST still covers every frame.
      if (mounted) setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _scoring = false);
    }
  }

  Future<void> _finish() async {
    setState(() => _finishing = true);
    await _subscription?.cancel();
    _subscription = null;
    try {
      final session = await _api.assessSession(
        exerciseId: widget.exercise.id,
        side: widget.side,
        frames: _sessionFrames,
      );

      await widget.onSessionComplete?.call(
        SessionRecord.fromAssessment(widget.exercise.id, session.json),
      );
      if (!mounted) return;
      Navigator.of(context).pushReplacement(MaterialPageRoute(
        builder: (_) => SummaryScreen(
          session: session,
          exercise: widget.exercise,
        ),
      ));
    } on ApiException catch (e) {
      if (mounted) {
        setState(() {
          _error = e.message;
          _finishing = false;
        });
      }
    }
  }

  /// The current repetition phase, or null when there is nothing to cue from
  RepPhase? get _phase {
    if (!_detected || _error != null) return null;
    return repPhase(
      angleDeg: _abduction.value ?? 0,
      target: _target,
      ceiling: widget.exercise.ceilingDeg,
      rising: _direction.rising,
      calibrating: _detector.calibrating,
      detectorInRep: _detector.inRepetition,
    );
  }

  String get _cue {
    if (_error != null) return _error!;
    if (!_detected) return 'Вас не видно - станьте перед камерою';

    // Trunk lean overrides the phase cue
    if ((_lean.value ?? 0) > _leanCueDeg) return 'Тулуб рівно, не нахиляйтеся';

    return switch (_phase) {
      RepPhase.calibrating => 'Тримайте руку внизу…',
      RepPhase.resting => 'Піднімайте руку в сторону',
      RepPhase.raising => 'Вище…',
      RepPhase.atTarget => 'Достатньо - затримайтесь',
      RepPhase.lowering => 'Повільно опускайте',
      RepPhase.overshoot => 'Занадто високо - опустіть до рівня плеча',
      null => '',
    };
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final abduction = _abduction.value ?? 0;
    final progress =
        (abduction / _target).clamp(0.0, 1.0).toDouble();
    final phase = _phase;
    final warning = (_lean.value ?? 0) > _leanCueDeg ||
        phase == RepPhase.overshoot ||
        !_detected ||
        _error != null;
    // Reaching the prescribed excursion is the one moment worth marking positively -
    // the patient cannot see the gauge and their own arm at the same time
    final onTarget = !warning && phase == RepPhase.atTarget;

    return Scaffold(
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand,
          children: [
            SkeletonOverlay(frame: _latest, image: _preview, mirror: true),
            Positioned(
              left: 16,
              right: 16,
              top: 8,
              child: Column(
                children: [
                  Card(
                    color: warning
                        ? scheme.errorContainer
                        : onTarget
                            ? scheme.primaryContainer
                            : scheme.surfaceContainerHigh,
                    child: Padding(
                      padding: const EdgeInsets.all(14),
                      child: Text(
                        _cue,
                        textAlign: TextAlign.center,
                        style: Theme.of(context).textTheme.titleMedium,
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
                  LinearProgressIndicator(value: progress, minHeight: 10),
                  const SizedBox(height: 6),
                  Text(
                    '${abduction.toStringAsFixed(0)}° / '
                    '${_target.toStringAsFixed(0)}°   ·   '
                    'повторень: ${_detector.completedReps}'
                    '${_scoring ? '  ·  оцінка…' : ''}',
                  ),
                ],
              ),
            ),
            Positioned(
              left: 16,
              right: 16,
              bottom: 12,
              child: Column(
                children: [
                  if (_repScores.isNotEmpty) _lastRepCard(context),
                  const SizedBox(height: 10),
                  FilledButton(
                    onPressed: _finishing ? null : _finish,
                    child: Text(_finishing ? 'Обробка…' : 'Завершити'),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _lastRepCard(BuildContext context) {
    final last = _repScores.last;
    final errors = last.errors;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Повторення ${_repScores.length}: '
                'EQI ${last.eqi?.toStringAsFixed(0) ?? '-'}'),
            if (errors.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  errors.entries
                      .map((e) =>
                          widget.exercise.watchedErrors[e.key] ?? e.key)
                      .join('\n'),
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
          ],
        ),
      ),
    );
  }
}
