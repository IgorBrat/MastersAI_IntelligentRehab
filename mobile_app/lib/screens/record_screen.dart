import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show Uint8List;
import 'package:permission_handler/permission_handler.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../analysis/cues.dart';
import '../analysis/live_angles.dart';
import '../analysis/rep_detector.dart';
import '../analysis/spoken_cues.dart';
import '../api/client.dart';
import '../config.dart';
import '../pose/pose_stream.dart';
import '../storage/local_store.dart';
import '../storage/session_record.dart';
import '../widgets/skeleton_overlay.dart';
import 'summary_screen.dart';

/// What the screen is doing. The framing check comes FIRST, deliberately
enum _Stage {
  /// Camera live, checking the framing. Nothing is being recorded yet
  preparing,

  /// Recording and scoring repetitions
  recording,

  /// Posting the set
  finishing,
}

/// The recording screen: framing check, live skeleton, live angle, rep counter
class RecordScreen extends StatefulWidget {
  const RecordScreen({
    super.key,
    required this.baseUrl,
    required this.exercise,
    required this.side,
    this.romTarget,
    this.reps,
    this.setIndex = 1,
    this.sets = 1,
    this.onSessionComplete,
    this.onLabel,
  });

  final String baseUrl;
  final ExerciseInfo exercise;
  final BodySide side;

  /// The coach's current target, when one has been applied
  final double? romTarget;

  /// Repetitions this set asks for. The set ends itself on reaching it
  final int? reps;

  /// Which set of how many, for the label only
  final int setIndex;
  final int sets;

  final Future<void> Function(SessionRecord)? onSessionComplete;

  /// Recording the supervisor's verdict, offered on the summary
  final Future<void> Function(SessionRecord, SupervisorLabel)? onLabel;

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {

  static const double _leanCueDeg = 12.0;

  /// Frames kept while preparing, for the readiness check
  static const int _readinessFrames = 60;

  /// How often the framing is re-checked while preparing
  static const Duration _readinessEvery = Duration(seconds: 2);

  late final RehabApi _api = RehabApi(widget.baseUrl);
  final SpokenCues _cues = SpokenCues();

  /// The target actually in force for this patient
  double get _target => widget.romTarget ?? widget.exercise.targetDeg;
  int get _repTarget => widget.reps ?? widget.exercise.defaultReps;

  late final RepDetector _detector = RepDetector(targetDeg: _target);
  /// The exercise's own primary channel, smoothed
  final Smoother _primary = Smoother();

  /// The compensation this exercise watches, smoothed
  final Smoother _compensation = Smoother();

  /// Which channels to derive on device, taken from the protocol
  late final LiveChannel _channel =
      LiveChannel.fromPrimary(widget.exercise.primaryChannel);
  final DirectionTracker _direction = DirectionTracker();

  int? _atTargetSinceMs;

  /// How long to ask the patient to hold at the top
  double get _holdCueSeconds =>
      widget.exercise.holdSeconds > 0 ? widget.exercise.holdSeconds : 2.0;

  StreamSubscription<PoseEvent>? _subscription;
  Timer? _readinessTimer;

  _Stage _stage = _Stage.preparing;

  /// Latest decoded preview frame
  ui.Image? _preview;
  bool _decoding = false;

  /// Frames of the repetition currently in progress, for the per-rep POST
  final List<PoseFrame> _repWindow = [];

  /// Every frame of the set, for the final POST
  final List<PoseFrame> _sessionFrames = [];

  PoseFrame? _latest;
  bool _detected = false;

  /// Something the patient cannot exercise through: no camera, no stream
  String? _fatalError;

  /// A failed request that the set carries on past
  String? _lastRepError;

  Readiness? _readiness;
  bool _checkingReadiness = false;

  bool _scoring = false;
  final List<Assessment> _repScores = [];

  @override
  void initState() {
    super.initState();
    _begin();
  }

  @override
  void dispose() {
    _readinessTimer?.cancel();
    _subscription?.cancel();
    _preview?.dispose();
    _cues.stop();
    // Releasing it here rather than on finishing: the patient can also leave via
    // the back button, and a screen that never sleeps again is a flat battery
    WakelockPlus.disable();
    _api.close();
    super.dispose();
  }

  Future<void> _begin() async {
    final granted = await Permission.camera.request();
    if (!granted.isGranted) {
      if (mounted) setState(() => _fatalError = 'Немає доступу до камери');
      return;
    }

    // A set runs for minutes with no touch input, so the screen would otherwise
    // dim and lock part way through
    await WakelockPlus.enable();
    await _cues.init();

    // The patient is side-on to the phone for a sagittal exercise and cannot see
    // the screen, so the BACK camera is the one pointing at them.
    final frontal = widget.exercise.requiredView.toUpperCase() == 'FRONTAL';
    await PoseStream.configure(frontCamera: frontal);

    _subscription = PoseStream.events().listen(
      _onEvent,
      onError: (Object e) {
        if (mounted) setState(() => _fatalError = '$e');
      },
    );
    _readinessTimer = Timer.periodic(_readinessEvery, (_) => _checkReadiness());
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

    _repWindow.add(frame);
    if (_stage == _Stage.preparing) {
      // Keep only enough to check the framing; nothing here is scored
      if (_repWindow.length > _readinessFrames) _repWindow.removeAt(0);
      if (mounted) {
        setState(() {
          _latest = frame;
          _detected = true;
        });
      }
      return;
    }

    final angles = LiveAngles.fromFrame(frame, widget.side, channel: _channel);
    final elevation = _primary.add(angles.primaryDeg);
    _compensation.add(angles.compensationDeg);
    // Direction is taken from the SMOOTHED angle, not the raw one
    _direction.add(elevation);

    _sessionFrames.add(frame);
    if (_repWindow.length > Config.maxFramesPerRep) {
      // The detector has not fired for a long time -- the patient is not reaching
      // the rise threshold. Drop the oldest frame rather than growing unbounded.
      _repWindow.removeAt(0);
    }

    // Entering and leaving the target band drives the hold timer. Using the same
    // tolerance as `repPhase` keeps the timer and the phase in step.
    if (elevation >= _target - kTargetToleranceDeg) {
      _atTargetSinceMs ??= frame.timestampMs;
    } else {
      _atTargetSinceMs = null;
    }

    final rep = _detector.update(elevation, frame.timestampMs);
    if (mounted) {
      setState(() {
        _latest = frame;
        _detected = true;
      });
    }
    if (rep != null) _onRepetition(rep);
    _speakCue();
  }

  /// Slicing the detected repetition out of the buffer and scoring it
  void _onRepetition(DetectedRep rep) {

    final window = [
      for (final f in _repWindow)
        if (f.timestampMs >= rep.startMs && f.timestampMs <= rep.endMs) f,
    ];
    _repWindow.removeWhere((f) => f.timestampMs <= rep.endMs);

    _cues.clearHistory();
    _atTargetSinceMs = null;
    if (window.length >= 10) _scoreRepetition(window);

    if (_detector.completedReps >= _repTarget) {
      _cues.say('Підхід завершено', urgent: true);
      _finish();
    }
  }

  Future<void> _scoreRepetition(List<PoseFrame> window) async {
    setState(() => _scoring = true);
    try {
      final assessment = await _api.assessRepetition(
        exerciseId: widget.exercise.id,
        side: widget.side,
        frames: window,
      );
      if (mounted) {
        setState(() {
          _repScores.add(assessment);
          _lastRepError = null;
        });
      }
    } on ApiException catch (e) {
      if (mounted) setState(() => _lastRepError = e.message);
    } finally {
      if (mounted) setState(() => _scoring = false);
    }
  }

  Future<void> _checkReadiness() async {
    if (_stage != _Stage.preparing || _checkingReadiness) return;
    if (_repWindow.length < 20) return;
    _checkingReadiness = true;
    try {
      final readiness = await _api.checkReadiness(
        exerciseId: widget.exercise.id,
        side: widget.side,
        frames: List<PoseFrame>.from(_repWindow),
      );
      if (mounted) setState(() => _readiness = readiness);
    } on ApiException catch (e) {
      if (mounted) setState(() => _lastRepError = e.message);
    } finally {
      _checkingReadiness = false;
    }
  }

  void _start() {
    _readinessTimer?.cancel();
    _readinessTimer = null;
    _repWindow.clear();
    _detector.reset();
    _primary.reset();
    _compensation.reset();
    _direction.reset();
    setState(() => _stage = _Stage.recording);
    _cues.say('Починайте', urgent: true);
  }

  Future<void> _finish() async {
    if (_stage == _Stage.finishing) return;
    setState(() => _stage = _Stage.finishing);
    await _subscription?.cancel();
    _subscription = null;
    await _cues.stop();

    if (_sessionFrames.isEmpty) {
      if (mounted) {
        setState(() {
          _lastRepError = 'Жодного кадру не записано';
          _stage = _Stage.preparing;
        });
      }
      return;
    }

    try {
      // Guard for a set that overran the service's frame cap
      final posted = RehabApi.fitToCap(_sessionFrames);
      final session = await _api.assessSession(
        exerciseId: widget.exercise.id,
        side: widget.side,
        frames: posted,
      );

      // Keep the exact input the score came from, so a later threshold or model
      // change can be re-scored against it rather than re-recorded
      final at = DateTime.now();
      String? recordingId;
      try {
        recordingId = await LocalStore.saveRecording(
          widget.exercise.id,
          at,
          await _api.recordingPayload(
            exerciseId: widget.exercise.id,
            side: widget.side,
            frames: posted,
          ),
        );
      } catch (_) {
        // A full disk must not cost the patient their score
      }

      final record = SessionRecord.fromAssessment(
        widget.exercise.id,
        session.json,
        at: at,
        recordingId: recordingId,
      );
      await widget.onSessionComplete?.call(record);
      if (!mounted) return;
      Navigator.of(context).pushReplacement(MaterialPageRoute(
        builder: (_) => SummaryScreen(
          session: session,
          exercise: widget.exercise,
          setsRemaining: (widget.sets - widget.setIndex).clamp(0, widget.sets),
          record: record,
          onLabel: widget.onLabel,
        ),
      ));
    } on ApiException catch (e) {
      if (mounted) {
        setState(() {
          _lastRepError = e.message;
          _stage = _Stage.recording;
        });
      }
    }
  }

  /// The current repetition phase, or null when there is nothing to cue from
  RepPhase? get _phase {
    if (!_detected || _fatalError != null) return null;
    return repPhase(
      angleDeg: _primary.value ?? 0,
      target: _target,
      ceiling: widget.exercise.ceilingDeg,
      rising: _direction.rising,
      calibrating: _detector.calibrating,
      detectorInRep: _detector.inRepetition,
      secondsAtTarget: _secondsAtTarget,
      holdSeconds: _holdCueSeconds,
    );
  }

  /// Seconds the angle has been inside the target band
  double get _secondsAtTarget {
    final since = _atTargetSinceMs;
    final now = _latest?.timestampMs;
    if (since == null || now == null) return 0.0;
    return (now - since) / 1000.0;
  }

  bool get _leaning => (_compensation.value ?? 0) > _leanCueDeg;

  /// What to show, and what to say. Speech and text stay in step by construction
  String get _cueText {
    if (_fatalError != null) return _fatalError!;
    if (!_detected) return 'Вас не видно - станьте перед камерою';
    if (_leaning) return _channel.compensationCueUk;

    return switch (_phase) {
      RepPhase.calibrating => 'Тримайте руку внизу…',
      RepPhase.resting => _channel.startCueUk,
      RepPhase.raising => 'Вище…',
      RepPhase.atTarget => 'Достатньо - затримайтесь',
      RepPhase.lowerNow => 'Тепер повільно опускайте руку',
      RepPhase.lowering => 'Опускайте до кінця',
      RepPhase.overshoot => 'Занадто високо - опустіть до рівня плеча',
      null => '',
    };
  }

  void _speakCue() {
    if (_stage != _Stage.recording) return;
    _cues.say(
      _cueText,
      urgent: _leaning || _phase == RepPhase.lowerNow,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand,
          children: [
            SkeletonOverlay(
              frame: _latest,
              image: _preview,
              mirror: widget.exercise.requiredView.toUpperCase() == 'FRONTAL',
            ),
            if (_stage == _Stage.preparing)
              _PreparePanel(
                exercise: widget.exercise,
                readiness: _readiness,
                detected: _detected,
                fatalError: _fatalError,
                checking: _checkingReadiness,
                onStart: _start,
              )
            else
              ..._recordingOverlay(context),
          ],
        ),
      ),
    );
  }

  List<Widget> _recordingOverlay(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final elevation = _primary.value ?? 0;
    final progress = (elevation / _target).clamp(0.0, 1.0).toDouble();
    final phase = _phase;
    final warning =
        _leaning || phase == RepPhase.overshoot || !_detected || _fatalError != null;
    // Reaching the prescribed excursion is the one moment worth marking
    // positively - the patient cannot see the gauge and their own arm at once
    final onTarget = !warning && phase == RepPhase.atTarget;

    return [
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
                  _cueText,
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.titleMedium,
                ),
              ),
            ),
            const SizedBox(height: 10),
            LinearProgressIndicator(value: progress, minHeight: 10),
            const SizedBox(height: 6),
            Text(
              '${elevation.toStringAsFixed(0)}° / ${_target.toStringAsFixed(0)}°'
              '   ·   ${_detector.completedReps} / $_repTarget'
              '${widget.sets > 1 ? '   ·   підхід ${widget.setIndex}/${widget.sets}' : ''}'
              '${_scoring ? '   ·   оцінка…' : ''}',
            ),
            if (_lastRepError != null)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  'Оцінка повторення не вдалася - підхід продовжується',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
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
              onPressed: _stage == _Stage.finishing ? null : _finish,
              child: Text(
                _stage == _Stage.finishing ? 'Обробка…' : 'Завершити',
              ),
            ),
          ],
        ),
      ),
    ];
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
                      .map((e) => widget.exercise.watchedErrors[e.key] ?? e.key)
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

/// The framing check, shown before anything is recorded
class _PreparePanel extends StatelessWidget {
  const _PreparePanel({
    required this.exercise,
    required this.readiness,
    required this.detected,
    required this.fatalError,
    required this.checking,
    required this.onStart,
  });

  final ExerciseInfo exercise;
  final Readiness? readiness;
  final bool detected;
  final String? fatalError;
  final bool checking;
  final VoidCallback onStart;

  bool get _ready => readiness?.ready ?? false;

  String get _headline {
    if (fatalError != null) return fatalError!;
    if (!detected) return 'Вас не видно - станьте перед камерою';
    final current = readiness;
    if (current == null) {
      return checking ? 'Перевіряємо кадр…' : 'Готуємось…';
    }
    return current.messageUk;
  }

  String get _placement => exercise.requiredView.toUpperCase() == 'SAGITTAL'
      ? 'Поставте телефон збоку від себе, на рівні плеча. '
        'Сядьте боком до камери, спина притиснута до спинки стільця.'
      : 'Поставте телефон перед собою, на рівні плеча. '
        'Сядьте обличчям до камери, спина притиснута до спинки стільця.';

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Positioned(
      left: 16,
      right: 16,
      top: 8,
      bottom: 12,
      child: Column(
        children: [
          Card(
            color: _ready ? scheme.primaryContainer : scheme.surfaceContainerHigh,
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Column(
                children: [
                  Row(
                    children: [
                      Icon(
                        _ready ? Icons.check_circle : Icons.center_focus_weak,
                        color: _ready ? scheme.primary : scheme.onSurfaceVariant,
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          _headline,
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      _placement,
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
                  if (readiness != null && !readiness!.viewOk) ...[
                    const SizedBox(height: 8),
                    Align(
                      alignment: Alignment.centerLeft,
                      child: Text(
                        'Потрібно: ${readiness!.requiredView.toLowerCase()}, '
                        'зараз: ${readiness!.detectedView.toLowerCase()} '
                        '(${readiness!.yawDeg.toStringAsFixed(0)}°)',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
          const Spacer(),
          Text(
            exercise.instructionsUk,
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 10),
          FilledButton.icon(
            onPressed: fatalError != null ? null : onStart,
            icon: Icon(_ready ? Icons.play_arrow : Icons.warning_amber),
            label: Text(_ready ? 'Почати' : 'Почати попри попередження'),
          ),
        ],
      ),
    );
  }
}
