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
import '../api/client.dart' show Assessment, ExerciseInfo, Readiness;
import '../config.dart';
import '../pose/pose_stream.dart';
import '../scoring/rehab_engine.dart';
import '../storage/local_store.dart';
import '../storage/session_record.dart';
import '../widgets/skeleton_overlay.dart';
import 'summary_screen.dart';
import '../clinical/error_names.dart';

/// What the screen is doing; the framing check comes first
enum _Stage {
  /// Camera live, checking the framing, nothing recorded yet
  preparing,
  recording,
  finishing,
}

/// The recording screen: framing check, live skeleton, live angle, rep counter
class RecordScreen extends StatefulWidget {
  const RecordScreen({
    super.key,
    required this.engine,
    required this.exercise,
    required this.side,
    this.romTarget,
    this.reps,
    this.setIndex = 1,
    this.sets = 1,
    this.onSessionComplete,
    this.onLabel,
  });

  final RehabEngine engine;
  final ExerciseInfo exercise;
  final BodySide side;

  /// The coach's current target, when one has been applied
  final double? romTarget;

  /// Repetitions this set asks for; the set ends on reaching it
  final int? reps;

  /// Which set of how many, for the label only. One screen visit is one set,
  /// so `fatigue_trend` never spans a rest period
  final int setIndex;
  final int sets;

  final Future<void> Function(SessionRecord)? onSessionComplete;

  /// Recording the supervisor's verdict, offered on the summary
  final Future<void> Function(SessionRecord, SupervisorLabel)? onLabel;

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  /// Lean cue threshold, degrees. UI heuristic, looser than the scored criterion
  static const double _leanCueDeg = 12.0;

  /// Girdle hike cue threshold, torso lengths. UI heuristic, looser than the
  /// scored band (~0.07 TL)
  static const double _hikeCueTl = 0.10;

  /// Speed cue fires above this multiple of the prescribed average phase speed.
  /// A minimum-jerk movement at tempo peaks at ~1.9x
  static const double _speedCueFactor = 3.5;

  /// Frames kept while preparing, for the readiness check
  static const int _readinessFrames = 60;

  /// How often the framing is re-checked while preparing
  static const Duration _readinessEvery = Duration(seconds: 2);

  RehabEngine get _api => widget.engine;
  final SpokenCues _cues = SpokenCues();

  /// The target actually in force for this patient
  double get _target => widget.romTarget ?? widget.exercise.targetDeg;
  int get _repTarget => widget.reps ?? widget.exercise.defaultReps;

  late final RepDetector _detector = RepDetector(targetDeg: _target);
  final Smoother _primary = Smoother();
  final Smoother _compensation = Smoother();

  /// Which channels to derive on device, taken from the protocol
  late final LiveChannel _channel =
      LiveChannel.fromPrimary(widget.exercise.primaryChannel);
  final DirectionTracker _direction = DirectionTracker();

  /// Girdle hike above the resting height, smoothed, torso lengths
  final Smoother _hike = Smoother();

  /// Girdle heights seen while the detector learns the resting pose
  final List<double> _girdleRest = [];
  double? _girdleBaseline;

  /// Angular velocity of the smoothed primary angle, degrees per second
  final Smoother _velocity = Smoother();
  double? _lastPrimary;
  int? _lastPrimaryMs;

  /// Frame timestamp when the angle entered the target band, null outside it
  int? _atTargetSinceMs;

  /// Hold to cue at the top; falls back to 2 s when self-paced scoring sends 0
  double get _holdCueSeconds =>
      widget.exercise.holdSeconds > 0 ? widget.exercise.holdSeconds : 2.0;

  StreamSubscription<PoseEvent>? _subscription;
  Timer? _readinessTimer;

  _Stage _stage = _Stage.preparing;

  ui.Image? _preview;
  bool _decoding = false;

  /// Frames of the repetition in progress
  final List<PoseFrame> _repWindow = [];

  /// Every frame of the set, for the session score
  final List<PoseFrame> _sessionFrames = [];

  PoseFrame? _latest;
  bool _detected = false;

  /// Error that stops the session: no camera, no stream
  String? _fatalError;

  /// Error the set carries on past; cleared by the next success
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
    // Released here so leaving via the back button also lets the screen sleep
    WakelockPlus.disable();
    super.dispose();
  }

  Future<void> _begin() async {
    final granted = await Permission.camera.request();
    if (!granted.isGranted) {
      if (mounted) setState(() => _fatalError = 'Немає доступу до камери');
      return;
    }

    // Keep the screen on through a set with no touch input
    await WakelockPlus.enable();
    await _cues.init();

    // Sagittal exercises use the back camera; the patient is side-on
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
      // Skip a corrupt preview frame
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
      // Keep only enough to check the framing
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
    _direction.add(elevation);

    // Speed off the smoothed angle, on frame timestamps
    final last = _lastPrimary;
    final lastMs = _lastPrimaryMs;
    if (last != null && lastMs != null && frame.timestampMs > lastMs) {
      _velocity.add((elevation - last) * 1000.0 / (frame.timestampMs - lastMs));
    }
    _lastPrimary = elevation;
    _lastPrimaryMs = frame.timestampMs;

    // Girdle resting height is learned with the detector baseline (median)
    if (_detector.calibrating) {
      _girdleRest.add(angles.girdleHeightTl);
    } else {
      final baseline = _girdleBaseline ??= _girdleRest.isEmpty
          ? angles.girdleHeightTl
          : (List<double>.from(_girdleRest)..sort())[_girdleRest.length ~/ 2];
      _hike.add(angles.girdleHikeTl(baseline));
    }

    _sessionFrames.add(frame);
    if (_repWindow.length > Config.maxFramesPerRep) {
      // No rep detected for a long time; cap the buffer
      _repWindow.removeAt(0);
    }

    // Hold timer, same tolerance as `repPhase`
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
    // Score only the movement bounds, not the rest before it
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
    } on ScoringException catch (e) {
      // The set continues; the session score still covers every frame
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
    } on ScoringException catch (e) {
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
    _hike.reset();
    _girdleRest.clear();
    _girdleBaseline = null;
    _velocity.reset();
    _lastPrimary = null;
    _lastPrimaryMs = null;
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
      // Trim a set that overran the frame cap
      final posted = RehabEngine.fitToCap(_sessionFrames);
      final session = await _api.assessSession(
        exerciseId: widget.exercise.id,
        side: widget.side,
        frames: posted,
      );

      // Save the scored input so it can be re-scored later
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
        // A failed save does not block the score
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
    } on ScoringException catch (e) {
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

  double get _secondsAtTarget {
    final since = _atTargetSinceMs;
    final now = _latest?.timestampMs;
    if (since == null || now == null) return 0.0;
    return (now - since) / 1000.0;
  }

  bool get _leaning => (_compensation.value ?? 0) > _leanCueDeg;

  /// Girdle rising beyond what the arm's elevation explains, inside a rep only
  bool get _hiking =>
      _girdleBaseline != null &&
      _detector.inRepetition &&
      (_hike.value ?? 0) > _hikeCueTl;

  /// Prescribed lifting and lowering speeds, deg/s; null when the primary
  /// channel is not derived on device
  (double, double)? get _phaseSpeeds {
    final channelMatches = switch (_channel) {
      LiveChannel.shoulderFlexion => widget.exercise.primaryChannel == 'shoulder_flexion',
      LiveChannel.shoulderAbduction => widget.exercise.primaryChannel == 'shoulder_abduction',
      LiveChannel.elbowFlexion => widget.exercise.primaryChannel == 'elbow_flexion',
      LiveChannel.shoulderElevation => false,
    };
    final baseline = _detector.baseline;
    if (!channelMatches || baseline == null) return null;
    final excursion = (_target - baseline).abs();
    final up = widget.exercise.concentricSeconds;
    final down = widget.exercise.eccentricSeconds;
    if (excursion < 1e-6 || up <= 0 || down <= 0) return null;
    return (excursion / up, excursion / down);
  }

  bool get _rushing {
    final speeds = _phaseSpeeds;
    final v = _velocity.value;
    if (speeds == null || v == null || !_detector.inRepetition) return false;
    final limit = _speedCueFactor * (v >= 0 ? speeds.$1 : speeds.$2);
    return v.abs() > limit;
  }

  /// A compensation is under way; its cue is urgent
  bool get _warningActive => _hiking || _rushing || _leaning;

  /// Cue text, shown and spoken
  String get _cueText {
    if (_fatalError != null) return _fatalError!;
    if (!_detected) return 'Вас не видно - станьте перед камерою';
    // Injury-risk errors before the lean
    if (_hiking) return 'Опустіть плече - не піднімайте його до вуха';
    if (_rushing) return 'Повільніше - контролюйте рух';
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
    // Compensation and end-of-hold interrupt; phase guidance waits its turn
    _cues.say(
      _cueText,
      urgent: _warningActive || _phase == RepPhase.lowerNow,
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
        _warningActive || phase == RepPhase.overshoot || !_detected || _fatalError != null;
    // Highlight reaching the target
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
                      .map((e) => errorCueUk(e.key,
                          watched: widget.exercise.watchedErrors))
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
          // Startable even when not ready; the warning stays visible
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
