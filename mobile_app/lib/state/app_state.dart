import 'package:flutter/foundation.dart';

import '../analysis/live_angles.dart';
import '../api/client.dart';
import '../scoring/rehab_engine.dart';
import '../storage/local_store.dart';
import '../storage/session_record.dart';

/// Shared app state: profile, stored history and the scoring engine
class AppState extends ChangeNotifier {
  Profile _profile = const Profile();
  List<SessionRecord> _history = const [];
  List<ExerciseInfo> _exercises = const [];

  bool _loading = true;
  String? _connectionError;
  Map<String, dynamic>? _serviceInfo;
  RehabEngine? _engine;

  /// The on-device scoring engine, once its assets have loaded
  RehabEngine? get engine => _engine;

  Profile get profile => _profile;
  List<SessionRecord> get history => _history;
  List<ExerciseInfo> get exercises => _exercises;
  bool get loading => _loading;
  String? get connectionError => _connectionError;
  Map<String, dynamic>? get serviceInfo => _serviceInfo;

  /// Whether the on-device engine and its assets are loaded
  bool get connected => _engine != null && _exercises.isNotEmpty && _connectionError == null;

  /// Whether the optional laptop video service is configured
  bool get hasVideoService => _profile.serviceUrl.trim().isNotEmpty;

  BodySide get side =>
      _profile.side == 'right' ? BodySide.right : BodySide.left;

  /// Exercises the patient may do: the first of the progression plus anything unlocked
  List<ExerciseInfo> get available {
    if (_exercises.isEmpty) return const [];
    final unlocked = _profile.unlocked.toSet();
    final visible = [
      for (final e in _exercises)
        if (unlocked.contains(e.id)) e,
    ];
    if (visible.isEmpty) return [_exercises.first];
    return visible;
  }

  Dose doseFor(ExerciseInfo exercise) =>
      _profile.doses[exercise.id] ??
      Dose(romTarget: exercise.targetDeg, reps: exercise.defaultReps, sets: 2);

  List<SessionRecord> historyFor(String exerciseId) =>
      [for (final s in _history) if (s.exerciseId == exerciseId) s];

  Future<void> load() async {
    _loading = true;
    notifyListeners();
    _profile = await LocalStore.profile();
    _history = await LocalStore.history();
    _loading = false;
    notifyListeners();
    await connect();
  }

  /// Loads the bundled protocol, thresholds and model
  Future<void> connect() async {
    try {
      final engine = _engine ?? await RehabEngine.load();
      _engine = engine;
      _serviceInfo = engine.health();
      _exercises = engine.exercises();
      _connectionError = null;
      // First run: unlock the first exercise of the progression
      if (_profile.unlocked.isEmpty && _exercises.isNotEmpty) {
        await saveProfile(_profile.copyWith(unlocked: [_exercises.first.id]));
        return;
      }
    } catch (e) {
      _connectionError = 'Не вдалося завантажити модуль оцінювання: $e';
      _exercises = const [];
    } finally {
      notifyListeners();
    }
  }

  Future<void> saveProfile(Profile profile) async {
    _profile = profile;
    await LocalStore.saveProfile(profile);
    notifyListeners();
  }

  Future<void> addSession(SessionRecord record) async {
    await LocalStore.addSession(record);
    _history = await LocalStore.history();
    notifyListeners();
  }

  Future<void> clearHistory() async {
    await LocalStore.clearHistory();
    _history = const [];
    notifyListeners();
  }

  /// Stores a supervising specialist's verdict on one set; the coach never sees it
  Future<void> labelSession(
    SessionRecord session, {
    required SupervisorLabel label,
    String? note,
  }) async {
    final updated = session.copyWith(label: label, note: note);
    await LocalStore.updateSession(updated);
    _history = await LocalStore.history();
    notifyListeners();
  }

  /// Criteria switched off for this sensor, by slug
  List<String> get excludedCriteria => [
        for (final slug in (_serviceInfo?['excluded_criteria'] as List? ?? []))
          slug.toString(),
      ];

  /// Applies a coach decision: stores the new dose and any unlocked exercise
  Future<void> applyAdvice(String exerciseId, CoachAdvice advice) async {
    final doses = Map<String, Dose>.from(_profile.doses);
    doses[exerciseId] = Dose(
      romTarget: advice.romTarget,
      reps: advice.reps,
      sets: advice.sets,
    );
    final unlocked = List<String>.from(_profile.unlocked);
    final newly = advice.unlockedExerciseId;
    if (newly != null && !unlocked.contains(newly)) unlocked.add(newly);
    await saveProfile(_profile.copyWith(doses: doses, unlocked: unlocked));
  }
}
