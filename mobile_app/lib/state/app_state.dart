import 'package:flutter/foundation.dart';

import '../analysis/live_angles.dart';
import '../api/client.dart';
import '../storage/local_store.dart';
import '../storage/session_record.dart';

/// Everything the four pages share: the profile, the stored history, the connection
class AppState extends ChangeNotifier {
  Profile _profile = const Profile();
  List<SessionRecord> _history = const [];
  List<ExerciseInfo> _exercises = const [];

  bool _loading = true;
  String? _connectionError;
  Map<String, dynamic>? _serviceInfo;

  Profile get profile => _profile;
  List<SessionRecord> get history => _history;
  List<ExerciseInfo> get exercises => _exercises;
  bool get loading => _loading;
  String? get connectionError => _connectionError;
  Map<String, dynamic>? get serviceInfo => _serviceInfo;

  bool get connected => _exercises.isNotEmpty && _connectionError == null;

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
    if (_profile.serviceUrl.trim().isNotEmpty) {
      await connect();
    }
  }

  /// Fetching the protocol from the service
  Future<void> connect() async {
    final url = _profile.serviceUrl.trim();
    if (url.isEmpty) {
      _connectionError = 'Адреса сервісу не вказана';
      notifyListeners();
      return;
    }
    final api = RehabApi(url);
    try {
      _serviceInfo = await api.health();
      _exercises = await api.exercises();
      _connectionError = null;
      // First run: seed the progression with its first exercise so the patient has
      // something to do before the coach has any history to reason about
      if (_profile.unlocked.isEmpty && _exercises.isNotEmpty) {
        await saveProfile(_profile.copyWith(unlocked: [_exercises.first.id]));
        return;
      }
    } on ApiException catch (e) {
      _connectionError = e.message;
      _exercises = const [];
    } finally {
      api.close();
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

  /// Recording a supervising specialist's verdict on one stored set
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

  /// Criteria the service has switched off, by slug
  List<String> get excludedCriteria => [
        for (final slug in (_serviceInfo?['excluded_criteria'] as List? ?? []))
          slug.toString(),
      ];

  /// Applying a coach decision: store the new dose and any unlocked exercise
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
