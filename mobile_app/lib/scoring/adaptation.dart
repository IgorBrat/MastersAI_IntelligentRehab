/// Next dose from session history (`rehab_ml.adaptation`) and its phrasing
/// (`rehab_ml.api.coach`). The LLM only words the deterministic decision.
library;

import 'dart:convert';

import 'package:http/http.dart' as http;

import 'exercises.dart';
import 'signal.dart';
import 'taxonomy.dart';

final double advanceEqi = eqiGradeBoundaries[3];
final double regressEqi = eqiGradeBoundaries[1];
const int consecutiveSessions = 3;
const double stepFraction = 0.05;
const double fatigueEqiPerRep = -1.8;
const int volumeStep = 2;
const int minReps = 5;
const double safetyFiringRate = 0.5;

final Set<String> safetySlugs = {
  for (final e in ErrorType.values)
    if (e.safetyRelevant) e.slug,
};

enum CoachAction {
  hold('hold'),
  needMoreSessions('need_more_sessions'),
  safetyHold('safety_hold'),
  reduceVolume('reduce_volume'),
  restoreVolume('restore_volume'),
  regressRange('regress_range'),
  advanceRange('advance_range'),
  unlockNext('unlock_next');

  const CoachAction(this.wire);
  final String wire;
}

/// One completed session's aggregate scores. `adaptation.SessionRecord`
class CoachSession {
  const CoachSession({
    required this.exerciseId,
    required this.meanEqi,
    required this.nReps,
    required this.correctRatio,
    required this.fatigueTrend,
    required this.errorFrequency,
  });

  final String exerciseId;
  final double meanEqi;
  final int nReps;
  final double correctRatio;
  final double fatigueTrend;
  final Map<String, double> errorFrequency;

  /// Missing scores coerce to zero, as `SessionRecord.from_dict` does
  factory CoachSession.fromJson(Map<String, dynamic> json) => CoachSession(
        exerciseId: '${json['exercise_id']}',
        meanEqi: (json['mean_eqi'] as num?)?.toDouble() ?? 0.0,
        nReps: (json['n_reps'] as num?)?.toInt() ?? 0,
        correctRatio: (json['correct_ratio'] as num?)?.toDouble() ?? 0.0,
        fatigueTrend: (json['fatigue_trend'] as num?)?.toDouble() ?? 0.0,
        errorFrequency: {
          for (final e in ((json['error_frequency'] as Map?) ?? const {}).entries)
            '${e.key}': (e.value as num).toDouble(),
        },
      );
}

class Decision {
  const Decision(this.action, this.prescription, this.reason, this.evidence, {this.unlockedExerciseId});

  final CoachAction action;
  final Prescription prescription;
  final String reason;
  final Map<String, dynamic> evidence;
  final String? unlockedExerciseId;

  bool get changed => !const {
        CoachAction.hold,
        CoachAction.needMoreSessions,
        CoachAction.safetyHold,
      }.contains(action);

  /// The service's JSON for a decision. `Decision.as_dict`
  Map<String, dynamic> asDict() => {
        'action': action.wire,
        'reason': reason,
        'changed': changed,
        'unlocked_exercise_id': unlockedExerciseId,
        'prescription': {
          'exercise_id': prescription.exerciseId,
          'side': prescription.sideWire,
          'rom_target': pyRoundTo(prescription.romTarget, 4),
          'sets': prescription.sets,
          'reps': prescription.reps,
          'tolerance_scale': prescription.toleranceScale,
        },
        'evidence': evidence,
      };
}

double _step(ExerciseDefinition e) => stepFraction * (e.rom.ceiling - e.rom.floor);

String? nextExercise(String exerciseId, Set<String> unlocked) {
  final index = progressionOrder.indexOf(exerciseId);
  if (index < 0) return null;
  for (final candidate in progressionOrder.skip(index + 1)) {
    if (!unlocked.contains(candidate)) return candidate;
  }
  return null;
}

/// Python's `f"{x:.0f}"` and `f"{x:.0%}"`, half to even
String _f0(double x) => pyRound(x).toString();
String _pct(double x) => '${pyRound(x * 100)}%';

/// `adaptation.decide`
Decision decide(List<CoachSession> history, Prescription current, {Set<String> unlocked = const {}}) {
  final exercise = getExercise(current.exerciseId);
  final same = [for (final s in history) if (s.exerciseId == current.exerciseId) s];
  final recent = same.length > consecutiveSessions ? same.sublist(same.length - consecutiveSessions) : same;
  final evidence = <String, dynamic>{
    'n_sessions_considered': recent.length,
    'sessions_required': consecutiveSessions,
    'rom_target': pyRoundTo(current.romTarget, 4),
    'reps': current.reps,
  };
  Map<String, dynamic> ev(String code) => {...evidence, 'reason_code': code};

  if (recent.length < consecutiveSessions) {
    return Decision(CoachAction.needMoreSessions, current,
        '${recent.length} of $consecutiveSessions sessions recorded', ev('need_more_sessions'));
  }

  final eqis = [for (final s in recent) s.meanEqi];
  final meanEqi = mean(eqis);
  evidence['mean_eqi_recent'] = pyRoundTo(meanEqi, 2);
  evidence['mean_eqi_per_session'] = [for (final v in eqis) pyRoundTo(v, 2)];

  // Safety is judged on the last session only
  String? slug;
  var rate = 0.0;
  recent.last.errorFrequency.forEach((k, v) {
    if (safetySlugs.contains(k) && (slug == null || v > rate)) {
      slug = k;
      rate = v;
    }
  });
  evidence['worst_safety_error'] = slug;
  evidence['worst_safety_rate'] = pyRoundTo(rate, 3);
  if (slug != null && rate > safetyFiringRate) {
    return Decision(CoachAction.safetyHold, current,
        '$slug fired in ${_pct(rate)} of the last session\'s repetitions', ev('safety_hold'));
  }

  // Volume is cut before range, and restored before range is raised
  final fatigue = mean([for (final s in recent) s.fatigueTrend]);
  evidence['mean_fatigue_trend'] = pyRoundTo(fatigue, 3);
  if (fatigue <= fatigueEqiPerRep && current.reps > minReps) {
    final reps = current.reps - volumeStep < minReps ? minReps : current.reps - volumeStep;
    return Decision(
      CoachAction.reduceVolume,
      current.copyWith(reps: reps),
      'quality fell ${fatigue.abs().toStringAsFixed(1)} EQI per repetition across the set',
      ev('reduce_volume'),
    );
  }

  if (eqis.every((v) => v < regressEqi)) {
    final lowered = exercise.rom.withTarget(current.romTarget - _step(exercise));
    if (lowered.target < current.romTarget) {
      return Decision(
        CoachAction.regressRange,
        current.copyWith(romTarget: lowered.target),
        'mean EQI ${_f0(meanEqi)} below ${_f0(regressEqi)} for $consecutiveSessions sessions',
        ev('regress_range'),
      );
    }
    return Decision(CoachAction.hold, current, 'already at the therapist\'s floor for this exercise',
        ev('at_floor'));
  }

  if (eqis.every((v) => v >= advanceEqi)) {
    if (current.reps < exercise.defaultReps) {
      final reps = current.reps + volumeStep > exercise.defaultReps
          ? exercise.defaultReps
          : current.reps + volumeStep;
      return Decision(CoachAction.restoreVolume, current.copyWith(reps: reps),
          'mean EQI ${_f0(meanEqi)} with volume still reduced', ev('restore_volume'));
    }
    final raised = exercise.rom.withTarget(current.romTarget + _step(exercise));
    if (raised.target > current.romTarget) {
      return Decision(
        CoachAction.advanceRange,
        current.copyWith(romTarget: raised.target),
        'mean EQI ${_f0(meanEqi)} at or above ${_f0(advanceEqi)} for $consecutiveSessions sessions',
        ev('advance_range'),
      );
    }
    final following = nextExercise(current.exerciseId, unlocked);
    if (following != null) {
      return Decision(
        CoachAction.unlockNext,
        current,
        'at the ceiling (${_g(exercise.rom.ceiling)}) with mean EQI ${_f0(meanEqi)}',
        ev('unlock_next'),
        unlockedExerciseId: following,
      );
    }
    return Decision(CoachAction.hold, current, 'at the ceiling with no further exercise in the progression',
        ev('at_ceiling_no_next'));
  }

  return Decision(
    CoachAction.hold,
    current,
    'mean EQI ${_f0(meanEqi)} is between ${_f0(regressEqi)} and ${_f0(advanceEqi)} - working at the right difficulty',
    ev('right_difficulty'),
  );
}

/// Python's `:g` for the protocol's ceilings, which are whole or one-decimal numbers
String _g(double x) => x == x.roundToDouble() ? x.toInt().toString() : x.toString();

// ---------------------------------------------------------------- phrasing

const Map<CoachAction, String> actionUk = {
  CoachAction.needMoreSessions:
      'Продовжуйте у тому ж режимі - потрібно більше занять, щоб оцінити прогрес.',
  CoachAction.hold: 'Залишаємо навантаження без змін: ви працюєте на потрібному рівні складності.',
  CoachAction.safetyHold: 'Навантаження не змінюємо, поки не приберемо помилку в техніці.',
  CoachAction.reduceVolume: 'Зменшуємо кількість повторень: якість помітно падала до кінця підходу.',
  CoachAction.restoreVolume: 'Повертаємо звичайну кількість повторень - якість стабільна.',
  CoachAction.regressRange: 'Тимчасово зменшуємо цільову амплітуду, щоб рух виконувався чисто.',
  CoachAction.advanceRange: 'Підвищуємо цільову амплітуду - ви впевнено виконуєте поточну.',
  CoachAction.unlockNext: 'Ви досягли верхньої межі для цієї вправи. Час додати наступну.',
};

List<MapEntry<String, double>> topErrors(CoachSession? session, {int limit = 2}) {
  if (session == null || session.errorFrequency.isEmpty) return const [];
  // Stable on ties, as Python's sorted is
  final ranked = session.errorFrequency.entries.toList();
  final order = List<int>.generate(ranked.length, (i) => i)
    ..sort((a, b) {
      final c = ranked[b].value.compareTo(ranked[a].value);
      return c != 0 ? c : a.compareTo(b);
    });
  return [for (final i in order) if (ranked[i].value > 0) ranked[i]].take(limit).toList();
}

String _cue(String slug) => ErrorType.bySlug(slug)?.patientUk ?? '';

/// Deterministic Ukrainian text. `coach.template_text`
String templateText(Decision decision, CoachSession? last) {
  final parts = <String>[actionUk[decision.action] ?? ''];
  final unlocked = decision.unlockedExerciseId;
  if (decision.action == CoachAction.unlockNext && unlocked != null) {
    parts.add('Наступна вправа: ${getExercise(unlocked).nameUk}.');
  }
  for (final e in topErrors(last)) {
    final cue = _cue(e.key);
    if (cue.isNotEmpty) parts.add('$cue (у ${_pct(e.value)} повторень)');
  }
  return parts.where((p) => p.isNotEmpty).join(' ').trim();
}

String _prompt(Decision decision, ExerciseDefinition exercise, CoachSession? last) {
  final errors = topErrors(last);
  final lines = errors.isEmpty
      ? '- помилок техніки не зафіксовано'
      : errors
          .map((e) => '- ${_cue(e.key).isEmpty ? e.key : _cue(e.key)} (у ${_pct(e.value)} повторень)')
          .join('\n');
  return '''Ти асистент реабілітолога. Напиши коротке повідомлення пацієнтові українською.

Вправа: ${exercise.nameUk}
Рішення програми: ${actionUk[decision.action] ?? decision.action.wire}
Підстава (технічна, не переказуй дослівно): ${decision.reason}
Що варто виправити:
$lines

ПРАВИЛА:
- 2-3 речення, звертайся на "ви", тепло але без надмірних похвал.
- НЕ додавай жодних чисел, градусів, кількості повторень чи відсотків. Їх покаже застосунок.
- НЕ давай медичних порад, НЕ згадуй діагнози, НЕ пропонуй інших вправ.
- Якщо є помилка техніки - назви одну, найважливішу, і як її виправити.
- Тільки текст повідомлення, без заголовків і форматування.''';
}

/// Gemini key comes from --dart-define; without it the template is used
const String _geminiKey = String.fromEnvironment('GEMINI_API_KEY');
const String _geminiModel = String.fromEnvironment('GEMINI_MODEL', defaultValue: 'gemini-2.5-flash');

/// `(text, source, error)`, source is 'gemini' or 'template'. `coach.phrase`
Future<(String, String, String?)> phrase(
  Decision decision,
  ExerciseDefinition exercise,
  CoachSession? last, {
  http.Client? client,
  Duration timeout = const Duration(seconds: 8),
}) async {
  final fallback = templateText(decision, last);
  final key = _geminiKey.trim();
  if (key.isEmpty) return (fallback, 'template', 'GEMINI_API_KEY not set at build time');

  final owned = client == null;
  final http.Client c = client ?? http.Client();
  try {
    final response = await c
        .post(
          Uri.parse('https://generativelanguage.googleapis.com/v1beta/models/'
              '$_geminiModel:generateContent?key=$key'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'contents': [
              {
                'parts': [
                  {'text': _prompt(decision, exercise, last)},
                ],
              },
            ],
            'generationConfig': {
              'temperature': 0.2,
              'maxOutputTokens': 1024,
              // Thinking tokens count against maxOutputTokens, so thinking is off
              'thinkingConfig': {'thinkingBudget': 0},
            },
          }),
        )
        .timeout(timeout);
    if (response.statusCode >= 400) {
      final body = utf8.decode(response.bodyBytes);
      return (fallback, 'template', 'HTTP ${response.statusCode}: ${body.length > 200 ? body.substring(0, 200) : body}');
    }
    final payload = jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>;
    final candidate = (payload['candidates'] as List).first as Map;
    final parts = (candidate['content'] as Map)['parts'] as List;
    final text = parts.map((p) => (p as Map)['text'] ?? '').join().trim();
    if (text.isEmpty) return (fallback, 'template', 'empty generation');
    // Truncated output falls back to the template
    final finish = candidate['finishReason'] as String?;
    if (finish != null && finish != 'STOP') {
      return (fallback, 'template', 'generation ended with $finish');
    }
    return (text, 'gemini', null);
  } catch (e) {
    return (fallback, 'template', '${e.runtimeType}: $e');
  } finally {
    if (owned) c.close();
  }
}
