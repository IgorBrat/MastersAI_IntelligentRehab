import 'package:flutter/material.dart';

import '../api/client.dart';
import '../state/app_state.dart';

/// The adaptive coach: what to change next, and why
class CoachScreen extends StatefulWidget {
  const CoachScreen({super.key, required this.state});

  final AppState state;

  @override
  State<CoachScreen> createState() => _CoachScreenState();
}

class _CoachScreenState extends State<CoachScreen> {
  ExerciseInfo? _exercise;
  CoachAdvice? _advice;
  bool _busy = false;
  bool _applied = false;
  String? _error;

  AppState get state => widget.state;

  ExerciseInfo? get _selected {
    final available = state.available;
    if (available.isEmpty) return null;
    final chosen = _exercise;
    if (chosen != null && available.any((e) => e.id == chosen.id)) return chosen;
    return available.first;
  }

  Future<void> _ask() async {
    final exercise = _selected;
    if (exercise == null) return;
    setState(() {
      _busy = true;
      _error = null;
      _advice = null;
      _applied = false;
    });
    final api = RehabApi(state.profile.serviceUrl.trim());
    try {
      final dose = state.doseFor(exercise);
      final advice = await api.coach(
        exerciseId: exercise.id,
        side: state.side,
        history: [
          for (final s in state.history) s.toCoachJson(),
        ],
        romTarget: dose.romTarget,
        reps: dose.reps,
        sets: dose.sets,
        unlocked: state.profile.unlocked,
      );
      if (mounted) setState(() => _advice = advice);
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    } finally {
      api.close();
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _apply() async {
    final exercise = _selected;
    final advice = _advice;
    if (exercise == null || advice == null) return;
    await state.applyAdvice(exercise.id, advice);
    if (mounted) setState(() => _applied = true);
  }

  @override
  Widget build(BuildContext context) {
    if (!state.connected) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(28),
          child: Text(
            'Тренер потребує зв\'язку зі сервісом.\n\n'
            'Рішення про навантаження ухвалює сервіс, а не телефон.',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    final exercise = _selected;
    if (exercise == null) {
      return const Center(child: Text('Немає доступних вправ'));
    }

    final sessions = state.historyFor(exercise.id).length;
    final advice = _advice;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        DropdownButtonFormField<String>(
          value: exercise.id,
          decoration: const InputDecoration(
            labelText: 'Вправа',
            border: OutlineInputBorder(),
          ),
          items: [
            for (final e in state.available)
              DropdownMenuItem(value: e.id, child: Text(e.nameUk)),
          ],
          onChanged: (id) => setState(() {
            _exercise = state.available.firstWhere((e) => e.id == id);
            _advice = null;
            _applied = false;
          }),
        ),
        const SizedBox(height: 12),
        Text(
          'Занять записано: $sessions',
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: 16),
        FilledButton.icon(
          onPressed: _busy ? null : _ask,
          icon: const Icon(Icons.auto_awesome),
          label: Text(_busy ? 'Аналіз…' : 'Що робити далі?'),
        ),
        if (_error != null) ...[
          const SizedBox(height: 14),
          Text(_error!,
              style: TextStyle(color: Theme.of(context).colorScheme.error)),
        ],
        if (advice != null) ...[
          const SizedBox(height: 20),
          _MessageCard(advice: advice),
          const SizedBox(height: 14),
          _DoseCard(advice: advice, exercise: exercise, state: state),
          const SizedBox(height: 14),
          _WhyCard(advice: advice),
          if (advice.changed) ...[
            const SizedBox(height: 16),
            FilledButton.tonalIcon(
              onPressed: _applied ? null : _apply,
              icon: Icon(_applied ? Icons.check : Icons.save_outlined),
              label: Text(_applied ? 'Застосовано' : 'Застосувати'),
            ),
          ],
        ],
      ],
    );
  }
}

/// The coaching sentence. Labelled with its origin so template text is never passed off as
/// generated, and a failed generation is visible rather than silent
class _MessageCard extends StatelessWidget {
  const _MessageCard({required this.advice});

  final CoachAdvice advice;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      color: scheme.secondaryContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(advice.textUk,
                style: Theme.of(context).textTheme.bodyLarge),
            const SizedBox(height: 10),
            Row(
              children: [
                Icon(
                  advice.generated ? Icons.auto_awesome : Icons.article_outlined,
                  size: 14,
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    advice.generated
                        ? 'Текст згенеровано ШІ. Рішення про навантаження - за правилами.'
                        : 'Стандартний текст${advice.textError != null ? ' (ШІ недоступний)' : ''}',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

/// The dose. Rendered from the prescription, never from the text above it
class _DoseCard extends StatelessWidget {
  const _DoseCard({
    required this.advice,
    required this.exercise,
    required this.state,
  });

  final CoachAdvice advice;
  final ExerciseInfo exercise;
  final AppState state;

  @override
  Widget build(BuildContext context) {
    final current = state.doseFor(exercise);
    final targetChanged = (advice.romTarget - current.romTarget).abs() > 0.01;
    final repsChanged = advice.reps != current.reps;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Навантаження',
                style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 10),
            _Row(
              label: 'Цільова амплітуда',
              from: '${current.romTarget.toStringAsFixed(0)}°',
              to: '${advice.romTarget.toStringAsFixed(0)}°',
              changed: targetChanged,
            ),
            _Row(
              label: 'Повторень',
              from: '${current.reps}',
              to: '${advice.reps}',
              changed: repsChanged,
            ),
            if (advice.unlockedExerciseId != null) ...[
              const Divider(height: 22),
              Row(
                children: [
                  const Icon(Icons.lock_open, size: 18),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Відкривається нова вправа',
                      style: Theme.of(context).textTheme.bodyMedium,
                    ),
                  ),
                ],
              ),
            ],
            if (!advice.changed) ...[
              const SizedBox(height: 8),
              Text(
                'Без змін',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
            const SizedBox(height: 10),
            Text(
              'Межі, задані протоколом: до ${exercise.ceilingDeg.toStringAsFixed(0)}°. '
              'Тренер не може вийти за них.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

class _Row extends StatelessWidget {
  const _Row({
    required this.label,
    required this.from,
    required this.to,
    required this.changed,
  });

  final String label;
  final String from;
  final String to;
  final bool changed;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        children: [
          Expanded(child: Text(label)),
          if (changed)
            Text('$from → ',
                style: TextStyle(
                  decoration: TextDecoration.lineThrough,
                  color: scheme.outline,
                )),
          Text(
            to,
            style: TextStyle(
              fontWeight: changed ? FontWeight.bold : FontWeight.normal,
              color: changed ? scheme.primary : null,
            ),
          ),
        ],
      ),
    );
  }
}

/// The audit trail. Present because a dose change a patient cannot interrogate is a dose
/// change they will not trust, and because a specialist reading the report needs it
class _WhyCard extends StatelessWidget {
  const _WhyCard({required this.advice});

  final CoachAdvice advice;

  @override
  Widget build(BuildContext context) {
    final evidence = advice.evidence;
    final eqis = evidence['mean_eqi_per_session'];
    return Card(
      child: ExpansionTile(
        leading: const Icon(Icons.rule),
        title: const Text('Чому саме так'),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        children: [
          Align(
            alignment: Alignment.centerLeft,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Правило: ${advice.action}'),
                const SizedBox(height: 4),
                Text(advice.reason),
                const SizedBox(height: 10),
                if (eqis is List)
                  Text('EQI за останні заняття: ${eqis.join(', ')}'),
                if (evidence['mean_fatigue_trend'] != null)
                  Text('Зміна якості за підхід: '
                      '${evidence['mean_fatigue_trend']} EQI/повторення'),
                if (evidence['worst_safety_error'] != null)
                  Text('Помилка безпеки: ${evidence['worst_safety_error']} '
                      '(${evidence['worst_safety_rate']})'),
                const SizedBox(height: 10),
                Text(
                  'Потрібно ${evidence['sessions_required'] ?? '-'} занять поспіль, '
                  'щоб змінити навантаження. Записано: '
                  '${evidence['n_sessions_considered'] ?? 0}.',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
