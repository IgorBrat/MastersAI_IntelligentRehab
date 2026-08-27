import 'package:flutter/material.dart';

import '../api/client.dart';
import '../state/app_state.dart';
import '../storage/local_store.dart';
import 'record_screen.dart';
import 'video_screen.dart';

/// The exercises the patient may do today, each with the dose in force
class ExerciseScreen extends StatelessWidget {
  const ExerciseScreen({super.key, required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    if (state.loading) {
      return const Center(child: CircularProgressIndicator());
    }
    if (!state.connected) {
      return _NotConnected(state: state);
    }

    final available = state.available;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text(
          state.profile.name.isEmpty
              ? 'Заняття на сьогодні'
              : 'Заняття на сьогодні, ${state.profile.name}',
          style: Theme.of(context).textTheme.titleLarge,
        ),
        const SizedBox(height: 4),
        Text(
          'Рука: ${state.side.wire == 'left' ? 'ліва' : 'права'}',
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: 16),
        for (final exercise in available)
          _ExerciseCard(
            state: state,
            exercise: exercise,
            dose: state.doseFor(exercise),
          ),
        if (available.length < state.exercises.length) ...[
          const SizedBox(height: 8),
          Card(
            child: ListTile(
              leading: const Icon(Icons.lock_outline),
              title: Text(
                '${state.exercises.length - available.length} вправ ще закрито',
              ),
              subtitle: const Text(
                'Тренер відкриє наступну, коли ви впевнено виконуватимете поточну',
              ),
            ),
          ),
        ],
      ],
    );
  }
}

class _ExerciseCard extends StatelessWidget {
  const _ExerciseCard({
    required this.state,
    required this.exercise,
    required this.dose,
  });

  final AppState state;
  final ExerciseInfo exercise;
  final Dose dose;

  /// True when the coach has moved this exercise off its protocol default
  bool get _adjusted => (dose.romTarget - exercise.targetDeg).abs() > 0.01 ||
      dose.reps != exercise.defaultReps;

  void _open(BuildContext context, Widget page) {
    Navigator.of(context).push(MaterialPageRoute(builder: (_) => page));
  }

  @override
  Widget build(BuildContext context) {
    final sessions = state.historyFor(exercise.id).length;
    return Card(
      margin: const EdgeInsets.only(bottom: 14),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(exercise.nameUk,
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Text(exercise.instructionsUk),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _Chip(
                  icon: Icons.open_in_full,
                  label: 'Ціль ${dose.romTarget.toStringAsFixed(0)}°',
                  highlight: _adjusted,
                ),
                _Chip(icon: Icons.repeat, label: '${dose.reps} повторень'),
                _Chip(icon: Icons.videocam, label: exercise.requiredView),
                if (sessions > 0)
                  _Chip(icon: Icons.history, label: '$sessions занять'),
              ],
            ),
            if (_adjusted) ...[
              const SizedBox(height: 8),
              Text(
                'Скориговано тренером (протокол: '
                '${exercise.targetDeg.toStringAsFixed(0)}°, '
                '${exercise.defaultReps} повторень)',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
            const SizedBox(height: 14),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: () => _open(
                      context,
                      RecordScreen(
                        baseUrl: state.profile.serviceUrl.trim(),
                        exercise: exercise,
                        side: state.side,
                        romTarget: dose.romTarget,
                        onSessionComplete: state.addSession,
                      ),
                    ),
                    icon: const Icon(Icons.videocam),
                    label: const Text('Камера'),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => _open(
                      context,
                      VideoScreen(
                        baseUrl: state.profile.serviceUrl.trim(),
                        exercise: exercise,
                        side: state.side,
                        onSessionComplete: state.addSession,
                      ),
                    ),
                    icon: const Icon(Icons.movie),
                    label: const Text('Відео'),
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

class _Chip extends StatelessWidget {
  const _Chip({required this.icon, required this.label, this.highlight = false});

  final IconData icon;
  final String label;
  final bool highlight;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Chip(
      avatar: Icon(icon, size: 16,
          color: highlight ? scheme.onPrimaryContainer : null),
      label: Text(label),
      backgroundColor: highlight ? scheme.primaryContainer : null,
      visualDensity: VisualDensity.compact,
    );
  }
}

/// Shown instead of the exercise list when the service is unreachable
class _NotConnected extends StatelessWidget {
  const _NotConnected({required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final noUrl = state.profile.serviceUrl.trim().isEmpty;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(28),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.cloud_off, size: 56, color: scheme.error),
            const SizedBox(height: 16),
            Text(
              noUrl ? 'Сервіс не налаштовано' : 'Немає зв\'язку зі сервісом',
              style: Theme.of(context).textTheme.titleMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              noUrl
                  ? 'Вкажіть LAN-адресу ноутбука у вкладці «Профіль».'
                  : state.connectionError ?? '',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 20),
            FilledButton.tonalIcon(
              onPressed: state.connect,
              icon: const Icon(Icons.refresh),
              label: const Text('Спробувати знову'),
            ),
            const SizedBox(height: 12),
            const Text(
              'Історія та профіль працюють без зв\'язку.',
              style: TextStyle(fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }
}
