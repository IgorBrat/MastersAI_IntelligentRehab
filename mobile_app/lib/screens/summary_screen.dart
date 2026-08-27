import 'package:flutter/material.dart';

import '../api/client.dart';

/// Session report. Everything shown here came from the service's rule engine - the
/// on-device cues during recording play no part in the score
class SummaryScreen extends StatelessWidget {
  const SummaryScreen({
    super.key,
    required this.session,
    required this.exercise,
  });

  final Assessment session;
  final ExerciseInfo exercise;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    if (!session.usable) {
      return Scaffold(
        appBar: AppBar(title: const Text('Запис не підходить')),
        body: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.videocam_off, size: 56, color: scheme.error),
              const SizedBox(height: 16),
              Text(
                session.rejectionUk ?? 'Запис не вдалося оцінити.',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 24),
              FilledButton(
                onPressed: () => Navigator.of(context)
                    .popUntil((route) => route.isFirst),
                child: const Text('Спробувати ще раз'),
              ),
            ],
          ),
        ),
      );
    }

    final reps = session.repetitions;
    final frequency = session.errorFrequency.entries
        .where((e) => e.value > 0)
        .toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    return Scaffold(
      appBar: AppBar(title: const Text('Результат')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(18),
              child: Column(
                children: [
                  Text(
                    session.meanEqi?.toStringAsFixed(0) ?? '-',
                    style: Theme.of(context).textTheme.displayMedium,
                  ),
                  const Text('середній індекс якості (EQI)'),
                  const SizedBox(height: 10),
                  Text('повторень: ${session.nReps}'
                      '${session.grade != null ? '  ·  оцінка: ${session.grade}/4' : ''}'),
                ],
              ),
            ),
          ),
          if (frequency.isNotEmpty) ...[
            const SizedBox(height: 18),
            Text('Що виправити', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            // Ordered by how often it happened, not by worst single repetition
            for (final entry in frequency)
              Card(
                child: ListTile(
                  title: Text(exercise.watchedErrors[entry.key] ?? entry.key),
                  trailing: Text('${(entry.value * 100).toStringAsFixed(0)}%'),
                ),
              ),
          ],
          const SizedBox(height: 18),
          Text('Повторення', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          for (var i = 0; i < reps.length; i++)
            ListTile(
              dense: true,
              leading: Text('${i + 1}'),
              title: Text('EQI ${reps[i].eqi?.toStringAsFixed(0) ?? '-'}'),
              subtitle: reps[i].errors.isEmpty
                  ? const Text('без помилок')
                  : Text(reps[i].errors.keys.join(', ')),
            ),
          const SizedBox(height: 24),
          FilledButton(
            onPressed: () =>
                Navigator.of(context).popUntil((route) => route.isFirst),
            child: const Text('До початку'),
          ),
        ],
      ),
    );
  }
}
