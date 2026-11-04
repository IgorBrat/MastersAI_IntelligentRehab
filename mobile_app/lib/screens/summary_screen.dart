import 'dart:async';

import 'package:flutter/material.dart';

import '../api/client.dart';
import '../storage/session_record.dart';
import '../clinical/error_names.dart';

/// Session report from the scoring engine; live cues play no part in the score
class SummaryScreen extends StatelessWidget {
  const SummaryScreen({
    super.key,
    required this.session,
    required this.exercise,
    this.setsRemaining = 0,
    this.record,
    this.onLabel,
  });

  final Assessment session;
  final ExerciseInfo exercise;

  /// Sets still due today
  final int setsRemaining;

  /// The stored session, when there is one to label
  final SessionRecord? record;

  final Future<void> Function(SessionRecord, SupervisorLabel)? onLabel;

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
            // Ordered by frequency
            for (final entry in frequency)
              Card(
                child: ListTile(
                  title: Text(errorNameUk(entry.key)),
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
                  : Text(reps[i].errors.keys.map(errorNameUk).join(', ')),
            ),
          if (record != null && onLabel != null) ...[
            const SizedBox(height: 18),
            _LabelCard(
              record: record!,
              onLabel: (label) => onLabel!(record!, label),
            ),
          ],
          if (setsRemaining > 0) ...[
            const SizedBox(height: 18),
            _RestCard(setsRemaining: setsRemaining),
          ],
          const SizedBox(height: 24),
          FilledButton(
            onPressed: () =>
                Navigator.of(context).popUntil((route) => route.isFirst),
            child: Text(
              setsRemaining > 0 ? 'До наступного підходу' : 'До початку',
            ),
          ),
        ],
      ),
    );
  }
}

/// The supervisor's verdict on the set, stored as a reference label and never
/// passed to the coach
class _LabelCard extends StatelessWidget {
  const _LabelCard({required this.record, required this.onLabel});

  final SessionRecord record;
  final Future<void> Function(SupervisorLabel) onLabel;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Оцінка спеціаліста',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 4),
            Text(
              'Не впливає на оцінку системи - зберігається як еталон для '
              'перевірки.',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 12),
            SegmentedButton<SupervisorLabel>(
              segments: const [
                ButtonSegment(
                  value: SupervisorLabel.correct,
                  icon: Icon(Icons.thumb_up_outlined),
                  label: Text('Правильно'),
                ),
                ButtonSegment(
                  value: SupervisorLabel.incorrect,
                  icon: Icon(Icons.thumb_down_outlined),
                  label: Text('З помилками'),
                ),
              ],
              selected: record.label == SupervisorLabel.none
                  ? const <SupervisorLabel>{}
                  : {record.label},
              emptySelectionAllowed: true,
              onSelectionChanged: (selected) {
                if (selected.isNotEmpty) onLabel(selected.first);
              },
            ),
            if (record.recordingId != null) ...[
              const SizedBox(height: 10),
              Text(
                'Запис збережено - можна перескорити пізніше',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// Rest countdown before the next set
class _RestCard extends StatefulWidget {
  const _RestCard({required this.setsRemaining});

  final int setsRemaining;

  @override
  State<_RestCard> createState() => _RestCardState();
}

class _RestCardState extends State<_RestCard> {
  static const int _restSeconds = 60;

  late int _left = _restSeconds;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!mounted) return;
      setState(() {
        if (_left > 0) _left -= 1;
      });
      if (_left <= 0) timer.cancel();
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final done = _left <= 0;
    return Card(
      color: done ? scheme.primaryContainer : scheme.surfaceContainerHigh,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            Icon(done ? Icons.check_circle : Icons.timer_outlined),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                done
                    ? 'Відпочинок завершено. Залишилось підходів: '
                        '${widget.setsRemaining}'
                    : 'Відпочиньте $_left с перед наступним підходом',
              ),
            ),
          ],
        ),
      ),
    );
  }
}
