import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../storage/session_record.dart';

/// Past sessions, newest first, from the phone's own store
class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key, required this.state});

  final AppState state;

  /// Exercise names, when the service is reachable. Falls back to the raw id: a stored
  /// session must remain readable after the protocol is renamed or while offline
  String _name(String exerciseId) {
    for (final e in state.exercises) {
      if (e.id == exerciseId) return e.nameUk;
    }
    return exerciseId;
  }

  String _cue(String slug) {
    for (final e in state.exercises) {
      final text = e.watchedErrors[slug];
      if (text != null && text.isNotEmpty) return text;
    }
    return slug;
  }

  @override
  Widget build(BuildContext context) {
    final sessions = state.history.reversed.toList();
    if (sessions.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(28),
          child: Text(
            'Ще немає занять.\n\nЗавершіть перше заняття у вкладці «Вправи» -\n'
            'воно збережеться тут, на цьому телефоні.',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _Summary(state: state),
        const SizedBox(height: 18),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text('Заняття', style: Theme.of(context).textTheme.titleMedium),
            TextButton.icon(
              onPressed: () => _confirmClear(context),
              icon: const Icon(Icons.delete_outline, size: 18),
              label: const Text('Очистити'),
            ),
          ],
        ),
        for (final session in sessions)
          _SessionTile(session: session, name: _name(session.exerciseId), cue: _cue),
      ],
    );
  }

  Future<void> _confirmClear(BuildContext context) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Очистити історію?'),
        content: const Text(
          'Усі збережені заняття буде видалено з цього телефону. '
          'Це неможливо скасувати, і тренер втратить дані для оцінки прогресу.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: const Text('Скасувати'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: const Text('Видалити'),
          ),
        ],
      ),
    );
    if (confirmed == true) await state.clearHistory();
  }
}

/// Aggregate figures across stored sessions
class _Summary extends StatelessWidget {
  const _Summary({required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    final scored = [
      for (final s in state.history)
        if (s.usable && s.meanEqi != null) s,
    ];
    final total = state.history.length;
    final mean = scored.isEmpty
        ? null
        : scored.map((s) => s.meanEqi!).reduce((a, b) => a + b) / scored.length;

    // Trend over the last five scored sessions against the five before them
    double? delta;
    if (scored.length >= 4) {
      final recent = scored.sublist(scored.length - (scored.length ~/ 2));
      final earlier = scored.sublist(0, scored.length ~/ 2);
      double avg(List<SessionRecord> xs) =>
          xs.map((s) => s.meanEqi!).reduce((a, b) => a + b) / xs.length;
      delta = avg(recent) - avg(earlier);
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            _Stat(label: 'занять', value: '$total'),
            _Stat(
              label: 'середній EQI',
              value: mean == null ? '-' : mean.toStringAsFixed(0),
            ),
            _Stat(
              label: 'динаміка',
              value: delta == null
                  ? '-'
                  : '${delta >= 0 ? '+' : ''}${delta.toStringAsFixed(1)}',
              positive: delta == null ? null : delta >= 0,
            ),
          ],
        ),
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({required this.label, required this.value, this.positive});

  final String label;
  final String value;
  final bool? positive;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final colour = positive == null
        ? null
        : (positive! ? scheme.primary : scheme.error);
    return Column(
      children: [
        Text(value,
            style: Theme.of(context)
                .textTheme
                .headlineSmall
                ?.copyWith(color: colour)),
        Text(label, style: Theme.of(context).textTheme.bodySmall),
      ],
    );
  }
}

class _SessionTile extends StatelessWidget {
  const _SessionTile({
    required this.session,
    required this.name,
    required this.cue,
  });

  final SessionRecord session;
  final String name;
  final String Function(String slug) cue;

  String get _when {
    final t = session.timestamp;
    String two(int v) => v.toString().padLeft(2, '0');
    return '${two(t.day)}.${two(t.month)}.${t.year}, ${two(t.hour)}:${two(t.minute)}';
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    // A rejected recording is shown
    if (!session.usable) {
      return Card(
        child: ListTile(
          leading: Icon(Icons.videocam_off, color: scheme.error),
          title: Text(name),
          subtitle: Text('$_when\n${session.rejectionUk ?? 'Запис не оцінено'}'),
          isThreeLine: true,
        ),
      );
    }

    final errors = session.errorFrequency.entries.where((e) => e.value > 0).toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    return Card(
      child: ExpansionTile(
        leading: CircleAvatar(
          backgroundColor: scheme.primaryContainer,
          child: Text(
            session.meanEqi?.toStringAsFixed(0) ?? '-',
            style: TextStyle(color: scheme.onPrimaryContainer, fontSize: 13),
          ),
        ),
        title: Text(name),
        subtitle: Text('$_when · ${session.nReps} повторень'
            '${session.grade != null ? ' · оцінка ${session.grade}/4' : ''}'),
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (session.correctRatio != null)
                  Text('Правильних повторень: '
                      '${(session.correctRatio! * 100).toStringAsFixed(0)}%'),
                Text('Зміна якості за підхід: '
                    '${session.fatigueTrend >= 0 ? '+' : ''}'
                    '${session.fatigueTrend.toStringAsFixed(2)} EQI/повторення'),
                if (errors.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  const Text('Що виправити:'),
                  const SizedBox(height: 4),
                  for (final e in errors)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 4),
                      child: Text(
                        '• ${cue(e.key)} - ${(e.value * 100).toStringAsFixed(0)}%',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}
