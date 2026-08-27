import 'package:flutter/material.dart';

import '../state/app_state.dart';


class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key, required this.state});

  final AppState state;

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  late final TextEditingController _name;
  late final TextEditingController _age;
  late final TextEditingController _url;
  late String _side;
  bool _dirty = false;
  bool _checking = false;

  AppState get state => widget.state;

  @override
  void initState() {
    super.initState();
    final profile = state.profile;
    _name = TextEditingController(text: profile.name);
    _age = TextEditingController(text: profile.age?.toString() ?? '');
    _url = TextEditingController(text: profile.serviceUrl);
    _side = profile.side;
    for (final c in [_name, _age, _url]) {
      c.addListener(() {
        if (!_dirty) setState(() => _dirty = true);
      });
    }
  }

  @override
  void dispose() {
    _name.dispose();
    _age.dispose();
    _url.dispose();
    super.dispose();
  }

  Future<void> _save({bool reconnect = true}) async {
    final ageText = _age.text.trim();
    final parsedAge = ageText.isEmpty ? null : int.tryParse(ageText);
    await state.saveProfile(
      state.profile.copyWith(
        name: _name.text.trim(),
        age: parsedAge,
        clearAge: ageText.isEmpty,
        side: _side,
        serviceUrl: _url.text.trim(),
      ),
    );
    if (!mounted) return;
    setState(() => _dirty = false);
    if (reconnect) {
      setState(() => _checking = true);
      await state.connect();
      if (mounted) setState(() => _checking = false);
    }
  }

  Future<void> _changeSide(String next) async {
    // Changing the affected side mid-programme makes previously stored sessions
    // incomparable: they measured the other arm. Say so rather than silently letting the
    // history mix two limbs.
    if (state.history.isNotEmpty && next != state.profile.side) {
      final confirmed = await showDialog<bool>(
        context: context,
        builder: (dialogContext) => AlertDialog(
          title: const Text('Змінити руку?'),
          content: Text(
            'В історії вже є ${state.history.length} занять, записаних для '
            '${state.profile.side == 'left' ? 'лівої' : 'правої'} руки. '
            'Нові заняття будуть для іншої руки, і порівнювати їх напряму не можна.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(dialogContext).pop(false),
              child: const Text('Скасувати'),
            ),
            FilledButton(
              onPressed: () => Navigator.of(dialogContext).pop(true),
              child: const Text('Змінити'),
            ),
          ],
        ),
      );
      if (confirmed != true) return;
    }
    setState(() {
      _side = next;
      _dirty = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final info = state.serviceInfo;

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('Пацієнт', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 4),
        Text(
          'Використовується лише для звіту фахівцеві. Дані не залишають телефон.',
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: 14),
        TextField(
          controller: _name,
          decoration: const InputDecoration(
            labelText: 'Ім\'я',
            border: OutlineInputBorder(),
          ),
          textCapitalization: TextCapitalization.words,
        ),
        const SizedBox(height: 12),
        TextField(
          controller: _age,
          decoration: const InputDecoration(
            labelText: 'Вік',
            border: OutlineInputBorder(),
          ),
          keyboardType: TextInputType.number,
        ),
        const SizedBox(height: 22),
        Text('Уражена рука', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 4),
        Text(
          'Потрібно системі: визначає, яке плече вимірюється.',
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: 10),
        SegmentedButton<String>(
          segments: const [
            ButtonSegment(value: 'left', label: Text('Ліва')),
            ButtonSegment(value: 'right', label: Text('Права')),
          ],
          selected: {_side},
          onSelectionChanged: (s) => _changeSide(s.first),
        ),
        const SizedBox(height: 22),
        Text('Сервіс', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 10),
        TextField(
          controller: _url,
          decoration: const InputDecoration(
            labelText: 'Адреса',
            helperText: 'LAN-адреса ноутбука, напр. http://192.168.0.10:8000 - не localhost',
            helperMaxLines: 2,
            border: OutlineInputBorder(),
          ),
          keyboardType: TextInputType.url,
          autocorrect: false,
        ),
        const SizedBox(height: 12),
        if (_checking)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 8),
            child: LinearProgressIndicator(),
          )
        else if (state.connected)
          _Status(
            icon: Icons.cloud_done,
            colour: scheme.primary,
            title: 'Підключено - ${state.exercises.length} вправ у протоколі',
            // Which thresholds are in force changes what the same movement scores, so a
            // stored report is uninterpretable without it.
            subtitle: info == null
                ? null
                : 'Пороги: ${info['thresholds'] != null ? 'каліброовані' : 'протокольні'}'
                    ' · вільний темп: ${info['self_paced'] == true ? 'так' : 'ні'}',
          )
        else
          _Status(
            icon: Icons.cloud_off,
            colour: scheme.error,
            title: 'Немає зв\'язку',
            subtitle: state.connectionError,
          ),
        const SizedBox(height: 20),
        FilledButton.icon(
          onPressed: _dirty ? () => _save() : null,
          icon: const Icon(Icons.save_outlined),
          label: const Text('Зберегти'),
        ),
        const SizedBox(height: 28),
        _Programme(state: state),
      ],
    );
  }
}

class _Status extends StatelessWidget {
  const _Status({
    required this.icon,
    required this.colour,
    required this.title,
    this.subtitle,
  });

  final IconData icon;
  final Color colour;
  final String title;
  final String? subtitle;

  @override
  Widget build(BuildContext context) => Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: colour, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: TextStyle(color: colour)),
                if (subtitle != null && subtitle!.isNotEmpty)
                  Text(subtitle!,
                      style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          ),
        ],
      );
}

/// The current programme: which exercises are open and at what dose
class _Programme extends StatelessWidget {
  const _Programme({required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    if (state.profile.doses.isEmpty && state.profile.unlocked.isEmpty) {
      return const SizedBox.shrink();
    }
    String name(String id) {
      for (final e in state.exercises) {
        if (e.id == id) return e.nameUk;
      }
      return id;
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Програма', style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 10),
            for (final id in state.profile.unlocked)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  children: [
                    Expanded(child: Text(name(id))),
                    Text(
                      state.profile.doses[id] == null
                          ? 'за протоколом'
                          : '${state.profile.doses[id]!.romTarget.toStringAsFixed(0)}° · '
                              '${state.profile.doses[id]!.reps} повт.',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}
