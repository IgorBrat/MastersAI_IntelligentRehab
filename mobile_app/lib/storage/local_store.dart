import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import 'session_record.dart';

/// Local persistence: one JSON file for history, one for the profile and dose
class LocalStore {
  LocalStore._();

  static const _historyFile = 'history.json';
  static const _profileFile = 'profile.json';

  static Future<File> _file(String name) async {
    final dir = await getApplicationDocumentsDirectory();
    return File('${dir.path}/$name');
  }

  static Future<void> _writeJson(String name, Object payload) async {
    final target = await _file(name);
    final temp = File('${target.path}.tmp');
    await temp.writeAsString(jsonEncode(payload), flush: true);
    await temp.rename(target.path);
  }

  static Future<Object?> _readJson(String name) async {
    try {
      final file = await _file(name);
      if (!await file.exists()) return null;
      final text = await file.readAsString();
      if (text.trim().isEmpty) return null;
      return jsonDecode(text);
    } catch (_) {
      return null;
    }
  }

  /// All stored sessions, newest last
  static Future<List<SessionRecord>> history() async {
    final raw = await _readJson(_historyFile);
    if (raw is! List) return [];
    final out = <SessionRecord>[];
    for (final item in raw) {
      if (item is Map) {
        try {
          out.add(SessionRecord.fromJson(item.cast<String, dynamic>()));
        } catch (_) {
          // Skip a single unreadable row rather than discarding the file
        }
      }
    }
    out.sort((a, b) => a.timestamp.compareTo(b.timestamp));
    return out;
  }

  static Future<void> addSession(SessionRecord record) async {
    final all = await history()..add(record);
    await _writeJson(_historyFile, [for (final s in all) s.toJson()]);
  }

  static Future<void> clearHistory() => _writeJson(_historyFile, const []);

  /// Replacing one stored session, matched on its timestamp
  static Future<void> updateSession(SessionRecord updated) async {
    final all = await history();
    final out = [
      for (final s in all)
        if (s.timestamp == updated.timestamp) updated else s,
    ];
    await _writeJson(_historyFile, [for (final s in out) s.toJson()]);
  }

  static Future<Profile> profile() async {
    final raw = await _readJson(_profileFile);
    if (raw is! Map) return const Profile();
    return Profile.fromJson(raw.cast<String, dynamic>());
  }

  static Future<void> saveProfile(Profile profile) =>
      _writeJson(_profileFile, profile.toJson());

  // Region - Recordings

  static Future<Directory> _recordingsDir() async {
    final dir = await getApplicationDocumentsDirectory();
    final out = Directory('${dir.path}/recordings');
    if (!await out.exists()) await out.create(recursive: true);
    return out;
  }

  /// Storing the raw landmark payload a session was scored from
  static Future<String> saveRecording(
    String exerciseId,
    DateTime at,
    Object payload,
  ) async {
    final id = '${exerciseId}_${at.toIso8601String().replaceAll(':', '-')}.json';
    final dir = await _recordingsDir();
    final target = File('${dir.path}/$id');
    final temp = File('${target.path}.tmp');
    await temp.writeAsString(jsonEncode(payload), flush: true);
    await temp.rename(target.path);
    return id;
  }

  /// Every stored recording, newest first, with its size
  static Future<List<({String id, int bytes})>> recordings() async {
    final dir = await _recordingsDir();
    final out = <({String id, int bytes})>[];
    await for (final entry in dir.list()) {
      if (entry is File && entry.path.endsWith('.json')) {
        out.add((
          id: entry.uri.pathSegments.last,
          bytes: await entry.length(),
        ));
      }
    }
    out.sort((a, b) => b.id.compareTo(a.id));
    return out;
  }

  static Future<File?> recordingFile(String id) async {
    final dir = await _recordingsDir();
    final file = File('${dir.path}/$id');
    return await file.exists() ? file : null;
  }

  /// Writing one file holding the history, the profile and every recording
  static Future<File> exportBundle() async {
    final dir = await getApplicationDocumentsDirectory();
    final stored = await history();
    final payloads = <String, Object?>{};
    for (final entry in await recordings()) {
      final file = await recordingFile(entry.id);
      if (file == null) continue;
      try {
        payloads[entry.id] = jsonDecode(await file.readAsString());
      } catch (_) {
        // A corrupt recording must not cost the export everything else
        payloads[entry.id] = {'error': 'unreadable'};
      }
    }

    final bundle = {
      'format': 'intelligent_rehab/export@1',
      'exported_at': DateTime.now().toIso8601String(),
      'profile': (await profile()).toJson(),
      'sessions': [for (final s in stored) s.toJson()],
      'recordings': payloads,
    };

    final target = File(
      '${dir.path}/rehab_export_'
      '${DateTime.now().toIso8601String().replaceAll(':', '-')}.json',
    );
    await target.writeAsString(jsonEncode(bundle), flush: true);
    return target;
  }
}

/// The patient's own details, the service address, and the dose in force
class Profile {
  const Profile({
    this.name = '',
    this.age,
    this.side = 'left',
    this.serviceUrl = '',
    this.doses = const {},
    this.unlocked = const [],
  });

  final String name;
  final int? age;

  final String side;

  final String serviceUrl;

  /// exercise_id -> current dose, as returned by `/coach`
  final Map<String, Dose> doses;

  /// Exercises the coach has already unlocked, so it does not offer one twice
  final List<String> unlocked;

  Profile copyWith({
    String? name,
    int? age,
    bool clearAge = false,
    String? side,
    String? serviceUrl,
    Map<String, Dose>? doses,
    List<String>? unlocked,
  }) =>
      Profile(
        name: name ?? this.name,
        age: clearAge ? null : (age ?? this.age),
        side: side ?? this.side,
        serviceUrl: serviceUrl ?? this.serviceUrl,
        doses: doses ?? this.doses,
        unlocked: unlocked ?? this.unlocked,
      );

  factory Profile.fromJson(Map<String, dynamic> json) => Profile(
        name: json['name'] as String? ?? '',
        age: (json['age'] as num?)?.toInt(),
        side: json['side'] as String? ?? 'left',
        serviceUrl: json['service_url'] as String? ?? '',
        doses: {
          for (final e in (json['doses'] as Map? ?? {}).entries)
            e.key.toString(): Dose.fromJson((e.value as Map).cast<String, dynamic>()),
        },
        unlocked: [
          for (final v in (json['unlocked'] as List? ?? [])) v.toString(),
        ],
      );

  Map<String, dynamic> toJson() => {
        'name': name,
        'age': age,
        'side': side,
        'service_url': serviceUrl,
        'doses': {for (final e in doses.entries) e.key: e.value.toJson()},
        'unlocked': unlocked,
      };
}

/// The dose in force for one exercise - what the coach last decided
class Dose {
  const Dose({required this.romTarget, required this.reps, required this.sets});

  final double romTarget;
  final int reps;
  final int sets;

  factory Dose.fromJson(Map<String, dynamic> json) => Dose(
        romTarget: (json['rom_target'] as num).toDouble(),
        reps: (json['reps'] as num).toInt(),
        sets: (json['sets'] as num).toInt(),
      );

  Map<String, dynamic> toJson() =>
      {'rom_target': romTarget, 'reps': reps, 'sets': sets};
}
