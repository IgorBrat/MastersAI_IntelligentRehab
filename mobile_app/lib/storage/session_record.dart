/// A supervising specialist's verdict on a recorded set, the pilot's reference label
enum SupervisorLabel {
  none('none'),
  correct('correct'),
  incorrect('incorrect');

  const SupervisorLabel(this.wire);
  final String wire;

  static SupervisorLabel parse(String? raw) => switch (raw) {
        'correct' => SupervisorLabel.correct,
        'incorrect' => SupervisorLabel.incorrect,
        _ => SupervisorLabel.none,
      };

  String get labelUk => switch (this) {
        SupervisorLabel.correct => 'правильно',
        SupervisorLabel.incorrect => 'з помилками',
        SupervisorLabel.none => 'не оцінено',
      };
}

/// One completed session, as stored on the phone and replayed to the coach
class SessionRecord {
  const SessionRecord({
    required this.exerciseId,
    required this.timestamp,
    required this.usable,
    this.meanEqi,
    this.grade,
    this.nReps = 0,
    this.correctRatio,
    this.fatigueTrend = 0.0,
    this.errorFrequency = const {},
    this.rejectionUk,
    this.label = SupervisorLabel.none,
    this.note = '',
    this.recordingId,
  });

  final String exerciseId;
  final DateTime timestamp;

  /// False for a recording the engine refused to score
  final bool usable;

  final double? meanEqi;
  final int? grade;
  final int nReps;
  final double? correctRatio;
  final double fatigueTrend;

  /// Error slug -> share of repetitions it fired in
  final Map<String, double> errorFrequency;

  final String? rejectionUk;

  final SupervisorLabel label;

  /// Free text from whoever labelled it
  final String note;

  /// Stored raw-landmark file name, or null if the recording was not kept
  final String? recordingId;

  SessionRecord copyWith({
    SupervisorLabel? label,
    String? note,
    String? recordingId,
  }) =>
      SessionRecord(
        exerciseId: exerciseId,
        timestamp: timestamp,
        usable: usable,
        meanEqi: meanEqi,
        grade: grade,
        nReps: nReps,
        correctRatio: correctRatio,
        fatigueTrend: fatigueTrend,
        errorFrequency: errorFrequency,
        rejectionUk: rejectionUk,
        label: label ?? this.label,
        note: note ?? this.note,
        recordingId: recordingId ?? this.recordingId,
      );

  /// Builds from a session assessment body
  factory SessionRecord.fromAssessment(
    String exerciseId,
    Map<String, dynamic> body, {
    DateTime? at,
    String? recordingId,
  }) {
    final errors = <String, double>{};
    final raw = body['error_frequency'];
    if (raw is Map) {
      raw.forEach((key, value) {
        if (value is num) errors[key.toString()] = value.toDouble();
      });
    }
    return SessionRecord(
      exerciseId: exerciseId,
      timestamp: at ?? DateTime.now(),
      usable: body['usable'] == true,
      meanEqi: (body['mean_eqi'] as num?)?.toDouble(),
      grade: (body['grade'] as num?)?.toInt(),
      nReps: (body['n_reps'] as num?)?.toInt() ?? 0,
      correctRatio: (body['correct_ratio'] as num?)?.toDouble(),
      fatigueTrend: (body['fatigue_trend'] as num?)?.toDouble() ?? 0.0,
      errorFrequency: errors,
      rejectionUk: body['rejection_reason_uk'] as String?,
      recordingId: recordingId,
    );
  }

  factory SessionRecord.fromJson(Map<String, dynamic> json) => SessionRecord(
        exerciseId: json['exercise_id'] as String,
        timestamp:
            DateTime.tryParse(json['timestamp'] as String? ?? '') ?? DateTime(1970),
        usable: json['usable'] as bool? ?? true,
        meanEqi: (json['mean_eqi'] as num?)?.toDouble(),
        grade: (json['grade'] as num?)?.toInt(),
        nReps: (json['n_reps'] as num?)?.toInt() ?? 0,
        correctRatio: (json['correct_ratio'] as num?)?.toDouble(),
        fatigueTrend: (json['fatigue_trend'] as num?)?.toDouble() ?? 0.0,
        errorFrequency: {
          for (final e in (json['error_frequency'] as Map? ?? {}).entries)
            e.key.toString(): (e.value as num).toDouble(),
        },
        rejectionUk: json['rejection_reason_uk'] as String?,
        label: SupervisorLabel.parse(json['label'] as String?),
        note: json['note'] as String? ?? '',
        recordingId: json['recording_id'] as String?,
      );

  Map<String, dynamic> toJson() => {
        'exercise_id': exerciseId,
        'timestamp': timestamp.toIso8601String(),
        'usable': usable,
        'mean_eqi': meanEqi,
        'grade': grade,
        'n_reps': nReps,
        'correct_ratio': correctRatio,
        'fatigue_trend': fatigueTrend,
        'error_frequency': errorFrequency,
        'rejection_reason_uk': rejectionUk,
        'label': label.wire,
        'note': note,
        'recording_id': recordingId,
      };

  /// Fields the coach uses. [label] is left out so the dose depends only on measurements
  Map<String, dynamic> toCoachJson() => {
        'exercise_id': exerciseId,
        'timestamp': timestamp.toIso8601String(),
        'mean_eqi': meanEqi,
        'n_reps': nReps,
        'correct_ratio': correctRatio,
        'fatigue_trend': fatigueTrend,
        'error_frequency': errorFrequency,
      };
}
