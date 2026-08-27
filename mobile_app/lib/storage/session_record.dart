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

  /// Building from an `/assess/session` response body
  factory SessionRecord.fromAssessment(
    String exerciseId,
    Map<String, dynamic> body, {
    DateTime? at,
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
      };

  /// The subset `/coach` accepts. Rejected sessions are sent too
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
