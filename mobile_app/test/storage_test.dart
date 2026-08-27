import 'package:flutter_test/flutter_test.dart';
import 'package:intelligent_rehab/storage/local_store.dart';
import 'package:intelligent_rehab/storage/session_record.dart';

/// Storage round-trips
void main() {
  group('SessionRecord', () {
    final record = SessionRecord(
      exerciseId: 'shoulder_abduction_seated',
      timestamp: DateTime(2026, 8, 21, 14, 2),
      usable: true,
      meanEqi: 84.2,
      grade: 4,
      nReps: 8,
      correctRatio: 0.75,
      fatigueTrend: -0.4,
      errorFrequency: {'jerky_movement': 0.25},
    );

    test('survives a json round trip with every field intact', () {
      final restored = SessionRecord.fromJson(record.toJson());
      expect(restored.exerciseId, record.exerciseId);
      expect(restored.timestamp, record.timestamp);
      expect(restored.usable, isTrue);
      expect(restored.meanEqi, closeTo(84.2, 1e-9));
      expect(restored.grade, 4);
      expect(restored.nReps, 8);
      expect(restored.correctRatio, closeTo(0.75, 1e-9));
      expect(restored.fatigueTrend, closeTo(-0.4, 1e-9));
      expect(restored.errorFrequency, {'jerky_movement': 0.25});
    });

    test('builds from an /assess/session response', () {
      final built = SessionRecord.fromAssessment('elbow_flexion_extension', {
        'usable': true,
        'mean_eqi': 71.5,
        'grade': 3,
        'n_reps': 6,
        'correct_ratio': 0.5,
        'fatigue_trend': -1.2,
        'error_frequency': {'trunk_lateral_lean': 0.33},
      });
      expect(built.exerciseId, 'elbow_flexion_extension');
      expect(built.meanEqi, closeTo(71.5, 1e-9));
      expect(built.errorFrequency, {'trunk_lateral_lean': 0.33});
    });

    /// a rejected recording has null aggregates
    test('handles a rejected session', () {
      final rejected = SessionRecord.fromAssessment('shoulder_flexion_seated', {
        'usable': false,
        'mean_eqi': null,
        'grade': null,
        'n_reps': 0,
        'rejection_reason_uk': 'Стан не підходить',
      });
      expect(rejected.usable, isFalse);
      expect(rejected.meanEqi, isNull);
      expect(rejected.rejectionUk, 'Стан не підходить');

      final restored = SessionRecord.fromJson(rejected.toJson());
      expect(restored.usable, isFalse);
      expect(restored.meanEqi, isNull);
    });

    test('an unparseable timestamp does not throw', () {
      final restored = SessionRecord.fromJson({
        'exercise_id': 'x',
        'timestamp': 'not a date',
      });
      expect(restored.timestamp.year, 1970);
    });

    test('the coach payload carries what adaptation.decide reads', () {
      final payload = record.toCoachJson();
      for (final key in [
        'exercise_id',
        'mean_eqi',
        'n_reps',
        'correct_ratio',
        'fatigue_trend',
        'error_frequency',
      ]) {
        expect(payload.containsKey(key), isTrue, reason: key);
      }
    });
  });

  group('Profile', () {
    test('round trips including doses and unlocked exercises', () {
      const profile = Profile(
        name: 'Ігор',
        age: 34,
        side: 'right',
        serviceUrl: 'http://192.168.0.10:8000',
        doses: {
          'shoulder_abduction_seated':
              Dose(romTarget: 95.25, reps: 8, sets: 2),
        },
        unlocked: ['elbow_flexion_extension', 'shoulder_abduction_seated'],
      );
      final restored = Profile.fromJson(profile.toJson());
      expect(restored.name, 'Ігор');
      expect(restored.age, 34);
      expect(restored.side, 'right');
      expect(restored.serviceUrl, 'http://192.168.0.10:8000');
      expect(restored.doses['shoulder_abduction_seated']!.romTarget,
          closeTo(95.25, 1e-9));
      expect(restored.unlocked.length, 2);
    });

    test('an empty store yields usable defaults', () {
      final restored = Profile.fromJson({});
      expect(restored.name, isEmpty);
      expect(restored.age, isNull);
      expect(restored.side, 'left');
      expect(restored.doses, isEmpty);
      expect(restored.unlocked, isEmpty);
    });

    test('clearing the age needs the explicit flag', () {
      const profile = Profile(age: 40);
      expect(profile.copyWith(age: null).age, 40);
      expect(profile.copyWith(clearAge: true).age, isNull);
    });
  });
}
