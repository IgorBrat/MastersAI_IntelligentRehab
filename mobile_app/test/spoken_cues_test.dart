import 'package:flutter_test/flutter_test.dart';
import 'package:intelligent_rehab/analysis/spoken_cues.dart';

/// Recording spoken cues instead of using the platform channel
class FakeSpeaker implements CueSpeaker {
  final List<String> spoken = [];
  int stops = 0;

  @override
  Future<void> configure() async {}

  @override
  Future<void> speak(String text) async => spoken.add(text);

  @override
  Future<void> stop() async => stops++;
}

void main() {
  late FakeSpeaker speaker;
  late SpokenCues cues;
  late DateTime clock;

  setUp(() {
    speaker = FakeSpeaker();
    cues = SpokenCues(speaker: speaker, minGap: const Duration(seconds: 2));
    clock = DateTime(2026, 9, 16, 12, 0, 0);
    cues.now = () => clock;
  });

  void advance(int ms) => clock = clock.add(Duration(milliseconds: ms));

  test('speaks the first cue immediately', () {
    expect(cues.say('Вище'), isTrue);
    expect(speaker.spoken, ['Вище']);
  });

  test('does not repeat the cue that is already current', () {
    cues.say('Вище');
    advance(5000);
    expect(cues.say('Вище'), isFalse, reason: 'still the same instruction');
    expect(speaker.spoken.length, 1);
  });

  test('holds a different cue until the gap has elapsed', () {
    cues.say('Вище');
    advance(500);
    expect(cues.say('Повільно опускайте'), isFalse);
    advance(1600);
    expect(cues.say('Повільно опускайте'), isTrue);
    expect(speaker.spoken, ['Вище', 'Повільно опускайте']);
  });

  test('an urgent cue bypasses the gap', () {
    cues.say('Вище');
    advance(100);
    expect(
      cues.say('Тулуб рівно', urgent: true),
      isTrue,
      reason: 'a late compensation warning is useless',
    );
  });

  test('an urgent cue repeats when it is still current, but not every frame', () {
    cues = SpokenCues(
      speaker: speaker,
      minGap: const Duration(seconds: 2),
      urgentMinGap: const Duration(milliseconds: 2500),
    );
    cues.now = () => clock;

    expect(cues.say('Тулуб рівно', urgent: true), isTrue);
    for (var i = 0; i < 70; i++) {
      advance(33);
      cues.say('Тулуб рівно', urgent: true);
    }
    expect(speaker.spoken.length, 1, reason: '2.3s of frames, gap not yet up');

    advance(300);
    expect(cues.say('Тулуб рівно', urgent: true), isTrue);
    expect(speaker.spoken.length, 2);
  });

  test('an ordinary cue does not hold back an urgent one', () {
    cues.say('Вище');
    advance(100);
    expect(cues.say('Тепер повільно опускайте руку', urgent: true), isTrue);
    advance(100);
    expect(cues.say('Тулуб рівно', urgent: true), isFalse,
        reason: 'urgent lines are spaced against each other');
  });

  test('clearHistory lets the same instruction be given for the next rep', () {
    cues.say('Піднімайте руку');
    advance(5000);
    expect(cues.say('Піднімайте руку'), isFalse);
    cues.clearHistory();
    expect(cues.say('Піднімайте руку'), isTrue);
  });

  test('disabled cues are silent', () {
    cues.enabled = false;
    expect(cues.say('Вище'), isFalse);
    expect(speaker.spoken, isEmpty);
  });

  test('an empty cue is never spoken', () {
    expect(cues.say(''), isFalse);
    expect(speaker.spoken, isEmpty);
  });
}
