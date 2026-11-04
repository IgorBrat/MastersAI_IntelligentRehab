import 'package:flutter_tts/flutter_tts.dart';

/// Anything that can speak a line; lets the throttle be tested without TTS
abstract class CueSpeaker {
  Future<void> configure();
  Future<void> speak(String text);
  Future<void> stop();
}

/// Ukrainian text-to-speech through the platform engine
class TtsSpeaker implements CueSpeaker {
  TtsSpeaker({this.language = 'uk-UA', this.rate = 0.45});

  final String language;
  final double rate;
  final FlutterTts _tts = FlutterTts();

  @override
  Future<void> configure() async {
    try {
      await _tts.setLanguage(language);
      await _tts.setSpeechRate(rate);
      await _tts.awaitSpeakCompletion(true);
    } catch (_) {
      // A missing uk-UA voice is not fatal; the on-screen cue remains
    }
  }

  @override
  Future<void> speak(String text) async {
    try {
      await _tts.speak(text);
    } catch (_) {
      // Speech failure never interrupts a set
    }
  }

  @override
  Future<void> stop() async {
    try {
      await _tts.stop();
    } catch (_) {}
  }
}

/// Speaking corrective cues, throttled so they stay intelligible.
/// The same line is not repeated, a new line waits [minGap], and an urgent
/// line skips both but waits [urgentMinGap] after the previous urgent one
class SpokenCues {
  SpokenCues({
    CueSpeaker? speaker,
    this.minGap = const Duration(milliseconds: 1800),
    this.urgentMinGap = const Duration(milliseconds: 2500),
    this.enabled = true,
  }) : _speaker = speaker ?? TtsSpeaker();

  final CueSpeaker _speaker;
  final Duration minGap;
  final Duration urgentMinGap;

  bool enabled;

  String? _lastSpoken;
  DateTime? _lastAt;
  DateTime? _lastUrgentAt;

  /// Injectable clock for tests
  DateTime Function() now = DateTime.now;

  Future<void> init() => _speaker.configure();

  /// Saying [text] if the throttle allows it. Returns whether it was spoken
  bool say(String text, {bool urgent = false}) {
    if (!enabled || text.isEmpty) return false;

    final at = now();
    if (text == _lastSpoken && !urgent) return false;

    final last = urgent ? _lastUrgentAt : _lastAt;
    final gap = urgent ? urgentMinGap : minGap;
    if (last != null && at.difference(last) < gap) return false;
    if (urgent) _lastUrgentAt = at;

    _lastSpoken = text;
    _lastAt = at;
    _speaker.speak(text);
    return true;
  }

  /// Forgetting the last line, so the next rep's first cue is spoken again
  void clearHistory() => _lastSpoken = null;

  Future<void> stop() => _speaker.stop();
}
