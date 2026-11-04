import 'package:flutter/material.dart';

import '../analysis/live_angles.dart';
import '../api/client.dart';
import '../storage/session_record.dart';
import 'summary_screen.dart';

/// Scoring a pre-recorded clip instead of the live camera
class VideoScreen extends StatefulWidget {
  const VideoScreen({
    super.key,
    required this.baseUrl,
    required this.exercise,
    required this.side,
    this.onSessionComplete,
  });

  final String baseUrl;
  final ExerciseInfo exercise;
  final BodySide side;

  /// Called once with the scored clip, which is stored as a session
  final Future<void> Function(SessionRecord)? onSessionComplete;

  @override
  State<VideoScreen> createState() => _VideoScreenState();
}

class _VideoScreenState extends State<VideoScreen> {
  late final RehabApi _api = RehabApi(widget.baseUrl);

  List<VideoInfo>? _videos;
  String? _error;
  String? _busy;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void dispose() {
    _api.close();
    super.dispose();
  }

  Future<void> _load() async {
    try {
      final videos = await _api.videos();
      if (mounted) setState(() => _videos = videos);
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    }
  }

  Future<void> _score(VideoInfo video) async {
    setState(() {
      _busy = video.relativePath;
      _error = null;
    });
    try {
      final session = await _api.assessVideo(
        exerciseId: widget.exercise.id,
        side: widget.side,
        relativePath: video.relativePath,
      );
      await widget.onSessionComplete?.call(
        SessionRecord.fromAssessment(widget.exercise.id, session.json),
      );
      if (!mounted) return;
      Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => SummaryScreen(session: session, exercise: widget.exercise),
      ));
    } on ApiException catch (e) {
      if (mounted) setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _busy = null);
    }
  }

  @override
  Widget build(BuildContext context) {
    final videos = _videos;
    return Scaffold(
      appBar: AppBar(title: const Text('Оцінити відео')),
      body: _error != null && videos == null
          ? _message(_error!, isError: true)
          : videos == null
              ? const Center(child: CircularProgressIndicator())
              : videos.isEmpty
                  ? _message(
                      'Сервіс запущено без каталогу відео.\n\n'
                      'Перезапустіть його з --video-dir <шлях>',
                    )
                  : ListView(
                      padding: const EdgeInsets.all(12),
                      children: [
                        if (_error != null)
                          Padding(
                            padding: const EdgeInsets.only(bottom: 12),
                            child: Text(
                              _error!,
                              style: TextStyle(
                                color: Theme.of(context).colorScheme.error,
                              ),
                            ),
                          ),
                        Text(
                          '${widget.exercise.nameUk} · '
                          '${widget.side == BodySide.left ? 'ліва' : 'права'} рука',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        const SizedBox(height: 8),
                        for (final v in videos)
                          Card(
                            child: ListTile(
                              title: Text(v.name),
                              subtitle: Text('${v.sizeMb.toStringAsFixed(1)} MB'),
                              trailing: _busy == v.relativePath
                                  ? const SizedBox(
                                      width: 20,
                                      height: 20,
                                      child: CircularProgressIndicator(strokeWidth: 2),
                                    )
                                  : const Icon(Icons.play_arrow),
                              // Disabled while any clip is scoring
                              onTap: _busy == null ? () => _score(v) : null,
                            ),
                          ),
                      ],
                    ),
    );
  }

  Widget _message(String text, {bool isError = false}) => Center(
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Text(
            text,
            textAlign: TextAlign.center,
            style: TextStyle(
              color: isError ? Theme.of(context).colorScheme.error : null,
            ),
          ),
        ),
      );
}
