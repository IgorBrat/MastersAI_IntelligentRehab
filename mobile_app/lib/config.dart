class Config {
  /// Fallback capture rate, used only when frame timestamps cannot give one
  static const double fps = 30.0;
  /// Frames kept per repetition window before the oldest is dropped
  static const int maxFramesPerRep = 600;
}
