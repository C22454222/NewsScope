import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Lightweight singleton that caches display-preference flags loaded
/// from SharedPreferences.
///
/// Call [AppPrefs.load()] once at startup, and again after any settings
/// change. Sync getters can then be read anywhere in the widget tree
/// without awaiting.
///
/// Widgets that need to rebuild on change should wrap with
/// [ValueListenableBuilder] and listen to [AppPrefs.notifier].
class AppPrefs {
  AppPrefs._();

  /// Incremented after every [load()] so listeners can trigger a rebuild.
  static final ValueNotifier<int> notifier = ValueNotifier(0);

  // Cached values, populated by [load()].
  static bool _showCredibility = true;
  static bool _showSentiment = true;
  static bool _compactCards = false;

  static bool get showCredibility => _showCredibility;
  static bool get showSentiment => _showSentiment;
  static bool get compactCards => _compactCards;

  /// Reads all display preferences from disk and bumps [notifier].
  ///
  /// Safe to call multiple times. Always awaited at app startup before
  /// [runApp] so the first frame reflects the user's saved settings.
  static Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    _showCredibility = prefs.getBool('show_credibility') ?? true;
    _showSentiment   = prefs.getBool('show_sentiment')   ?? true;
    _compactCards    = prefs.getBool('compact_cards')    ?? false;
    notifier.value++;
  }
}
