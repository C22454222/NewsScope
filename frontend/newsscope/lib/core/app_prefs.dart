import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Lightweight singleton that caches display-preference flags loaded from
/// SharedPreferences. Call [AppPrefs.load()] once at startup (or after any
/// settings change) then read the sync getters anywhere in the widget tree.
///
/// ArticleCard, HomeFeedTab, and CompareScreen all read from here so the
/// Show Credibility / Show Sentiment / Compact Cards settings take effect
/// immediately after the next [load()] call (triggered by SettingsScreen).
class AppPrefs {
  AppPrefs._();

  // ── Notifier ── rebuild-aware widgets subscribe to this ──────────────────
  static final ValueNotifier<int> notifier = ValueNotifier(0);

  // ── Cached values ─────────────────────────────────────────────────────────
  static bool _showCredibility = true;
  static bool _showSentiment = true;
  static bool _compactCards = false;

  static bool get showCredibility => _showCredibility;
  static bool get showSentiment => _showSentiment;
  static bool get compactCards => _compactCards;

  // ── Load from disk ─────────────────────────────────────────────────────────
  /// Call after app launch and after the user changes a display setting.
  /// Increments [notifier] so any [ValueListenableBuilder] wrappers rebuild.
  static Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    _showCredibility = prefs.getBool('show_credibility') ?? true;
    _showSentiment   = prefs.getBool('show_sentiment')   ?? true;
    _compactCards    = prefs.getBool('compact_cards')    ?? false;
    notifier.value++;
  }
}
