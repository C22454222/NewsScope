// lib/utils/score_helpers.dart
import 'package:flutter/material.dart';

/// Shared colour and label helpers for bias, sentiment, and credibility scores.
///
/// Centralises logic previously duplicated across article_detail_screen,
/// home_screen, compare_screen, and profile_screen. Import this file
/// instead of re-implementing the same thresholds per screen.

Color getBiasColor(double? score) {
  if (score == null) return Colors.grey[300]!;
  if (score < -0.3) return Colors.blue[700]!;
  if (score > 0.3) return Colors.red[700]!;
  return Colors.purple[400]!;
}

/// Five-band label — matches backend _SOURCE_BIAS_MAP granularity.
String getBiasLabel(double? score) {
  if (score == null) return 'Unscored';
  if (score < -0.5) return 'Left';
  if (score < -0.2) return 'Center-Left';
  if (score < 0.2) return 'Center';
  if (score < 0.5) return 'Center-Right';
  return 'Right';
}

/// Three-band label for list/card contexts where space is limited.
String getBiasLabelShort(double? score) {
  if (score == null) return 'Unscored';
  if (score < -0.3) return 'Left';
  if (score > 0.3) return 'Right';
  return 'Center';
}

Color getSentimentColor(double? score) {
  if (score == null) return Colors.grey;
  if (score > 0.1) return Colors.green[700]!;
  if (score < -0.1) return Colors.orange[800]!;
  return Colors.grey[600]!;
}

String getSentimentLabel(double? score) {
  if (score == null) return '--';
  if (score > 0.1) return 'Positive';
  if (score < -0.1) return 'Negative';
  return 'Neutral';
}

Color getCredibilityColor(double? score) {
  if (score == null) return Colors.grey;
  if (score >= 75) return Colors.green[700]!;
  if (score >= 50) return Colors.orange[700]!;
  return Colors.red[700]!;
}

String getCredibilityLabel(double? score) {
  if (score == null) return 'Unverified';
  if (score >= 75) return 'Credible';
  if (score >= 50) return 'Mixed';
  return 'Questionable';
}

/// Map PolitiFact ruling strings to emoji indicators.
String getRulingEmoji(String ruling) {
  final r = ruling.toLowerCase();
  if (r.contains('true') && !r.contains('mostly')) return '✅';
  if (r.contains('mostly true')) return '🟢';
  if (r.contains('half')) return '🟡';
  if (r.contains('mostly false')) return '🟠';
  if (r.contains('false') || r.contains('pants')) return '❌';
  return '❓';
}

/// Title-case a raw category string from the backend.
String formatCategory(String? category) {
  if (category == null || category.isEmpty) return 'General';
  final c = category.toLowerCase();
  return c[0].toUpperCase() + c.substring(1);
}
