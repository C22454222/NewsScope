import 'package:flutter/material.dart';

// ── Bias ──────────────────────────────────────────────────────────────────────

Color getBiasColor(double? score) {
  if (score == null) return Colors.grey;
  if (score > 0.3) return Colors.red;
  if (score < -0.3) return Colors.blue;
  return Colors.green;
}

String getBiasLabelShort(double? score) {
  if (score == null) return 'Unknown';
  if (score > 0.3) return 'Right';
  if (score < -0.3) return 'Left';
  return 'Centre';
}

String getBiasLabel(double? score) {
  if (score == null) return 'Unknown';
  if (score > 0.6) return 'Far Right';
  if (score > 0.3) return 'Right';
  if (score > 0.1) return 'Centre Right';
  if (score >= -0.1) return 'Centre';
  if (score >= -0.3) return 'Centre Left';
  if (score >= -0.6) return 'Left';
  return 'Far Left';
}

// ── Sentiment ─────────────────────────────────────────────────────────────────

Color getSentimentColor(double? score) {
  if (score == null) return Colors.grey;
  if (score > 0.1) return Colors.green;
  if (score < -0.1) return Colors.red;
  return Colors.orange;
}

String getSentimentLabel(double? score) {
  if (score == null) return 'Neutral';
  if (score > 0.1) return 'Positive';
  if (score < -0.1) return 'Negative';
  return 'Neutral';
}

// ── Credibility ───────────────────────────────────────────────────────────────

/// Green >= 70 (reliable), Orange >= 40 (mixed), Red < 40 (low credibility).
Color getCredibilityColor(double? score) {
  if (score == null) return Colors.grey;
  if (score >= 70) return Colors.green;
  if (score >= 40) return Colors.orange;
  return Colors.red;
}

String getCredibilityLabel(double? score) {
  if (score == null) return 'Unverified';
  if (score >= 70) return 'Reliable';
  if (score >= 40) return 'Mixed';
  return 'Low';
}

/// Maps PolitiFact ruling strings to emoji indicators.
/// Used by ArticleDetailScreen._buildFactCheckTile.
String getRulingEmoji(String ruling) {
  final r = ruling.toLowerCase();
  if (r.contains('true') && !r.contains('mostly')) return '✅';
  if (r.contains('mostly true')) return '🟢';
  if (r.contains('half')) return '🟡';
  if (r.contains('mostly false')) return '🟠';
  if (r.contains('false') || r.contains('pants')) return '❌';
  return '❓';
}

// ── Category ──────────────────────────────────────────────────────────────────

/// Capitalises the first letter of a category string for display.
/// Returns empty string for null or empty input.
String formatCategory(String? category) {
  if (category == null || category.isEmpty) return '';
  return category[0].toUpperCase() + category.substring(1);
}
