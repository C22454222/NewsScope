import 'package:flutter/material.dart';

// ── Bias ──────────────────────────────────────────────────────────────────────
// blue[700] / purple[400] / red[700] — matches pie chart exactly.
// Centre changed from green (sentiment colour) → purple for clear distinction.

Color getBiasColor(double? score) {
  if (score == null) return Colors.grey.shade500;
  if (score > 0.3) return Colors.red[700]!;
  if (score < -0.3) return Colors.blue[700]!;
  return Colors.purple[400]!;
}

String getBiasLabelShort(double? score) {
  if (score == null) return 'Unknown';
  if (score > 0.3) return 'Right Wing';
  if (score < -0.3) return 'Left Wing';
  return 'Centre';
}

/// Returns a granular bias label based on score position.
///
/// Thresholds:
///   > 0.3             → Right Wing
///   > 0.1 to 0.3      → Centre Right
///   -0.1 to 0.1       → Centre
///   -0.3 to -0.1      → Centre Left
///   < -0.3            → Left Wing
String getBiasLabel(double? score) {
  if (score == null) return 'Unknown';
  if (score > 0.3) return 'Right Wing';
  if (score > 0.1) return 'Centre Right';
  if (score >= -0.1) return 'Centre';
  if (score >= -0.3) return 'Centre Left';
  return 'Left Wing';
}

// ── Sentiment ─────────────────────────────────────────────────────────────────
// green[600] / orange[600] / red[600] — never overlaps with bias colours.

Color getSentimentColor(double? score) {
  if (score == null) return Colors.grey.shade500;
  if (score > 0.1) return Colors.green[600]!;
  if (score < -0.1) return Colors.red[600]!;
  return Colors.orange[600]!;
}

String getSentimentLabel(double? score) {
  if (score == null) return 'Neutral';
  if (score > 0.1) return 'Positive';
  if (score < -0.1) return 'Negative';
  return 'Neutral';
}

// ── Credibility ───────────────────────────────────────────────────────────────
// green[700] / amber[700] / red[800] — stronger shades than sentiment
// so credibility chips are visually distinct even when adjacent.

Color getCredibilityColor(double? score) {
  if (score == null) return Colors.grey.shade500;
  if (score >= 70) return Colors.green[700]!;
  if (score >= 40) return Colors.amber[700]!;
  return Colors.red[800]!;
}

String getCredibilityLabel(double? score) {
  if (score == null) return 'Unverified';
  if (score >= 70) return 'Reliable';
  if (score >= 40) return 'Mixed';
  return 'Low';
}

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

String formatCategory(String? category) {
  if (category == null || category.isEmpty) return '';
  return category
      .split(' ')
      .map((w) => w.isEmpty ? '' : w[0].toUpperCase() + w.substring(1))
      .join(' ');
}
