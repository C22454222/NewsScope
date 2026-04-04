import 'package:flutter/material.dart';

// ── Bias ──────────────────────────────────────────────────────────────────────
// Stronger blue for left, teal for centre (fully distinct from both sides),
// stronger red for right.

Color getBiasColor(double? score) {
  if (score == null) return Colors.grey.shade500;
  if (score > 0.3) return Colors.red[800]!;
  if (score < -0.3) return Colors.blue[800]!;
  return Colors.teal[600]!;
}

String getBiasLabelShort(double? score) {
  if (score == null) return 'Unknown';
  if (score > 0.3) return 'Right Wing';
  if (score < -0.3) return 'Left Wing';
  return 'Centre';
}

String getBiasLabel(double? score) {
  if (score == null) return 'Unknown';
  if (score > 0.3) return 'Right Wing';
  if (score > 0.1) return 'Centre Right';
  if (score >= -0.1) return 'Centre';
  if (score >= -0.3) return 'Centre Left';
  return 'Left Wing';
}

// ── Sentiment ─────────────────────────────────────────────────────────────────
// Strong green for positive, deep orange for negative (distinct from red/right),
// amber for neutral.

Color getSentimentColor(double? score) {
  if (score == null) return Colors.grey.shade500;
  if (score > 0.1) return Colors.green[700]!;
  if (score < -0.1) return Colors.deepOrange[600]!;
  return Colors.amber[600]!;
}

String getSentimentLabel(double? score) {
  if (score == null) return 'Neutral';
  if (score > 0.1) return 'Positive';
  if (score < -0.1) return 'Negative';
  return 'Neutral';
}

// ── Credibility ───────────────────────────────────────────────────────────────

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
  const preserve = {'us': 'US', 'uk': 'UK', 'eu': 'EU', 'gaa': 'GAA'};
  return category
      .split(' ')
      .map((w) {
        final lower = w.toLowerCase();
        if (preserve.containsKey(lower)) return preserve[lower]!;
        return w.isEmpty ? '' : w[0].toUpperCase() + w.substring(1);
      })
      .join(' ');
}
