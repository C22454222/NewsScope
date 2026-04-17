import 'package:flutter/material.dart';

// ── Bias (source-level, numeric score from sources table) ────────────────────
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

// ── Political bias (article-level, RoBERTa string label) ─────────────────────
// Backend returns 'LEFT', 'CENTER', 'RIGHT' from the fine-tuned RoBERTa
// classifier operating on the article's own text. Colour palette deliberately
// matches the source-level bias palette so users can compare them visually.

Color getPoliticalBiasColor(String? label) {
  if (label == null || label.isEmpty) return Colors.grey.shade500;
  final upper = label.toUpperCase();
  if (upper == 'LEFT') return Colors.blue[800]!;
  if (upper == 'RIGHT') return Colors.red[800]!;
  if (upper == 'CENTER' || upper == 'CENTRE') return Colors.teal[600]!;
  return Colors.grey.shade500;
}

String getPoliticalBiasLabel(String? label) {
  if (label == null || label.isEmpty) return 'Unknown';
  final upper = label.toUpperCase();
  if (upper == 'LEFT') return 'Left Wing';
  if (upper == 'RIGHT') return 'Right Wing';
  if (upper == 'CENTER' || upper == 'CENTRE') return 'Centre';
  return 'Unknown';
}

// Map a RoBERTa label to a numeric score so it can feed components that
// expect a [-1, +1] value (e.g. weighted averages on the bias profile).
double? politicalBiasToScore(String? label) {
  if (label == null || label.isEmpty) return null;
  final upper = label.toUpperCase();
  if (upper == 'LEFT') return -1.0;
  if (upper == 'RIGHT') return 1.0;
  if (upper == 'CENTER' || upper == 'CENTRE') return 0.0;
  return null;
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
