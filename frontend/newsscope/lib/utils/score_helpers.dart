import 'package:flutter/material.dart';

// Source-level bias helpers
//
// Uses three buckets only: Left Wing, Centre, Right Wing.
// Blue for left, teal for centre (visually distinct from both sides),
// red for right.

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
  if (score < -0.3) return 'Left Wing';
  return 'Centre';
}

// Article-level political bias helpers (RoBERTa string label)

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

/// Converts a RoBERTa string label to a numeric score on [-1, +1].
/// Returns null for unrecognised labels.
double? politicalBiasToScore(String? label) {
  if (label == null || label.isEmpty) return null;
  final upper = label.toUpperCase();
  if (upper == 'LEFT') return -1.0;
  if (upper == 'RIGHT') return 1.0;
  if (upper == 'CENTER' || upper == 'CENTRE') return 0.0;
  return null;
}

// Sentiment helpers

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

// Credibility helpers

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

/// Maps a fact-check ruling string to a display emoji.
String getRulingEmoji(String ruling) {
  final r = ruling.toLowerCase();
  if (r.contains('true') && !r.contains('mostly')) return '✅';
  if (r.contains('mostly true')) return '🟢';
  if (r.contains('half')) return '🟡';
  if (r.contains('mostly false')) return '🟠';
  if (r.contains('false') || r.contains('pants')) return '❌';
  return '❓';
}

// Category helpers

/// Title-cases a category string, preserving known acronyms.
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
