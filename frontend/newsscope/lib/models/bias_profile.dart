// lib/models/bias_profile.dart

/// Typed model for the /api/bias-profile response.
///
/// Replaces Map (String, dynamic) usage in profile_screen and
/// api_service, providing null-safe field access and a single
/// fromJson factory consistent with Article.fromJson.
class BiasProfile {
  final double avgBias;
  final double avgSentiment;
  final int totalArticlesRead;
  final int leftCount;
  final int centerCount;
  final int rightCount;
  final String mostReadSource;
  final Map<String, double> biasDistribution;
  final int readingTimeTotalMinutes;
  final int positiveCount;
  final int neutralCount;
  final int negativeCount;

  const BiasProfile({
    required this.avgBias,
    required this.avgSentiment,
    required this.totalArticlesRead,
    required this.leftCount,
    required this.centerCount,
    required this.rightCount,
    required this.mostReadSource,
    required this.biasDistribution,
    required this.readingTimeTotalMinutes,
    required this.positiveCount,
    required this.neutralCount,
    required this.negativeCount,
  });

  factory BiasProfile.fromJson(Map<String, dynamic> json) {
    final rawDist =
        (json['bias_distribution'] as Map<String, dynamic>?) ?? {};

    return BiasProfile(
      avgBias: (json['avg_bias'] as num?)?.toDouble() ?? 0.0,
      avgSentiment: (json['avg_sentiment'] as num?)?.toDouble() ?? 0.0,
      totalArticlesRead:
          (json['total_articles_read'] as num?)?.toInt() ?? 0,
      leftCount: (json['left_count'] as num?)?.toInt() ?? 0,
      centerCount: (json['center_count'] as num?)?.toInt() ?? 0,
      rightCount: (json['right_count'] as num?)?.toInt() ?? 0,
      mostReadSource: json['most_read_source']?.toString() ?? 'N/A',
      biasDistribution: rawDist.map(
        (k, v) => MapEntry(k, (v as num).toDouble()),
      ),
      readingTimeTotalMinutes:
          (json['reading_time_total_minutes'] as num?)?.toInt() ?? 0,
      positiveCount: (json['positive_count'] as num?)?.toInt() ?? 0,
      neutralCount: (json['neutral_count'] as num?)?.toInt() ?? 0,
      negativeCount: (json['negative_count'] as num?)?.toInt() ?? 0,
    );
  }

  /// True when the user has not read any articles yet.
  bool get isEmpty => totalArticlesRead == 0;

  /// Dominant sentiment derived from article counts.
  String get sentimentLabel {
    if (positiveCount == 0 && neutralCount == 0 && negativeCount == 0) {
      return 'Neutral';
    }
    if (positiveCount >= neutralCount && positiveCount >= negativeCount) {
      return 'Positive';
    }
    if (negativeCount >= positiveCount && negativeCount >= neutralCount) {
      return 'Negative';
    }
    return 'Neutral';
  }
}
