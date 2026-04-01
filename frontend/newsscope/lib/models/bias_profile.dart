/// Typed model for the /api/bias-profile response.
///
/// Mirrors the Pydantic BiasProfile schema exactly — every field
/// returned by the backend is parsed and typed here.
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

  /// Top-12 sources by article count.
  /// Powers the source breakdown bar chart on the profile screen.
  /// Null when the user has read fewer than 1 article.
  final Map<String, int>? sourceBreakdown;

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
    this.sourceBreakdown,
  });

  factory BiasProfile.fromJson(Map<String, dynamic> json) {
    final rawDist =
        (json['bias_distribution'] as Map<String, dynamic>?) ?? {};
    final rawSources =
        (json['source_breakdown'] as Map<String, dynamic>?);

    return BiasProfile(
      avgBias: (json['avg_bias'] as num?)?.toDouble() ?? 0.0,
      avgSentiment: (json['avg_sentiment'] as num?)?.toDouble() ?? 0.0,
      totalArticlesRead:
          (json['total_articles_read'] as num?)?.toInt() ?? 0,
      leftCount: (json['left_count'] as num?)?.toInt() ?? 0,
      centerCount: (json['center_count'] as num?)?.toInt() ?? 0,
      rightCount: (json['right_count'] as num?)?.toInt() ?? 0,
      mostReadSource:
          json['most_read_source']?.toString() ?? 'N/A',
      biasDistribution: rawDist.map(
        (k, v) => MapEntry(k, (v as num).toDouble()),
      ),
      readingTimeTotalMinutes:
          (json['reading_time_total_minutes'] as num?)?.toInt() ?? 0,
      positiveCount: (json['positive_count'] as num?)?.toInt() ?? 0,
      neutralCount: (json['neutral_count'] as num?)?.toInt() ?? 0,
      negativeCount: (json['negative_count'] as num?)?.toInt() ?? 0,
      sourceBreakdown: rawSources?.map(
        (k, v) => MapEntry(k, (v as num).toInt()),
      ),
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
