/// Typed model for the /api/bias-profile response.
///
/// Mirrors the Pydantic BiasProfile schema exactly.
///
/// Outlet-level fields (leftCount, centerCount, rightCount,
/// biasDistribution, avgBias) derive from reading_history.bias_score —
/// the publisher baseline rating copied from sources at read time.
///
/// Article-level fields (articleLeftCount, articleCenterCount,
/// articleRightCount, articleBiasDistribution, avgArticleBias) derive
/// from articles.political_bias, the per-article RoBERTa label joined
/// from the articles table by the backend at query time.
class BiasProfile {
  // ── Outlet-level bias ────────────────────────────────────────────────────
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

  /// Top-12 sources by article count. Powers the source bar chart.
  final Map<String, int>? sourceBreakdown;

  /// Mean credibility score across all articles the user has read.
  final double? avgCredibility;

  // ── Article-level bias (RoBERTa per-article) ─────────────────────────────
  final int articleLeftCount;
  final int articleCenterCount;
  final int articleRightCount;

  /// Percentage distribution of article-level labels.
  /// Keys: 'left', 'center', 'right'. Values: 0.0–100.0.
  final Map<String, double> articleBiasDistribution;

  /// Time-weighted average article bias on the [-1, +1] scale.
  /// LEFT = -1, CENTER = 0, RIGHT = +1. 0.0 when no data.
  final double avgArticleBias;

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
    this.avgCredibility,
    this.articleLeftCount = 0,
    this.articleCenterCount = 0,
    this.articleRightCount = 0,
    this.articleBiasDistribution = const {},
    this.avgArticleBias = 0.0,
  });

  factory BiasProfile.fromJson(Map<String, dynamic> json) {
    final rawDist =
        (json['bias_distribution'] as Map<String, dynamic>?) ?? {};
    final rawSources =
        (json['source_breakdown'] as Map<String, dynamic>?);
    final rawArticleDist =
        (json['article_bias_distribution'] as Map<String, dynamic>?) ??
            {};

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
      sourceBreakdown: rawSources?.map(
        (k, v) => MapEntry(k, (v as num).toInt()),
      ),
      avgCredibility:
          (json['avg_credibility'] as num?)?.toDouble(),
      articleLeftCount:
          (json['article_left_count'] as num?)?.toInt() ?? 0,
      articleCenterCount:
          (json['article_center_count'] as num?)?.toInt() ?? 0,
      articleRightCount:
          (json['article_right_count'] as num?)?.toInt() ?? 0,
      articleBiasDistribution: rawArticleDist.map(
        (k, v) => MapEntry(k, (v as num).toDouble()),
      ),
      avgArticleBias:
          (json['avg_article_bias'] as num?)?.toDouble() ?? 0.0,
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
