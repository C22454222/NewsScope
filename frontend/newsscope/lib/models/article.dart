/// NewsScope Article model.
/// Fields match the backend ArticleResponse schema including
/// credibility, fact-checks, and general bias columns.
class Article {
  final String id;
  final String title;
  final String? content;
  final String? description;
  final String url;
  final String source;
  final DateTime? publishedAt;
  final double? sentimentScore;
  final double? biasScore;
  final double? biasIntensity;
  final String? category;
  final String? generalBias;
  final double? generalBiasScore;

  // Credibility + fact-checking fields
  final double? credibilityScore;
  final Map<String, dynamic>? factChecks;
  final int? claimsChecked;
  final String? credibilityReason;

  Article({
    required this.id,
    required this.title,
    this.content,
    this.description,
    required this.url,
    required this.source,
    this.publishedAt,
    this.sentimentScore,
    this.biasScore,
    this.biasIntensity,
    this.category,
    this.generalBias,
    this.generalBiasScore,
    this.credibilityScore,
    this.factChecks,
    this.claimsChecked,
    this.credibilityReason,
  });

  factory Article.fromJson(Map<String, dynamic> json) {
    return Article(
      id: json['id']?.toString() ?? '',
      title: json['title'] ?? 'No Title',
      content: json['content'],
      description: json['description'],
      url: json['url'] ?? '',
      source: json['source'] is Map
          ? (json['source']['name'] ?? 'Unknown Source')
          : (json['source'] ?? 'Unknown Source'),
      publishedAt: json['published_at'] != null
          ? DateTime.tryParse(json['published_at'])
          : null,
      sentimentScore: json['sentiment_score'] != null
          ? (json['sentiment_score'] as num).toDouble()
          : null,
      biasScore: json['bias_score'] != null
          ? (json['bias_score'] as num).toDouble()
          : null,
      biasIntensity: json['bias_intensity'] != null
          ? (json['bias_intensity'] as num).toDouble()
          : null,
      category: json['category']?.toString(),
      generalBias: json['general_bias']?.toString(),
      generalBiasScore: json['general_bias_score'] != null
          ? (json['general_bias_score'] as num).toDouble()
          : null,
      credibilityScore: json['credibility_score'] != null
          ? (json['credibility_score'] as num).toDouble()
          : null,
      factChecks: json['fact_checks'] is Map
          ? Map<String, dynamic>.from(json['fact_checks'])
          : null,
      claimsChecked: json['claims_checked'] != null
          ? (json['claims_checked'] as num).toInt()
          : null,
      credibilityReason: json['credibility_reason']?.toString(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'title': title,
      'content': content,
      'description': description,
      'url': url,
      'source': source,
      'published_at': publishedAt?.toIso8601String(),
      'sentiment_score': sentimentScore,
      'bias_score': biasScore,
      'bias_intensity': biasIntensity,
      'category': category,
      'general_bias': generalBias,
      'general_bias_score': generalBiasScore,
      'credibility_score': credibilityScore,
      'fact_checks': factChecks,
      'claims_checked': claimsChecked,
      'credibility_reason': credibilityReason,
    };
  }
}
