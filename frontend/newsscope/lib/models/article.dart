/// NewsScope Article model.
///
/// Fields mirror the backend ArticleResponse Pydantic schema and the
/// Supabase articles table. Every DB column has a corresponding field.
///
/// DB column to Dart field mapping:
///   id                     -> id
///   source                 -> source
///   source_id              -> sourceId
///   url                    -> url
///   title                  -> title
///   content                -> content
///   bias_score             -> biasScore        (source-level [-1, +1])
///   bias_intensity         -> biasIntensity
///   sentiment_score        -> sentimentScore
///   published_at           -> publishedAt
///   created_at             -> createdAt
///   updated_at             -> updatedAt
///   category               -> category
///   general_bias           -> generalBias
///   general_bias_score     -> generalBiasScore
///   political_bias         -> politicalBias    (RoBERTa article-level)
///   political_bias_score   -> politicalBiasScore
///   credibility_score      -> credibilityScore
///   fact_checks            -> factChecks       (JSONB)
///   claims_checked         -> claimsChecked
///   credibility_reason     -> credibilityReason
///   credibility_updated_at -> credibilityUpdatedAt
///   bias_explanation       -> biasExplanation  (LIME JSONB list)
class Article {
  final String id;
  final String? sourceId;
  final String source;
  final String url;
  final String title;
  final String? content;
  final DateTime? publishedAt;
  final DateTime? createdAt;
  final DateTime? updatedAt;

  // NLP scores
  final double? biasScore;
  final double? biasIntensity;
  final double? sentimentScore;

  // Classification
  final String? category;
  final String? generalBias;
  final double? generalBiasScore;
  final String? politicalBias;
  final double? politicalBiasScore;

  // Credibility and fact-checking
  final double? credibilityScore;
  final Map<String, dynamic>? factChecks;
  final int? claimsChecked;
  final String? credibilityReason;
  final DateTime? credibilityUpdatedAt;

  // LIME bias explainability token weights
  final List<Map<String, dynamic>>? biasExplanation;

  Article({
    required this.id,
    this.sourceId,
    required this.source,
    required this.url,
    required this.title,
    this.content,
    this.publishedAt,
    this.createdAt,
    this.updatedAt,
    this.biasScore,
    this.biasIntensity,
    this.sentimentScore,
    this.category,
    this.generalBias,
    this.generalBiasScore,
    this.politicalBias,
    this.politicalBiasScore,
    this.credibilityScore,
    this.factChecks,
    this.claimsChecked,
    this.credibilityReason,
    this.credibilityUpdatedAt,
    this.biasExplanation,
  });

  factory Article.fromJson(Map<String, dynamic> json) {
    return Article(
      id: json['id']?.toString() ?? '',
      sourceId: json['source_id']?.toString(),
      // source may arrive as a nested object from some endpoints.
      source: json['source'] is Map
          ? (json['source']['name'] ?? 'Unknown Source')
          : (json['source']?.toString() ?? 'Unknown Source'),
      url: json['url']?.toString() ?? '',
      title: json['title']?.toString() ?? 'No Title',
      content: json['content']?.toString(),
      publishedAt: _parseDate(json['published_at']),
      createdAt: _parseDate(json['created_at']),
      updatedAt: _parseDate(json['updated_at']),
      biasScore: _toDouble(json['bias_score']),
      biasIntensity: _toDouble(json['bias_intensity']),
      sentimentScore: _toDouble(json['sentiment_score']),
      category: json['category']?.toString(),
      generalBias: json['general_bias']?.toString(),
      generalBiasScore: _toDouble(json['general_bias_score']),
      politicalBias: json['political_bias']?.toString(),
      politicalBiasScore: _toDouble(json['political_bias_score']),
      credibilityScore: _toDouble(json['credibility_score']),
      factChecks: json['fact_checks'] is Map
          ? Map<String, dynamic>.from(json['fact_checks'])
          : null,
      claimsChecked: json['claims_checked'] != null
          ? (json['claims_checked'] as num).toInt()
          : null,
      credibilityReason: json['credibility_reason']?.toString(),
      credibilityUpdatedAt: _parseDate(json['credibility_updated_at']),
      biasExplanation: json['bias_explanation'] != null
          ? List<Map<String, dynamic>>.from(
              (json['bias_explanation'] as List).map(
                (e) => Map<String, dynamic>.from(e as Map),
              ),
            )
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'source_id': sourceId,
      'source': source,
      'url': url,
      'title': title,
      'content': content,
      'published_at': publishedAt?.toIso8601String(),
      'created_at': createdAt?.toIso8601String(),
      'updated_at': updatedAt?.toIso8601String(),
      'bias_score': biasScore,
      'bias_intensity': biasIntensity,
      'sentiment_score': sentimentScore,
      'category': category,
      'general_bias': generalBias,
      'general_bias_score': generalBiasScore,
      'political_bias': politicalBias,
      'political_bias_score': politicalBiasScore,
      'credibility_score': credibilityScore,
      'fact_checks': factChecks,
      'claims_checked': claimsChecked,
      'credibility_reason': credibilityReason,
      'credibility_updated_at': credibilityUpdatedAt?.toIso8601String(),
      'bias_explanation': biasExplanation,
    };
  }

  // Private parsing helpers used by fromJson.

  static DateTime? _parseDate(dynamic value) {
    if (value == null) return null;
    return DateTime.tryParse(value.toString());
  }

  static double? _toDouble(dynamic value) {
    if (value == null) return null;
    return (value as num).toDouble();
  }
}
