// lib/models/article.dart

class Article {
  final String id;
  final String title;
  final String? content;
  final String url;
  final String source;
  final DateTime? publishedAt;
  final double? sentimentScore;
  final double? biasScore;
  final double? biasIntensity;
  final String? category; // NEW

  Article({
    required this.id,
    required this.title,
    this.content,
    required this.url,
    required this.source,
    this.publishedAt,
    this.sentimentScore,
    this.biasScore,
    this.biasIntensity,
    this.category, // NEW
  });

  factory Article.fromJson(Map<String, dynamic> json) {
    return Article(
      // Handle potential ID types (int vs string) safely
      id: json['id']?.toString() ?? '',

      title: json['title'] ?? 'No Title',
      content: json['content'],
      url: json['url'] ?? '',

      // Handle nested source object or flat string
      source: json['source'] is Map
          ? (json['source']['name'] ?? 'Unknown Source')
          : (json['source'] ?? 'Unknown Source'),

      publishedAt: json['published_at'] != null
          ? DateTime.tryParse(json['published_at'])
          : null,

      // Safely parse numbers (API might return int or double)
      sentimentScore: json['sentiment_score'] != null
          ? (json['sentiment_score'] as num).toDouble()
          : null,
      biasScore: json['bias_score'] != null
          ? (json['bias_score'] as num).toDouble()
          : null,
      biasIntensity: json['bias_intensity'] != null
          ? (json['bias_intensity'] as num).toDouble()
          : null,

      // NEW: category from backend JSON
      category: json['category']?.toString(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'title': title,
      'content': content,
      'url': url,
      'source': source,
      'published_at': publishedAt?.toIso8601String(),
      'sentiment_score': sentimentScore,
      'bias_score': biasScore,
      'bias_intensity': biasIntensity,
      'category': category, // NEW
    };
  }
}
