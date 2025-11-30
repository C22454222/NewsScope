// lib/models/article.dart

class Article {
  final String id;
  final String title;
  final String? content;
  final String url;
  final String source;
  final DateTime? publishedAt;
  // New analysis fields
  final double? sentimentScore;
  final double? biasScore;

  Article({
    required this.id,
    required this.title,
    this.content,
    required this.url,
    required this.source,
    this.publishedAt,
    this.sentimentScore,
    this.biasScore,
  });

  factory Article.fromJson(Map<String, dynamic> json) {
    return Article(
      // Handle potential ID types (int vs string) safely
      id: json['id']?.toString() ?? '',
      
      title: json['title'] ?? 'No Title',
      content: json['content'],
      url: json['url'] ?? '',
      
      // Handle nested source object if your API returns it, or flat string
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
    );
  }
}
