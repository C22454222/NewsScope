// lib/services/api_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/article.dart';


class ApiService {
  final String baseUrl = "https://newsscope-backend.onrender.com";

  /// Fetches the list of processed articles from the backend.
  /// Returns a list of Article objects.
  Future<List<Article>> getArticles() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/articles'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        return data.map((json) => Article.fromJson(json)).toList();
      } else {
        throw Exception("Failed to load articles: ${response.statusCode}");
      }
    } catch (e) {
      throw Exception("Error connecting to backend: $e");
    }
  }

  /// Fetches articles matching a search topic for comparison.
  Future<List<Article>> compareArticles(String topic) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/articles/compare?topic=${Uri.encodeComponent(topic)}'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        return data.map((json) => Article.fromJson(json)).toList();
      } else {
        throw Exception("Failed to compare articles: ${response.statusCode}");
      }
    } catch (e) {
      throw Exception("Error fetching comparison: $e");
    }
  }

  /// Fetches user's reading history (mock data for now).
  Future<List<Article>> getUserReadingHistory(String userId) async {
    await Future.delayed(const Duration(seconds: 1));

    return [
      Article(
        id: '1',
        title: 'Climate Policy Debate Heats Up',
        source: 'BBC News',
        url: 'https://bbc.co.uk/news/climate',
        publishedAt: DateTime.now().subtract(const Duration(days: 1)),
        biasScore: -0.4,
        sentimentScore: -0.2,
      ),
      Article(
        id: '2',
        title: 'Economic Growth Continues',
        source: 'CNN',
        url: 'https://cnn.com/business',
        publishedAt: DateTime.now().subtract(const Duration(days: 2)),
        biasScore: -0.1,
        sentimentScore: 0.6,
      ),
      Article(
        id: '3',
        title: 'Government Announces Tax Cuts',
        source: 'GB News',
        url: 'https://gbnews.com/politics',
        publishedAt: DateTime.now().subtract(const Duration(days: 3)),
        biasScore: 0.5,
        sentimentScore: 0.3,
      ),
    ];
  }

  /// Gets aggregated bias profile stats (mock for now).
  Future<Map<String, int>> getBiasProfile(String userId) async {
    await Future.delayed(const Duration(seconds: 1));

    return {
      'Left': 7,
      'Center': 2,
      'Right': 1,
    };
  }
}
