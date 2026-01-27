// lib/services/api_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/article.dart';

class ApiService {
  static const String baseUrl =
      'https://newsscope-backend.onrender.com';

  final supabase = Supabase.instance.client;

  /// Get authentication token from Supabase
  Future<String?> _getToken() async {
    try {
      final session = supabase.auth.currentSession;
      return session?.accessToken;
    } catch (e) {
      print('Error getting token: $e');
      return null;
    }
  }

  /// Fetches the list of processed articles from the backend
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
        throw Exception(
          'Failed to load articles: ${response.statusCode}'
        );
      }
    } catch (e) {
      throw Exception('Error connecting to backend: $e');
    }
  }

  /// Track reading time for bias profile
  Future<void> trackReading({
    required String articleId,
    required int timeSpentSeconds,
  }) async {
    final token = await _getToken();
    if (token == null) {
      print('No auth token available');
      return;
    }

    try {
      final response = await http.post(
        Uri.parse('$baseUrl/api/reading-history'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $token',
        },
        body: jsonEncode({
          'article_id': articleId,
          'time_spent_seconds': timeSpentSeconds,
        }),
      );

      if (response.statusCode != 200) {
        print('Failed to track reading: ${response.statusCode}');
      }
    } catch (e) {
      print('Error tracking reading: $e');
    }
  }

  /// Get user's bias profile
  Future<Map<String, dynamic>?> getBiasProfile() async {
    final token = await _getToken();
    if (token == null) return null;

    try {
      final response = await http.get(
        Uri.parse('$baseUrl/api/bias-profile'),
        headers: {
          'Authorization': 'Bearer $token',
        },
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        print('Failed to load bias profile: ${response.statusCode}');
      }
    } catch (e) {
      print('Error fetching bias profile: $e');
    }
    return null;
  }

  /// Compare articles by topic
  Future<Map<String, dynamic>?> compareArticles(
    String topic, {
    int limit = 5,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/api/articles/compare'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'topic': topic,
          'limit': limit,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        print(
          'Failed to compare articles: ${response.statusCode}'
        );
      }
    } catch (e) {
      print('Error comparing articles: $e');
    }
    return null;
  }

  /// Get fact-checks for an article
  Future<List<dynamic>?> getFactChecks(String articleId) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/api/fact-checks/$articleId'),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
    } catch (e) {
      print('Error fetching fact-checks: $e');
    }
    return null;
  }
}
