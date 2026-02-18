import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:firebase_auth/firebase_auth.dart';
import '../models/article.dart';

class ApiService {
  static const String baseUrl = 'https://newsscope-backend.onrender.com';

  Future<String?> _getToken() async {
    try {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null) {
        debugPrint('No Firebase user logged in');
        return null;
      }
      final token = await user.getIdToken();
      debugPrint('Got Firebase token for user: ${user.uid}');
      return token;
    } catch (e) {
      debugPrint('Error getting Firebase token: $e');
      return null;
    }
  }

  Future<List<Article>> getArticles({String? category}) async {
    try {
      final uri = Uri.parse(
        category == null || category.isEmpty
            ? '$baseUrl/articles'
            : '$baseUrl/articles?category=$category',
      );

      final response = await http.get(
        uri,
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        return data.map((json) => Article.fromJson(json)).toList();
      } else {
        throw Exception('Failed to load articles: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error connecting to backend: $e');
    }
  }

  Future<void> trackReading({
    required String articleId,
    required int timeSpentSeconds,
  }) async {
    final token = await _getToken();

    debugPrint('Token: ${token?.substring(0, 20)}...');
    debugPrint('Tracking: $articleId for ${timeSpentSeconds}s');

    if (token == null) {
      debugPrint('No auth token available for tracking');
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

      debugPrint('Response: ${response.statusCode}');
      debugPrint('Body: ${response.body}');

      if (response.statusCode != 200) {
        debugPrint('Failed to track reading: ${response.statusCode}');
      } else {
        debugPrint('Reading tracked successfully!');
      }
    } catch (e) {
      debugPrint('Error tracking reading: $e');
    }
  }

  Future<Map<String, dynamic>?> getBiasProfile() async {
    final token = await _getToken();
    if (token == null) {
      debugPrint('No token for bias profile');
      return null;
    }

    try {
      debugPrint('Fetching bias profile...');

      final response = await http.get(
        Uri.parse('$baseUrl/api/bias-profile'),
        headers: {
          'Authorization': 'Bearer $token',
        },
      );

      debugPrint('📡 Profile response: ${response.statusCode}');

      if (response.statusCode == 200) {
        final profile = jsonDecode(response.body);
        debugPrint('Profile loaded: ${profile['total_articles_read']} articles');
        return profile;
      } else {
        debugPrint('Failed to load bias profile: ${response.statusCode}');
        debugPrint('Response: ${response.body}');
      }
    } catch (e) {
      debugPrint('Error fetching bias profile: $e');
    }
    return null;
  }

  Future<Map<String, dynamic>?> compareArticles(
    String topic, {
    int limit = 5,
    String? category,
  }) async {
    try {
      final uri = Uri.parse('$baseUrl/articles/compare').replace(
        queryParameters: {
          'topic': topic,
          'limit': limit.toString(),
          if (category != null && category != 'All') 'category': category,
        },
      );

      final response = await http.get(
        uri,
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        debugPrint('Failed to compare articles: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error comparing articles: $e');
    }
    return null;
  }

  Future<List<dynamic>?> getFactChecks(String articleId) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/api/fact-checks/$articleId'),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
    } catch (e) {
      debugPrint('Error fetching fact-checks: $e');
    }
    return null;
  }
}
