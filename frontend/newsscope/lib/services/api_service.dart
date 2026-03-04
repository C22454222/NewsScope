import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:firebase_auth/firebase_auth.dart';

import '../models/article.dart';

class ApiService {
  static const String baseUrl = 'https://newsscope-backend.onrender.com';

  // ── Auth ────────────────────────────────────────────────────────────────────

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

  // ── Articles ────────────────────────────────────────────────────────────────

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
        return data.map((j) => Article.fromJson(j)).toList();
      } else {
        throw Exception('Failed to load articles: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error connecting to backend: $e');
    }
  }

  Future<Article?> getArticle(String articleId) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/articles/$articleId'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        return Article.fromJson(jsonDecode(response.body));
      } else {
        debugPrint('Failed to load article: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error fetching article: $e');
    }
    return null;
  }

  // ── Reading history ─────────────────────────────────────────────────────────

  Future<void> trackReading({
    required String articleId,
    required int timeSpentSeconds,
  }) async {
    final token = await _getToken();

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

      if (response.statusCode != 200) {
        debugPrint('Failed to track reading: ${response.statusCode}');
      } else {
        debugPrint('Reading tracked successfully');
      }
    } catch (e) {
      debugPrint('Error tracking reading: $e');
    }
  }

  // ── Bias profile ────────────────────────────────────────────────────────────

  Future<Map<String, dynamic>?> getBiasProfile() async {
    final token = await _getToken();
    if (token == null) {
      debugPrint('No token for bias profile');
      return null;
    }

    try {
      final response = await http.get(
        Uri.parse('$baseUrl/api/bias-profile'),
        headers: {'Authorization': 'Bearer $token'},
      );

      if (response.statusCode == 200) {
        final profile = jsonDecode(response.body);
        debugPrint(
          'Profile loaded: ${profile['total_articles_read']} articles',
        );
        return profile;
      } else {
        debugPrint('Failed to load bias profile: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error fetching bias profile: $e');
    }
    return null;
  }

  // ── Article comparison ──────────────────────────────────────────────────────

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

  // ── Fact-checks ─────────────────────────────────────────────────────────────

  Future<List<dynamic>?> getFactChecks(String articleId) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/api/fact-checks/$articleId'),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        debugPrint('Failed to load fact-checks: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error fetching fact-checks: $e');
    }
    return null;
  }

  /// Manually trigger a fact-check re-run for a single article.
  /// Returns the updated credibility data or null on failure.
  Future<Map<String, dynamic>?> triggerFactCheck(String articleId) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/articles/$articleId/factcheck'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        debugPrint('Fact-check trigger failed: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error triggering fact-check: $e');
    }
    return null;
  }
}
