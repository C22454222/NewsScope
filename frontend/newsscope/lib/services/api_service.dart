import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:firebase_auth/firebase_auth.dart';

import '../models/article.dart';
import '../models/bias_profile.dart';

class ApiService {
  static const String _baseUrl = 'https://newsscope-backend.onrender.com';

  // ── Auth ──────────────────────────────────────────────────────────────────

  Future<String?> _getToken() async {
    try {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null) {
        debugPrint('No Firebase user logged in');
        return null;
      }
      return await user.getIdToken();
    } catch (e) {
      debugPrint('Error getting Firebase token: $e');
      return null;
    }
  }

  // ── User registration ─────────────────────────────────────────────────────

  Future<void> registerUser({
    required String uid,
    required String email,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/users'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'id': uid, 'email': email}),
      );
      if (response.statusCode != 200) {
        debugPrint('Failed to register user: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error registering user: $e');
    }
  }

  // ── Articles ──────────────────────────────────────────────────────────────

  Future<List<Article>> getArticles({
    String? category,
    String? source,
  }) async {
    try {
      final params = <String, String>{};
      if (category != null && category.isNotEmpty) {
        params['category'] = category;
      }
      if (source != null && source.isNotEmpty) {
        params['source'] = source;
      }

      final uri = params.isEmpty
          ? Uri.parse('$_baseUrl/articles')
          : Uri.parse('$_baseUrl/articles')
              .replace(queryParameters: params);

      final response = await http.get(
        uri,
        headers: {'Content-Type': 'application/json'},
      );
      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        return data.map((j) => Article.fromJson(j)).toList();
      } else {
        throw Exception(
            'Failed to load articles: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Error connecting to backend: $e');
    }
  }

  Future<Article?> getArticle(String articleId) async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl/articles/$articleId'),
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

  // ── Reading history ───────────────────────────────────────────────────────

  Future<void> trackReading({
    required String articleId,
    required int timeSpentSeconds,
  }) async {
    final token = await _getToken();
    if (token == null) {
      debugPrint('No auth token available for tracking');
      return;
    }
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/api/reading-history'),
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
        debugPrint(
            'Reading tracked: $articleId for ${timeSpentSeconds}s');
      }
    } catch (e) {
      debugPrint('Error tracking reading: $e');
    }
  }

  // ── Bias profile ──────────────────────────────────────────────────────────

  Future<BiasProfile?> getBiasProfile() async {
    final token = await _getToken();
    if (token == null) {
      debugPrint('No token for bias profile');
      return null;
    }
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl/api/bias-profile'),
        headers: {'Authorization': 'Bearer $token'},
      );
      if (response.statusCode == 200) {
        final profile = BiasProfile.fromJson(
          jsonDecode(response.body) as Map<String, dynamic>,
        );
        debugPrint(
            'Profile loaded: ${profile.totalArticlesRead} articles');
        return profile;
      } else {
        debugPrint(
            'Failed to load bias profile: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error fetching bias profile: $e');
    }
    return null;
  }

  // ── Article comparison ────────────────────────────────────────────────────

  Future<Map<String, dynamic>?> compareArticles(
    String topic, {
    int limit = 5,
    String? category,
    String? source,
  }) async {
    try {
      final params = <String, String>{'limit': '$limit'};
      if (topic.isNotEmpty) params['topic'] = topic;
      if (category != null && category.isNotEmpty) {
        params['category'] = category.toLowerCase();
      }
      if (source != null && source.isNotEmpty) {
        params['source'] = source;
      }

      final uri = Uri.parse('$_baseUrl/articles/compare')
          .replace(queryParameters: params);

      final response = await http.get(
        uri,
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        // Backend returns a flat list — split by general_bias for
        // the left / centre / right column layout in CompareScreen
        final List<dynamic> articles = jsonDecode(response.body);
        return {
          'left_articles': articles
              .where((a) => a['general_bias'] == 'left')
              .toList(),
          'center_articles': articles
              .where((a) => a['general_bias'] == 'center')
              .toList(),
          'right_articles': articles
              .where((a) => a['general_bias'] == 'right')
              .toList(),
          'total_found': articles.length,
        };
      } else {
        debugPrint(
            'Failed to compare articles: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error comparing articles: $e');
    }
    return null;
  }

  // ── Fact-checks ───────────────────────────────────────────────────────────

  Future<Map<String, dynamic>?> triggerFactCheck(
      String articleId) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/articles/$articleId/factcheck'),
        headers: {'Content-Type': 'application/json'},
      );
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        debugPrint(
            'Fact-check trigger failed: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error triggering fact-check: $e');
    }
    return null;
  }
}
