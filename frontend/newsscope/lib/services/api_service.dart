// lib/services/api_service.dart
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

  Future<List<Article>> getArticles({String? category}) async {
    try {
      final uri = Uri.parse(
        category == null || category.isEmpty
            ? '$_baseUrl/articles'
            : '$_baseUrl/articles?category=$category',
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
        debugPrint('Reading tracked: $articleId for ${timeSpentSeconds}s');
      }
    } catch (e) {
      debugPrint('Error tracking reading: $e');
    }
  }

  // ── Bias profile ──────────────────────────────────────────────────────────

  /// Returns a typed [BiasProfile] parsed from /api/bias-profile.
  /// Returns null if unauthenticated or on network failure.
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
          'Profile loaded: ${profile.totalArticlesRead} articles',
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

  // ── Article comparison ────────────────────────────────────────────────────

  Future<Map<String, dynamic>?> compareArticles(
    String topic, {
    int limit = 5,
    String? category,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/api/articles/compare'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'topic': topic, 'limit': limit}),
      );
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        debugPrint('Failed to compare articles: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error comparing articles: $e');
    }
    return null;
  }

  // ── Fact-checks ───────────────────────────────────────────────────────────

  Future<Map<String, dynamic>?> triggerFactCheck(String articleId) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/articles/$articleId/factcheck'),
        headers: {'Content-Type': 'application/json'},
      );
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      } else {
        debugPrint('Fact-check trigger failed: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint('Error triggering fact-check: $e');
    }
    return null;
  }
}
