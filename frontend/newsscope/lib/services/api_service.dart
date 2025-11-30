// lib/services/api_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/article.dart'; // Import your new model

class ApiService {
  // Base URL for the hosted FastAPI backend on Render
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
        // Convert JSON list to List<Article>
        return data.map((json) => Article.fromJson(json)).toList();
      } else {
        throw Exception("Failed to load articles: ${response.statusCode}");
      }
    } catch (e) {
      throw Exception("Error connecting to backend: $e");
    }
  }
}
