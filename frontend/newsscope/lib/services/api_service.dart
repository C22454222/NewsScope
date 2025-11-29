// lib/services/api_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // Base URL for the hosted FastAPI backend on Render
  final String baseUrl = "https://newsscope-backend.onrender.com";

  /// Fetches the list of processed articles from the backend.
  /// Returns a list of JSON maps (dynamic) representing articles.
  Future<List<dynamic>> getArticles() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/articles'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        // FastAPI returns a JSON list directly, so we decode it as a List
        return json.decode(response.body) as List<dynamic>;
      } else {
        throw Exception("Failed to load articles: ${response.statusCode}");
      }
    } catch (e) {
      throw Exception("Error connecting to backend: $e");
    }
  }
}
