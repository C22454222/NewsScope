import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {

  final String baseUrl = "https://newsscope.onrender.com";

  Future<List<dynamic>> getArticles() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/articles'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        // Your FastAPI /articles endpoint returns a list directly
        return json.decode(response.body) as List<dynamic>;
      } else {
        throw Exception("Failed to load articles: ${response.statusCode}");
      }
    } catch (e) {
      throw Exception("Error connecting to backend: $e");
    }
  }
}
