// lib/screens/compare_screen.dart
import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'article_detail_screen.dart';

class CompareScreen extends StatefulWidget {
  const CompareScreen({super.key});

  @override
  State<CompareScreen> createState() => _CompareScreenState();
}

class _CompareScreenState extends State<CompareScreen> {
  final TextEditingController _searchController =
      TextEditingController();
  final ApiService _apiService = ApiService();

  Map<String, dynamic>? _results;
  bool _isLoading = false;
  String? _errorMessage;

  Future<void> _searchTopic() async {
    final topic = _searchController.text.trim();
    if (topic.isEmpty) {
      setState(() {
        _errorMessage = 'Please enter a topic';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _results = null;
    });

    try {
      final results = await _apiService.compareArticles(topic);
      setState(() {
        _results = results;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to load articles: $e';
        _isLoading = false;
      });
    }
  }

  Color _getBiasColor(double? score) {
    if (score == null) return Colors.grey[300]!;
    if (score < -0.3) return Colors.blue[700]!;
    if (score > 0.3) return Colors.red[700]!;
    return Colors.purple[400]!;
  }

  String _getBiasLabel(double? score) {
    if (score == null) return 'Unscored';
    if (score < -0.3) return 'Left';
    if (score > 0.3) return 'Right';
    return 'Center';
  }

  Widget _buildArticleSection(
    String title,
    List<dynamic>? articles,
    Color color,
  ) {
    if (articles == null || articles.isEmpty) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Container(
              width: 4,
              height: 20,
              color: color,
            ),
            const SizedBox(width: 8),
            Text(
              title,
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            const SizedBox(width: 8),
            Chip(
              label: Text('${articles.length}'),
              backgroundColor: color.withAlpha((255 * 0.2).round()),
              labelStyle: TextStyle(color: color, fontSize: 12),
            ),
          ],
        ),
        const SizedBox(height: 12),
        ...articles.map((article) {
          return Card(
            margin: const EdgeInsets.only(bottom: 12),
            child: ListTile(
              title: Text(
                article['title'] ?? 'Untitled',
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              subtitle: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 4),
                  Text(article['source'] ?? 'Unknown'),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      Chip(
                        label: Text(
                          _getBiasLabel(article['bias_score'])
                        ),
                        backgroundColor: _getBiasColor(
                          article['bias_score']
                        ).withAlpha((255 * 0.2).round()),
                        labelStyle: TextStyle(
                          fontSize: 10,
                          color: _getBiasColor(article['bias_score']),
                        ),
                      ),
                      if (article['bias_intensity'] != null)
                        const SizedBox(width: 8),
                      if (article['bias_intensity'] != null)
                        Text(
                          '${((article['bias_intensity'] ?? 0) * 100).round()}% biased',
                          style: const TextStyle(
                            fontSize: 12,
                            color: Colors.grey,
                          ),
                        ),
                    ],
                  ),
                ],
              ),
              isThreeLine: true,
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => ArticleDetailScreen(
                      id: article['id'],
                      title: article['title'] ?? 'Untitled',
                      sourceName: article['source'] ?? 'Unknown',
                      content: article['content'],
                      url: article['url'] ?? '',
                      biasScore: article['bias_score'],
                      biasIntensity: article['bias_intensity'],
                      sentimentScore: article['sentiment_score'],
                    ),
                  ),
                );
              },
            ),
          );
        }),
        const SizedBox(height: 24),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Compare Coverage'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _searchController,
              decoration: InputDecoration(
                labelText: 'Enter a topic to compare',
                hintText: 'e.g., climate, housing, election',
                prefixIcon: const Icon(Icons.search),
                border: const OutlineInputBorder(),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () {
                    _searchController.clear();
                    setState(() {
                      _results = null;
                      _errorMessage = null;
                    });
                  },
                ),
              ),
              onSubmitted: (_) => _searchTopic(),
            ),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: _isLoading ? null : _searchTopic,
              icon: const Icon(Icons.compare_arrows),
              label: const Text('Compare Coverage'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 14),
              ),
            ),
            const SizedBox(height: 16),
            if (_isLoading)
              const Expanded(
                child: Center(child: CircularProgressIndicator()),
              )
            else if (_errorMessage != null)
              Expanded(
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(
                        Icons.error_outline,
                        size: 64,
                        color: Colors.red,
                      ),
                      const SizedBox(height: 16),
                      Text(
                        _errorMessage!,
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              )
            else if (_results != null)
              Expanded(
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Results for "${_results!['topic']}"',
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        '${_results!['total_found']} articles found',
                        style: const TextStyle(color: Colors.grey),
                      ),
                      const SizedBox(height: 24),
                      _buildArticleSection(
                        'Left-leaning',
                        _results!['left_articles'],
                        Colors.blue[700]!,
                      ),
                      _buildArticleSection(
                        'Center',
                        _results!['center_articles'],
                        Colors.purple[400]!,
                      ),
                      _buildArticleSection(
                        'Right-leaning',
                        _results!['right_articles'],
                        Colors.red[700]!,
                      ),
                    ],
                  ),
                ),
              )
            else
              const Expanded(
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.compare_arrows,
                        size: 64,
                        color: Colors.grey,
                      ),
                      SizedBox(height: 16),
                      Text('Enter a topic above to start comparing.'),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }
}
