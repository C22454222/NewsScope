// lib/screens/compare_screen.dart
import 'package:flutter/material.dart';

import '../models/article.dart';
import '../services/api_service.dart';
import '../widgets/article_card.dart';
import '../screens/article_detail_screen.dart';

class CompareScreen extends StatefulWidget {
  final VoidCallback onArticleRead;

  const CompareScreen({super.key, required this.onArticleRead});

  @override
  State<CompareScreen> createState() => _CompareScreenState();
}

class _CompareScreenState extends State<CompareScreen> {
  final TextEditingController _searchController = TextEditingController();
  final ApiService _apiService = ApiService();

  Map<String, dynamic>? _rawResults;
  bool _isLoading = false;
  String? _errorMessage;

  String? _selectedCategory;
  final List<String> _categories = const [
    'All',
    'Politics',
    'World',
    'Business',
    'Tech',
    'Sport',
    'Entertainment',
    'Health',
    'Science',
  ];

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _searchTopic() async {
    final topic = _searchController.text.trim();
    if (topic.isEmpty) {
      setState(() {
        _errorMessage = 'Please enter a topic to search';
        _rawResults = null;
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _rawResults = null;
    });

    try {
      // Category passed to backend — backend filters before returning.
      // No client-side limit — backend returns all matching articles.
      final results = await _apiService.compareArticles(
        topic,
        category:
            (_selectedCategory == null || _selectedCategory == 'All')
                ? null
                : _selectedCategory!.toLowerCase(),
      );
      if (!mounted) return;
      setState(() {
        _rawResults = results;
        _isLoading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Failed to load articles: $e';
        _isLoading = false;
      });
    }
  }

  void _onCategorySelected(String label) {
    setState(() => _selectedCategory = label == 'All' ? null : label);
    // Re-run search with new category if a topic is already entered.
    if (_searchController.text.trim().isNotEmpty) _searchTopic();
  }

  /// Converts raw API map list to Article objects for ArticleCard.
  List<Article> _toArticles(List<dynamic>? raw) {
    if (raw == null) return [];
    return raw
        .whereType<Map<String, dynamic>>()
        .map(Article.fromJson)
        .toList();
  }

  Widget _buildArticleList(List<dynamic>? raw) {
    final articles = _toArticles(raw);

    if (articles.isEmpty) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(24.0),
          child: Text(
            'No articles in this band for this topic.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.grey),
          ),
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.only(top: 8, bottom: 24),
      itemCount: articles.length,
      itemBuilder: (context, index) {
        final article = articles[index];
        // ArticleCard matches home feed exactly — bias, sentiment,
        // credibility and source all visible before clicking in.
        return ArticleCard(
          article: article,
          onTap: () async {
            await Navigator.push(
              context,
              MaterialPageRoute(
                builder: (_) =>
                    ArticleDetailScreen.fromArticle(article),
              ),
            );
            widget.onArticleRead();
          },
        );
      },
    );
  }

  Widget _buildResultsBody() {
    if (_rawResults == null) {
      return const Center(
        child: Text(
          'Enter a topic above to compare how\n'
          'different outlets cover the same story.',
          textAlign: TextAlign.center,
          style: TextStyle(color: Colors.grey),
        ),
      );
    }

    final leftArticles = _rawResults!['left_articles'] as List<dynamic>?;
    final centreArticles =
        _rawResults!['center_articles'] as List<dynamic>?;
    final rightArticles = _rawResults!['right_articles'] as List<dynamic>?;
    final total = _rawResults!['total_found'] ?? 0;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Results for "${_rawResults!['topic']}"',
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          '$total articles found',
          style: const TextStyle(color: Colors.grey),
        ),
        const SizedBox(height: 16),
        Expanded(
          child: DefaultTabController(
            length: 3,
            child: Column(
              children: [
                TabBar(
                  labelColor: Theme.of(context).colorScheme.primary,
                  unselectedLabelColor: Colors.grey[600],
                  indicatorColor: Theme.of(context).colorScheme.primary,
                  tabs: [
                    Tab(
                      icon: Icon(
                        Icons.arrow_back,
                        color: Colors.blue[700],
                      ),
                      text: 'Left Wing (${leftArticles?.length ?? 0})',
                    ),
                    Tab(
                      icon: Icon(
                        Icons.horizontal_rule,
                        color: Colors.purple[400],
                      ),
                      text: 'Centre (${centreArticles?.length ?? 0})',
                    ),
                    Tab(
                      icon: Icon(
                        Icons.arrow_forward,
                        color: Colors.red[700],
                      ),
                      text: 'Right Wing (${rightArticles?.length ?? 0})',
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Expanded(
                  child: TabBarView(
                    children: [
                      _buildArticleList(leftArticles),
                      _buildArticleList(centreArticles),
                      _buildArticleList(rightArticles),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Compare Coverage')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SizedBox(
              height: 40,
              child: ListView.separated(
                scrollDirection: Axis.horizontal,
                padding: EdgeInsets.zero,
                itemCount: _categories.length,
                separatorBuilder: (_, _) => const SizedBox(width: 8),
                itemBuilder: (context, index) {
                  final label = _categories[index];
                  final isSelected =
                      (_selectedCategory ?? 'All') == label;
                  return ChoiceChip(
                    label: Text(label),
                    selected: isSelected,
                    onSelected: (_) => _onCategorySelected(label),
                  );
                },
              ),
            ),

            const SizedBox(height: 12),

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
                      _rawResults = null;
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

            Expanded(
              child: _isLoading
                  ? const Center(child: CircularProgressIndicator())
                  : _errorMessage != null
                      ? Center(
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
                        )
                      : _buildResultsBody(),
            ),
          ],
        ),
      ),
    );
  }
}
