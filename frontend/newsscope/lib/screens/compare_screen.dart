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

  // ── Search ────────────────────────────────────────────────────────────────

  Future<void> _searchTopic() async {
    final topic = _searchController.text.trim();

    if (topic.isEmpty && _selectedCategory == null) {
      setState(() {
        _errorMessage = null;
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
      final results = await _apiService.compareArticles(
        topic,
        category: _selectedCategory?.toLowerCase(),
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
    final newCategory = label == 'All' ? null : label;
    final hasTopic = _searchController.text.trim().isNotEmpty;

    setState(() {
      _selectedCategory = newCategory;
      if (newCategory == null && !hasTopic) {
        _rawResults = null;
        _errorMessage = null;
      }
    });

    if (newCategory != null || hasTopic) {
      _searchTopic();
    }
  }

  // ── Article list ──────────────────────────────────────────────────────────

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
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.article_outlined, size: 48, color: Colors.grey[300]),
              const SizedBox(height: 12),
              Text(
                'No articles found in this band.',
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.grey[500], fontSize: 14),
              ),
            ],
          ),
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.only(top: 8, bottom: 24),
      itemCount: articles.length,
      itemBuilder: (context, index) {
        final article = articles[index];
        return ArticleCard(
          article: article,
          onTap: () async {
            await Navigator.push(
              context,
              MaterialPageRoute(
                builder: (_) => ArticleDetailScreen.fromArticle(article),
              ),
            );
            widget.onArticleRead();
          },
        );
      },
    );
  }

  // ── Results body ──────────────────────────────────────────────────────────

  Widget _buildResultsBody() {
    if (_rawResults == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.compare_arrows, size: 64, color: Colors.grey[300]),
            const SizedBox(height: 16),
            Text(
              'Pick a category or enter a topic\nto see how outlets across the\npolitical spectrum cover it.',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.grey[500],
                fontSize: 14,
                height: 1.6,
              ),
            ),
          ],
        ),
      );
    }

    final leftArticles = _rawResults!['left_articles'] as List<dynamic>?;
    final centreArticles = _rawResults!['center_articles'] as List<dynamic>?;
    final rightArticles = _rawResults!['right_articles'] as List<dynamic>?;
    final total = _rawResults!['total_found'] ?? 0;

    final topic = (_rawResults!['topic'] as String?)?.trim() ?? '';
    final isCategoryOnly = topic.isEmpty;
    final headerText = isCategoryOnly
        ? '${_selectedCategory ?? 'All'} — all coverage'
        : '"$topic"${_selectedCategory != null ? ' · ${_selectedCategory!}' : ''}';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(
                headerText,
                style: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.blue[50],
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.blue[200]!),
              ),
              child: Text(
                '$total articles',
                style: TextStyle(
                  color: Colors.blue[700],
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Expanded(
          child: DefaultTabController(
            length: 3,
            child: Column(
              children: [
                Container(
                  decoration: BoxDecoration(
                    color: Colors.grey[100],
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: TabBar(
                    labelColor: Colors.white,
                    unselectedLabelColor: Colors.grey[600],
                    indicator: BoxDecoration(
                      borderRadius: BorderRadius.circular(12),
                      color: Colors.blue[700],
                    ),
                    dividerColor: Colors.transparent,
                    tabs: [
                      Tab(
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.arrow_back,
                                size: 14, color: Colors.blue[300]),
                            const SizedBox(width: 4),
                            Text(
                              'Left (${leftArticles?.length ?? 0})',
                              style: const TextStyle(fontSize: 12),
                            ),
                          ],
                        ),
                      ),
                      Tab(
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.horizontal_rule,
                                size: 14, color: Colors.purple[300]),
                            const SizedBox(width: 4),
                            Text(
                              'Centre (${centreArticles?.length ?? 0})',
                              style: const TextStyle(fontSize: 12),
                            ),
                          ],
                        ),
                      ),
                      Tab(
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.arrow_forward,
                                size: 14, color: Colors.red[300]),
                            const SizedBox(width: 4),
                            Text(
                              'Right (${rightArticles?.length ?? 0})',
                              style: const TextStyle(fontSize: 12),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
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

  Widget _buildSpectrumTitle() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(Icons.compare_arrows, size: 20, color: Colors.blue[200]),
        const SizedBox(width: 8),
        RichText(
          text: TextSpan(
            style: const TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              letterSpacing: 0.3,
            ),
            children: [
              const TextSpan(
                text: 'The ',
                style: TextStyle(color: Colors.white),
              ),
              TextSpan(
                text: 'Spectrum',
                style: TextStyle(color: Colors.blue[200]),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final hasTopic = _searchController.text.trim().isNotEmpty;
    final buttonLabel = (!hasTopic && _selectedCategory != null)
        ? 'Browse $_selectedCategory'
        : 'Search the Spectrum';

    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: _buildSpectrumTitle(),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Browse by category',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: Colors.grey[600],
                    letterSpacing: 0.5,
                  ),
                ),
                const SizedBox(height: 8),
                SizedBox(
                  height: 36,
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
                        label: Text(label,
                            style: const TextStyle(fontSize: 12)),
                        selected: isSelected,
                        selectedColor: Colors.blue[700],
                        labelStyle: TextStyle(
                          color: isSelected
                              ? Colors.white
                              : Colors.grey[700],
                          fontWeight: isSelected
                              ? FontWeight.w600
                              : FontWeight.normal,
                        ),
                        onSelected: (_) => _onCategorySelected(label),
                      );
                    },
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),

            Row(
              children: [
                Expanded(child: Divider(color: Colors.grey[300])),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  child: Text(
                    'or refine with a keyword',
                    style: TextStyle(fontSize: 11, color: Colors.grey[500]),
                  ),
                ),
                Expanded(child: Divider(color: Colors.grey[300])),
              ],
            ),

            const SizedBox(height: 12),

            TextField(
              controller: _searchController,
              decoration: InputDecoration(
                labelText: _selectedCategory != null
                    ? 'Filter by keyword (optional)'
                    : 'Enter a topic to compare',
                hintText: 'e.g., climate, housing, election',
                prefixIcon: const Icon(Icons.search),
                border: const OutlineInputBorder(),
                suffixIcon: _searchController.text.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () {
                          _searchController.clear();
                          setState(() {});
                          if (_selectedCategory != null) {
                            _searchTopic();
                          } else {
                            setState(() {
                              _rawResults = null;
                              _errorMessage = null;
                            });
                          }
                        },
                      )
                    : null,
              ),
              onChanged: (_) => setState(() {}),
              onSubmitted: (_) => _searchTopic(),
            ),

            const SizedBox(height: 12),

            ElevatedButton.icon(
              onPressed: _isLoading ? null : _searchTopic,
              icon: const Icon(Icons.compare_arrows),
              label: Text(buttonLabel),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue[700],
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 14),
                textStyle: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                ),
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
                              const Icon(Icons.error_outline,
                                  size: 64, color: Colors.red),
                              const SizedBox(height: 16),
                              Text(
                                _errorMessage!,
                                textAlign: TextAlign.center,
                              ),
                              const SizedBox(height: 16),
                              ElevatedButton(
                                onPressed: _searchTopic,
                                child: const Text('Retry'),
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
