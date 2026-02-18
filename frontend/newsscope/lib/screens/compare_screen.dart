import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'article_detail_screen.dart';

class CompareScreen extends StatefulWidget {
  final VoidCallback onArticleRead;

  const CompareScreen({
    super.key,
    required this.onArticleRead,
  });

  @override
  State<CompareScreen> createState() => _CompareScreenState();
}

class _CompareScreenState extends State<CompareScreen>
    with SingleTickerProviderStateMixin {
  final TextEditingController _searchController = TextEditingController();
  final ApiService _apiService = ApiService();

  Map<String, dynamic>? _results;
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

  Future<void> _searchTopic() async {
    final topic = _searchController.text.trim();
    if (topic.isEmpty) {
      setState(() {
        _errorMessage = 'Please enter a topic';
        _results = null;
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _results = null;
    });

    try {
      final results = await _apiService.compareArticles(
        topic,
        category: _selectedCategory == 'All' ? null : _selectedCategory,
      );
      if (!mounted) return;
      setState(() {
        _results = results;
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
    setState(() {
      _selectedCategory = label == 'All' ? null : label;
    });
    if (_searchController.text.isNotEmpty) {
      _searchTopic();
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

  String _formatCategory(String? category) {
    if (category == null || category.isEmpty) return 'General';
    final c = category.toLowerCase();
    return c[0].toUpperCase() + c.substring(1);
  }

  Widget _buildArticleList(List<dynamic>? articles) {
    if (articles == null || articles.isEmpty) {
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
                if (article['category'] != null) ...[
                  const SizedBox(height: 4),
                  Text(
                    _formatCategory(article['category']),
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                      color: Colors.blueGrey[700],
                    ),
                  ),
                ],
                const SizedBox(height: 8),
                Row(
                  children: [
                    Chip(
                      label: Text(
                        _getBiasLabel(
                          (article['bias_score'] as num?)?.toDouble(),
                        ),
                      ),
                      backgroundColor: _getBiasColor(
                        (article['bias_score'] as num?)?.toDouble(),
                      ).withAlpha((255 * 0.2).round()),
                      labelStyle: TextStyle(
                        fontSize: 10,
                        color: _getBiasColor(
                          (article['bias_score'] as num?)?.toDouble(),
                        ),
                      ),
                    ),
                    if (article['bias_intensity'] != null)
                      const SizedBox(width: 8),
                    if (article['bias_intensity'] != null)
                      Text(
                        '${(((article['bias_intensity'] ?? 0) as num) * 100).round()}% biased',
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
            onTap: () async {
              await Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => ArticleDetailScreen(
                    id: article['id'],
                    title: article['title'] ?? 'Untitled',
                    sourceName: article['source'] ?? 'Unknown',
                    content: article['content'],
                    url: article['url'] ?? '',
                    biasScore: (article['bias_score'] as num?)?.toDouble(),
                    biasIntensity:
                        (article['bias_intensity'] as num?)?.toDouble(),
                    sentimentScore:
                        (article['sentiment_score'] as num?)?.toDouble(),
                  ),
                ),
              );
              widget.onArticleRead();
            },
          ),
        );
      },
    );
  }

  Widget _buildResultsWithTabs() {
    if (_results == null) {
      return const Center(
        child: Text('Enter a topic above to start comparing.'),
      );
    }

    final leftArticles = _results!['left_articles'] as List<dynamic>?;
    final centreArticles = _results!['center_articles'] as List<dynamic>?;
    final rightArticles = _results!['right_articles'] as List<dynamic>?;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Results for "${_results!['topic']}"',
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          '${_results!['total_found']} articles found',
          style: const TextStyle(color: Colors.grey),
        ),
        const SizedBox(height: 12),
        SizedBox(
          height: 40,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            padding: EdgeInsets.zero,
            itemCount: _categories.length,
            separatorBuilder: (_, _) => const SizedBox(width: 8),
            itemBuilder: (context, index) {
              final label = _categories[index];
              final isSelected = (_selectedCategory ?? 'All') == label;
              return ChoiceChip(
                label: Text(label),
                selected: isSelected,
                onSelected: (_) => _onCategorySelected(label),
              );
            },
          ),
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
                      icon: Icon(Icons.arrow_back, color: Colors.blue[700]),
                      text: 'Left (${leftArticles?.length ?? 0})',
                    ),
                    Tab(
                      icon: Icon(
                        Icons.horizontal_rule,
                        color: Colors.purple[400],
                      ),
                      text: 'Centre (${centreArticles?.length ?? 0})',
                    ),
                    Tab(
                      icon: Icon(Icons.arrow_forward, color: Colors.red[700]),
                      text: 'Right (${rightArticles?.length ?? 0})',
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
                      : _buildResultsWithTabs(),
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
