import 'package:flutter/material.dart';

import '../models/article.dart';
import '../services/api_service.dart';
import '../widgets/article_card.dart';
import '../screens/article_detail_screen.dart';

/// Controls how the Left / Centre / Right tabs are populated.
///
/// [source] uses the outlet's baseline bias score (server-side grouping).
/// [article] uses the article-level RoBERTa label, rebucketed client-side.
enum CompareGrouping { source, article }

/// Compare screen shows how a topic is covered across the political spectrum.
class CompareScreen extends StatefulWidget {
  final VoidCallback onArticleRead;

  const CompareScreen({super.key, required this.onArticleRead});

  @override
  State<CompareScreen> createState() => _CompareScreenState();
}

class _CompareScreenState extends State<CompareScreen>
    with SingleTickerProviderStateMixin {
  final TextEditingController _searchController = TextEditingController();
  final FocusNode _searchFocusNode = FocusNode();
  final ApiService _apiService = ApiService();

  late TabController _tabController;

  // Raw API response, kept so we can regroup without refetching.
  Map<String, dynamic>? _rawResults;
  bool _isLoading = false;
  String? _errorMessage;
  int _activeTab = 0;

  String? _selectedCategory;
  String? _selectedSource;

  // Default to outlet-level grouping, matching the original feature.
  CompareGrouping _grouping = CompareGrouping.source;

  static const List<Map<String, String>> _categories = [
    {'label': 'All', 'value': ''},
    {'label': 'Politics', 'value': 'politics'},
    {'label': 'World', 'value': 'world'},
    {'label': 'US', 'value': 'us'},
    {'label': 'UK', 'value': 'uk'},
    {'label': 'Ireland', 'value': 'ireland'},
    {'label': 'Europe', 'value': 'europe'},
    {'label': 'Business', 'value': 'business'},
    {'label': 'Tech', 'value': 'tech'},
    {'label': 'Science', 'value': 'science'},
    {'label': 'Health', 'value': 'health'},
    {'label': 'Environment', 'value': 'environment'},
    {'label': 'Sport', 'value': 'sport'},
    {'label': 'Entertainment', 'value': 'entertainment'},
    {'label': 'Crime', 'value': 'crime'},
    {'label': 'Opinion', 'value': 'opinion'},
  ];

  // Maps the short chip label to the canonical source name used by the API.
  static const Map<String, String> _sourceMap = {
    'BBC': 'BBC News',
    'RTE': 'RTÉ News',
    'Guardian': 'The Guardian',
    'CNN': 'CNN',
    'Irish Times': 'The Irish Times',
    'AP News': 'AP News',
    'Sky News': 'Sky News',
    'Independent': 'The Independent',
    'NPR': 'NPR',
    'DW': 'Deutsche Welle',
    'GB News': 'GB News',
    'Fox News': 'Fox News',
  };

  static const List<String> _sourceLabels = [
    'All', 'BBC', 'RTE', 'Guardian', 'CNN', 'Irish Times',
    'AP News', 'Sky News', 'Independent', 'NPR', 'DW',
    'GB News', 'Fox News',
  ];

  // Tab indicator colours: Left (blue), Centre (teal), Right (red).
  static const _tabColors = [
    Color(0xFF1565C0),
    Color(0xFF00796B),
    Color(0xFFC62828),
  ];

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _tabController.addListener(() {
      if (_tabController.indexIsChanging ||
          _tabController.index != _activeTab) {
        setState(() => _activeTab = _tabController.index);
      }
    });
  }

  @override
  void dispose() {
    _searchController.dispose();
    _searchFocusNode.dispose();
    _tabController.dispose();
    super.dispose();
  }

  // True when the user has entered a keyword or selected a category/source.
  bool get _hasAnyFilter =>
      _searchController.text.trim().isNotEmpty ||
      (_selectedCategory != null && _selectedCategory!.isNotEmpty) ||
      _selectedSource != null;

  /// Fetches compare results from the API for the current filter selection.
  Future<void> _searchTopic() async {
    FocusScope.of(context).unfocus();

    // Nothing to search for: clear any stale results and bail.
    if (!_hasAnyFilter) {
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
        _searchController.text.trim(),
        category: (_selectedCategory == null || _selectedCategory!.isEmpty)
            ? null
            : _selectedCategory,
        source: _selectedSource,
      );
      if (!mounted) return;
      setState(() {
        _rawResults = results;
        _isLoading = false;
        _tabController.animateTo(0);
        _activeTab = 0;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Failed to load articles: $e';
        _isLoading = false;
      });
    }
  }

  void _onCategorySelected(String value) {
    setState(() {
      _selectedCategory = value.isEmpty ? null : value;
    });
    if (_hasAnyFilter) {
      _searchTopic();
    } else {
      setState(() {
        _rawResults = null;
        _errorMessage = null;
      });
    }
  }

  void _onSourceSelected(String label) {
    setState(() {
      _selectedSource = label == 'All' ? null : _sourceMap[label];
    });
    if (_hasAnyFilter) {
      _searchTopic();
    } else {
      setState(() {
        _rawResults = null;
        _errorMessage = null;
      });
    }
  }

  Widget _buildFilterLabel(String label) {
    return Text(
      label,
      style: TextStyle(
        fontSize: 10,
        fontWeight: FontWeight.bold,
        color: Colors.blue[700],
        letterSpacing: 1.2,
      ),
    );
  }

  Widget _buildChip({
    required String label,
    required bool isSelected,
    required VoidCallback onTap,
  }) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected
              ? Colors.blue[700]
              : (isDark ? Colors.grey[800] : Colors.grey[100]),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isSelected
                ? Colors.blue[700]!
                : (isDark ? Colors.grey[600]! : Colors.grey[300]!),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (isSelected) ...[
              const Icon(Icons.check, size: 12, color: Colors.white),
              const SizedBox(width: 4),
            ],
            Text(
              label,
              style: TextStyle(
                fontSize: 12,
                fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
                color: isSelected
                    ? Colors.white
                    : (isDark ? Colors.grey[300] : Colors.grey[700]),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCategoryChips() {
    return SizedBox(
      height: 36,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: EdgeInsets.zero,
        itemCount: _categories.length,
        separatorBuilder: (_, _) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          final cat = _categories[index];
          final label = cat['label']!;
          final value = cat['value']!;
          final isSelected = value.isEmpty
              ? (_selectedCategory == null || _selectedCategory!.isEmpty)
              : _selectedCategory == value;
          return _buildChip(
            label: label,
            isSelected: isSelected,
            onTap: () => _onCategorySelected(value),
          );
        },
      ),
    );
  }

  Widget _buildSourceChips() {
    return SizedBox(
      height: 36,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: EdgeInsets.zero,
        itemCount: _sourceLabels.length,
        separatorBuilder: (_, _) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          final label = _sourceLabels[index];
          final isSelected = label == 'All'
              ? _selectedSource == null
              : _sourceMap[label] == _selectedSource;
          return _buildChip(
            label: label,
            isSelected: isSelected,
            onTap: () => _onSourceSelected(label),
          );
        },
      ),
    );
  }

  // Grouping toggle: switches tab buckets between outlet baseline and
  // article-level RoBERTa classification.
  Widget _buildGroupingToggle() {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: isDark ? Colors.grey[850] : Colors.grey[100],
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
            color: isDark ? Colors.grey[700]! : Colors.grey[300]!),
      ),
      child: Row(
        children: [
          Expanded(
            child: _buildGroupingButton(
              label: 'By outlet',
              icon: Icons.source,
              isActive: _grouping == CompareGrouping.source,
              onTap: () =>
                  setState(() => _grouping = CompareGrouping.source),
            ),
          ),
          const SizedBox(width: 4),
          Expanded(
            child: _buildGroupingButton(
              label: 'By article',
              icon: Icons.article_outlined,
              isActive: _grouping == CompareGrouping.article,
              onTap: () =>
                  setState(() => _grouping = CompareGrouping.article),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildGroupingButton({
    required String label,
    required IconData icon,
    required bool isActive,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.symmetric(vertical: 7),
        decoration: BoxDecoration(
          color: isActive ? Colors.blue[700] : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon,
                size: 13,
                color: isActive ? Colors.white : Colors.grey[600]),
            const SizedBox(width: 5),
            Text(
              label,
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: isActive ? Colors.white : Colors.grey[700],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Bucketing helpers.

  List<Article> _toArticles(List<dynamic>? raw) {
    if (raw == null) return [];
    return raw
        .whereType<Map<String, dynamic>>()
        .map(Article.fromJson)
        .toList();
  }

  /// Merges all three server buckets into a single deduplicated list.
  /// Used when regrouping client-side by article-level RoBERTa label.
  List<Article> _allArticlesFromResults(Map<String, dynamic>? results) {
    if (results == null) return [];
    final merged = <Article>[];
    merged.addAll(_toArticles(results['left_articles'] as List<dynamic>?));
    merged.addAll(_toArticles(results['center_articles'] as List<dynamic>?));
    merged.addAll(_toArticles(results['right_articles'] as List<dynamic>?));

    // Server buckets should already be disjoint, but dedupe defensively.
    final seen = <String>{};
    return merged.where((a) => seen.add(a.id)).toList();
  }

  /// Buckets an article list into (left, centre, right) using each article's
  /// RoBERTa [politicalBias] label. Articles without a label fall back to
  /// their outlet-level [biasScore] so they still appear in the UI.
  ({List<Article> left, List<Article> centre, List<Article> right})
      _bucketByArticle(List<Article> articles) {
    final left = <Article>[];
    final centre = <Article>[];
    final right = <Article>[];

    for (final a in articles) {
      final label = a.politicalBias?.toUpperCase();
      if (label == 'LEFT') {
        left.add(a);
      } else if (label == 'RIGHT') {
        right.add(a);
      } else if (label == 'CENTER' || label == 'CENTRE') {
        centre.add(a);
      } else {
        // No RoBERTa label, fall back to source-level score.
        final score = a.biasScore;
        if (score == null) {
          centre.add(a);
        } else if (score < -0.3) {
          left.add(a);
        } else if (score > 0.3) {
          right.add(a);
        } else {
          centre.add(a);
        }
      }
    }

    return (left: left, centre: centre, right: right);
  }

  /// Returns the three buckets for the currently selected grouping mode.
  ({List<Article> left, List<Article> centre, List<Article> right})
      _currentBuckets() {
    if (_rawResults == null) {
      return (left: [], centre: [], right: []);
    }

    if (_grouping == CompareGrouping.source) {
      return (
        left: _toArticles(_rawResults!['left_articles'] as List<dynamic>?),
        centre:
            _toArticles(_rawResults!['center_articles'] as List<dynamic>?),
        right:
            _toArticles(_rawResults!['right_articles'] as List<dynamic>?),
      );
    }

    // Article-level grouping: merge server buckets and rebucket locally.
    final merged = _allArticlesFromResults(_rawResults);
    return _bucketByArticle(merged);
  }

  Widget _buildArticleList(List<Article> articles) {
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

  Widget _buildBiasTabBar(int leftCount, int centreCount, int rightCount) {
    final activeColor = _tabColors[_activeTab];
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      decoration: BoxDecoration(
        color: isDark ? Colors.grey[850] : Colors.grey[200],
        borderRadius: BorderRadius.circular(12),
      ),
      child: TabBar(
        controller: _tabController,
        labelColor: Colors.white,
        unselectedLabelColor: Colors.grey[600],
        indicator: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          color: activeColor,
        ),
        dividerColor: Colors.transparent,
        tabs: [
          _buildTab(
            icon: Icons.arrow_back,
            iconColor: const Color(0xFF1565C0),
            label: 'Left ($leftCount)',
            isSelected: _activeTab == 0,
          ),
          _buildTab(
            icon: Icons.horizontal_rule,
            iconColor: const Color(0xFF00796B),
            label: 'Centre ($centreCount)',
            isSelected: _activeTab == 1,
          ),
          _buildTab(
            icon: Icons.arrow_forward,
            iconColor: const Color(0xFFC62828),
            label: 'Right ($rightCount)',
            isSelected: _activeTab == 2,
          ),
        ],
      ),
    );
  }

  Tab _buildTab({
    required IconData icon,
    required Color iconColor,
    required String label,
    required bool isSelected,
  }) {
    return Tab(
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 14, color: isSelected ? Colors.white : iconColor),
          const SizedBox(width: 4),
          Text(label, style: const TextStyle(fontSize: 12)),
        ],
      ),
    );
  }

  Widget _buildResultsBody() {
    // Empty state: prompt user to pick a filter.
    if (_rawResults == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.compare_arrows, size: 64, color: Colors.grey[300]),
            const SizedBox(height: 16),
            Text(
              'Pick a category, outlet, or keyword\n'
              'to compare how the story is covered\n'
              'across the political spectrum.',
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

    final buckets = _currentBuckets();
    final total = buckets.left.length + buckets.centre.length + buckets.right.length;

    // Build a human-readable header from the active filters.
    final topic = _searchController.text.trim();
    final headerParts = <String>[];
    if (topic.isNotEmpty) headerParts.add('"$topic"');
    if (_selectedCategory != null && _selectedCategory!.isNotEmpty) {
      headerParts.add(
        _selectedCategory![0].toUpperCase() + _selectedCategory!.substring(1),
      );
    }
    if (_selectedSource != null) {
      final sourceLabel = _sourceMap.entries
          .firstWhere(
            (e) => e.value == _selectedSource,
            orElse: () => MapEntry(_selectedSource!, _selectedSource!),
          )
          .key;
      headerParts.add(sourceLabel);
    }
    final headerText =
        headerParts.isEmpty ? 'All articles' : headerParts.join(' · ');

    final groupingCaption = _grouping == CompareGrouping.source
        ? 'Grouped by outlet rating · tap "By article" for RoBERTa classification'
        : 'Grouped by article-level RoBERTa classification · '
            'reflects each article\'s own text';

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
                overflow: TextOverflow.ellipsis,
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
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
        const SizedBox(height: 10),
        _buildGroupingToggle(),
        const SizedBox(height: 6),
        Text(
          groupingCaption,
          style: TextStyle(
            fontSize: 10,
            color: Colors.grey[500],
            height: 1.4,
          ),
        ),
        const SizedBox(height: 10),
        _buildBiasTabBar(
            buckets.left.length, buckets.centre.length, buckets.right.length),
        const SizedBox(height: 8),
        Expanded(
          child: TabBarView(
            controller: _tabController,
            children: [
              _buildArticleList(buckets.left),
              _buildArticleList(buckets.centre),
              _buildArticleList(buckets.right),
            ],
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    // Button label adapts to whether the user has entered a keyword
    // or is just browsing a category/source.
    final hasCat = _selectedCategory != null && _selectedCategory!.isNotEmpty;
    final buttonLabel =
        (!_searchController.text.trim().isNotEmpty && _hasAnyFilter)
            ? 'Browse'
                '${hasCat ? " ${_selectedCategory![0].toUpperCase()}${_selectedCategory!.substring(1)}" : ""}'
                '${_selectedSource != null ? " · ${_sourceMap.entries.firstWhere((e) => e.value == _selectedSource, orElse: () => MapEntry(_selectedSource!, _selectedSource!)).key}" : ""}'
            : 'Compare Coverage';

    final isDark = Theme.of(context).brightness == Brightness.dark;

    return GestureDetector(
      // Tap outside any input to dismiss the keyboard.
      onTap: () => FocusScope.of(context).unfocus(),
      child: Scaffold(
        appBar: AppBar(
          centerTitle: true,
          title: Text(
            'Story Comparison',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.bold,
              letterSpacing: 0.3,
              color: Colors.blue[800],
            ),
          ),
          actions: [
            IconButton(
              icon: const Icon(Icons.refresh),
              tooltip: 'Refresh',
              onPressed: _hasAnyFilter && !_isLoading ? _searchTopic : null,
            ),
          ],
        ),
        body: Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Flexible(
                flex: 0,
                child: SingleChildScrollView(
                  physics: const ClampingScrollPhysics(),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const SizedBox(height: 4),
                      Text(
                        'Pick a category and / or outlet to get started',
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.grey[600],
                          height: 1.4,
                        ),
                      ),
                      const SizedBox(height: 10),
                      _buildFilterLabel('CATEGORY'),
                      const SizedBox(height: 6),
                      _buildCategoryChips(),
                      const SizedBox(height: 10),
                      _buildFilterLabel('OUTLET'),
                      const SizedBox(height: 6),
                      _buildSourceChips(),
                      const SizedBox(height: 14),
                      Row(
                        children: [
                          Expanded(child: Divider(color: Colors.grey[300])),
                          Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 12),
                            child: Text(
                              'or refine with a keyword',
                              style: TextStyle(
                                  fontSize: 11, color: Colors.grey[500]),
                            ),
                          ),
                          Expanded(child: Divider(color: Colors.grey[300])),
                        ],
                      ),
                      const SizedBox(height: 10),
                      TextField(
                        controller: _searchController,
                        focusNode: _searchFocusNode,
                        textInputAction: TextInputAction.search,
                        onTapOutside: (_) => _searchFocusNode.unfocus(),
                        decoration: InputDecoration(
                          filled: true,
                          fillColor: isDark ? Colors.grey[850] : Colors.white,
                          labelText: _hasAnyFilter
                              ? 'Add a keyword (optional)'
                              : 'Enter a topic to compare',
                          hintText: 'e.g., climate, housing, election',
                          prefixIcon: const Icon(Icons.search),
                          border: const OutlineInputBorder(),
                          isDense: true,
                          contentPadding: const EdgeInsets.symmetric(
                              vertical: 12, horizontal: 12),
                          suffixIcon: _searchController.text.isNotEmpty
                              ? IconButton(
                                  icon: const Icon(Icons.clear),
                                  onPressed: () {
                                    _searchController.clear();
                                    setState(() {});
                                    if (_hasAnyFilter) {
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
                      const SizedBox(height: 10),
                      ElevatedButton.icon(
                        onPressed: _isLoading ? null : _searchTopic,
                        icon: const Icon(Icons.compare_arrows),
                        label: Text(buttonLabel),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.blue[700],
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 13),
                          textStyle: const TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      const SizedBox(height: 10),
                    ],
                  ),
                ),
              ),
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
      ),
    );
  }
}
