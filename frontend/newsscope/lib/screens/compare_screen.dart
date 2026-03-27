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

class _CompareScreenState extends State<CompareScreen>
    with SingleTickerProviderStateMixin {
  final TextEditingController _searchController = TextEditingController();
  final FocusNode _searchFocusNode = FocusNode();
  final ApiService _apiService = ApiService();

  late TabController _tabController;

  Map<String, dynamic>? _rawResults;
  bool _isLoading = false;
  String? _errorMessage;
  int _activeTab = 0;

  String? _selectedCategory;
  String? _selectedSource;

  // ── Soft background colour ─────────────────────────────────────────────────
  static const Color _scaffoldBg = Color(0xFFF0F2F5);

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

  bool get _hasAnyFilter =>
      _searchController.text.trim().isNotEmpty ||
      (_selectedCategory != null && _selectedCategory!.isNotEmpty) ||
      _selectedSource != null;

  Future<void> _searchTopic() async {
    FocusScope.of(context).unfocus();

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
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected ? Colors.blue[700] : Colors.grey[100],
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isSelected ? Colors.blue[700]! : Colors.grey[300]!,
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
                color: isSelected ? Colors.white : Colors.grey[700],
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

  List<Article> _toArticles(List<dynamic>? raw) {
    if (raw == null) return [];
    return raw.whereType<Map<String, dynamic>>().map(Article.fromJson).toList();
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

  Widget _buildBiasTabBar(
    List<dynamic>? l,
    List<dynamic>? c,
    List<dynamic>? r,
  ) {
    final activeColor = _tabColors[_activeTab];
    return Container(
      decoration: BoxDecoration(
        color: Colors.grey[200],
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
            label: 'Left (${l?.length ?? 0})',
            isSelected: _activeTab == 0,
          ),
          _buildTab(
            icon: Icons.horizontal_rule,
            iconColor: const Color(0xFF00796B),
            label: 'Centre (${c?.length ?? 0})',
            isSelected: _activeTab == 1,
          ),
          _buildTab(
            icon: Icons.arrow_forward,
            iconColor: const Color(0xFFC62828),
            label: 'Right (${r?.length ?? 0})',
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

    final leftArticles = _rawResults!['left_articles'] as List<dynamic>?;
    final centreArticles = _rawResults!['center_articles'] as List<dynamic>?;
    final rightArticles = _rawResults!['right_articles'] as List<dynamic>?;
    final total = _rawResults!['total_found'] ?? 0;

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
        const SizedBox(height: 12),
        _buildBiasTabBar(leftArticles, centreArticles, rightArticles),
        const SizedBox(height: 8),
        Expanded(
          child: TabBarView(
            controller: _tabController,
            children: [
              _buildArticleList(leftArticles),
              _buildArticleList(centreArticles),
              _buildArticleList(rightArticles),
            ],
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final hasCat = _selectedCategory != null && _selectedCategory!.isNotEmpty;
    final buttonLabel =
        (!_searchController.text.trim().isNotEmpty && _hasAnyFilter)
            ? 'Browse'
                '${hasCat ? " ${_selectedCategory![0].toUpperCase()}${_selectedCategory!.substring(1)}" : ""}'
                '${_selectedSource != null ? " · ${_sourceMap.entries.firstWhere((e) => e.value == _selectedSource, orElse: () => MapEntry(_selectedSource!, _selectedSource!)).key}" : ""}'
            : 'Compare Coverage';

    return GestureDetector(
      onTap: () => FocusScope.of(context).unfocus(),
      child: Scaffold(
        // FIX: Consistent soft background
        backgroundColor: _scaffoldBg,
        // FIX: resizeToAvoidBottomInset=false prevents the keyboard from
        // pushing the fixed filter/button area and causing an overflow.
        // The results area (Expanded) shrinks naturally with the keyboard.
        resizeToAvoidBottomInset: false,
        appBar: AppBar(
          backgroundColor: _scaffoldBg,
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
        ),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
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
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(child: Divider(color: Colors.grey[300])),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    child: Text(
                      'or refine with a keyword',
                      style:
                          TextStyle(fontSize: 11, color: Colors.grey[500]),
                    ),
                  ),
                  Expanded(child: Divider(color: Colors.grey[300])),
                ],
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _searchController,
                focusNode: _searchFocusNode,
                textInputAction: TextInputAction.search,
                onTapOutside: (_) => _searchFocusNode.unfocus(),
                decoration: InputDecoration(
                  filled: true,
                  fillColor: Colors.white,
                  labelText: _hasAnyFilter
                      ? 'Add a keyword (optional)'
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
              // FIX: Results area — keyboard does NOT push this section because
              // resizeToAvoidBottomInset=false. The Expanded fills remaining space.
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.only(top: 16),
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
              ),
            ],
          ),
        ),
      ),
    );
  }
}
