// lib/screens/home_screen.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../services/api_service.dart';
import '../screens/article_detail_screen.dart';
import '../screens/compare_screen.dart';
import '../screens/profile_screen.dart';
import '../models/article.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;
  int _profileKey = 0; // Key to force ProfileScreen rebuild

  late List<Widget> _screens;

  @override
  void initState() {
    super.initState();
    _screens = [
      HomeFeedTab(onArticleRead: _handleArticleRead),
      CompareScreen(onArticleRead: _handleArticleRead),
      ProfileScreen(key: ValueKey(_profileKey)),
    ];
  }

  // Called when an article is read from any screen
  void _handleArticleRead() {
    setState(() {
      _profileKey++; // Increment key to force ProfileScreen rebuild
      _screens[2] = ProfileScreen(key: ValueKey(_profileKey));
    });
  }

  void _onTabTapped(int index) {
    setState(() {
      _currentIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.compare_arrows),
            label: 'Compare',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person),
            label: 'Profile',
          ),
        ],
        onTap: _onTabTapped,
      ),
    );
  }
}

class HomeFeedTab extends StatefulWidget {
  final VoidCallback onArticleRead;

  const HomeFeedTab({
    super.key,
    required this.onArticleRead,
  });

  @override
  State<HomeFeedTab> createState() => _HomeFeedTabState();
}

class _HomeFeedTabState extends State<HomeFeedTab> {
  final user = FirebaseAuth.instance.currentUser;
  final ApiService _apiService = ApiService();

  Future<List<Article>>? _articlesFuture;

  // NEW: category filter state
  String? _selectedCategory; // null = all
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
  void initState() {
    super.initState();
    Future.delayed(const Duration(milliseconds: 100), () {
      if (mounted) {
        setState(() {
          _articlesFuture = _apiService.getArticles();
        });
      }
    });
  }

  void _loadArticles() {
    // Backend expects lowercase category keys (politics, world, etc.)
    final backendCategory = _selectedCategory == null || _selectedCategory == 'All'
        ? null
        : _selectedCategory!.toLowerCase();

    _articlesFuture = _apiService.getArticles(category: backendCategory);
  }

  Future<void> _refreshArticles() async {
    setState(() {
      _loadArticles();
    });
  }

  Future<void> _handleLogout() async {
    await FirebaseAuth.instance.signOut();
    if (!mounted) return;
    Navigator.of(context).popUntil((route) => route.isFirst);
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

  Color _getSentimentColor(double? score) {
    if (score == null) return Colors.grey;
    if (score > 0.1) return Colors.green[700]!;
    if (score < -0.1) return Colors.orange[800]!;
    return Colors.grey[600]!;
  }

  String _getSentimentLabel(double? score) {
    if (score == null) return '--';
    if (score > 0.1) return 'Positive';
    if (score < -0.1) return 'Negative';
    return 'Neutral';
  }

  String _formatDate(DateTime? dt) {
    if (dt == null) return 'Unknown date';
    return '${dt.day.toString().padLeft(2, '0')}/'
        '${dt.month.toString().padLeft(2, '0')}/'
        '${dt.year}';
  }

  String _formatCategory(String? category) {
    if (category == null || category.isEmpty) return 'General';
    final c = category.toLowerCase();
    return c[0].toUpperCase() + c.substring(1);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('NewsScope'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _refreshArticles,
          ),
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: _handleLogout,
            tooltip: 'Sign Out',
          ),
        ],
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Greeting
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Hello, ${user?.displayName ?? 'Reader'}!',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Latest articles from your feed.',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ),
          ),

          // NEW: Horizontal category selector
          SizedBox(
            height: 40,
            child: ListView.separated(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              scrollDirection: Axis.horizontal,
              itemCount: _categories.length,
              separatorBuilder: (_, _) => const SizedBox(width: 8),
              itemBuilder: (context, index) {
                final label = _categories[index];
                final isSelected = (_selectedCategory ?? 'All') == label;
                return ChoiceChip(
                  label: Text(label),
                  selected: isSelected,
                  onSelected: (_) {
                    setState(() {
                      _selectedCategory = label == 'All' ? null : label;
                      _loadArticles();
                    });
                  },
                );
              },
            ),
          ),

          const SizedBox(height: 8),

          Expanded(
            child: FutureBuilder<List<Article>>(
              future: _articlesFuture ?? Future.value([]),
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                } else if (snapshot.hasError) {
                  return Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(
                          Icons.error_outline,
                          size: 64,
                          color: Colors.red,
                        ),
                        const SizedBox(height: 16),
                        Text('Error: ${snapshot.error}'),
                        const SizedBox(height: 16),
                        ElevatedButton(
                          onPressed: _refreshArticles,
                          child: const Text('Retry'),
                        ),
                      ],
                    ),
                  );
                } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
                  return Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.article_outlined,
                          size: 64,
                          color: Colors.grey.shade400,
                        ),
                        const SizedBox(height: 16),
                        const Text('No articles found.'),
                        const SizedBox(height: 16),
                        ElevatedButton(
                          onPressed: _refreshArticles,
                          child: const Text('Refresh'),
                        ),
                      ],
                    ),
                  );
                }

                final articles = snapshot.data!;
                final groupedArticles = <String, List<Article>>{};

                // Group by date (YYYY-M-D) so every article has a date section
                for (var article in articles) {
                  final date = article.publishedAt;
                  final key = date != null
                      ? '${date.year}-${date.month}-${date.day}'
                      : 'Unknown Date';

                  groupedArticles.putIfAbsent(key, () => []);
                  groupedArticles[key]!.add(article);
                }

                final sortedKeys = groupedArticles.keys.toList()
                  ..sort((a, b) {
                    if (a == 'Unknown Date') return 1;
                    if (b == 'Unknown Date') return -1;
                    final partsA = a.split('-');
                    final partsB = b.split('-');
                    final dA = DateTime(
                      int.parse(partsA[0]),
                      int.parse(partsA[1]),
                      int.parse(partsA[2]),
                    );
                    final dB = DateTime(
                      int.parse(partsB[0]),
                      int.parse(partsB[1]),
                      int.parse(partsB[2]),
                    );
                    return dB.compareTo(dA);
                  });

                return RefreshIndicator(
                  onRefresh: _refreshArticles,
                  child: ListView.builder(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    itemCount: sortedKeys.length,
                    itemBuilder: (context, sectionIndex) {
                      final dateKey = sortedKeys[sectionIndex];
                      final sectionArticles = groupedArticles[dateKey]!;

                      String headerText = dateKey;
                      final now = DateTime.now();
                      final todayKey =
                          '${now.year}-${now.month}-${now.day}';
                      final yesterday =
                          now.subtract(const Duration(days: 1));
                      final yesterdayKey =
                          '${yesterday.year}-${yesterday.month}-${yesterday.day}';

                      if (dateKey == todayKey) {
                        headerText = 'Today';
                      } else if (dateKey == yesterdayKey) {
                        headerText = 'Yesterday';
                      } else if (dateKey != 'Unknown Date') {
                        final parts = dateKey.split('-');
                        headerText = '${parts[2]}/${parts[1]}/${parts[0]}';
                      }

                      return Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Padding(
                            padding: const EdgeInsets.only(
                              top: 16.0,
                              bottom: 8.0,
                            ),
                            child: Text(
                              headerText,
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Colors.grey[800],
                              ),
                            ),
                          ),
                          ...sectionArticles.map((article) {
                            final biasScore = article.biasScore;
                            final biasIntensity = article.biasIntensity;
                            final sentimentScore = article.sentimentScore;
                            final sourceName = article.source;
                            final url = article.url;
                            final content = article.content;
                            final publishedAt = article.publishedAt;
                            final category = article.category; // NEW
                            String title = article.title;

                            if ((title.startsWith('http') ||
                                    title.contains('.html')) &&
                                url.isNotEmpty) {
                              final uri = Uri.tryParse(url);
                              if (uri != null &&
                                  uri.pathSegments.isNotEmpty) {
                                String pathSegment = uri.pathSegments.last;
                                title = pathSegment
                                    .replaceAll('.html', '')
                                    .replaceAll('.htm', '')
                                    .replaceAll('-', ' ')
                                    .replaceAll('_', ' ');
                              }
                            }

                            return Card(
                              elevation: 2,
                              margin: const EdgeInsets.only(bottom: 12),
                              child: InkWell(
                                onTap: () async {
                                  // Wait for article screen to close
                                  await Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                      builder: (_) => ArticleDetailScreen(
                                        id: article.id,
                                        title: title,
                                        sourceName: sourceName,
                                        content: content,
                                        url: url,
                                        biasScore: biasScore,
                                        biasIntensity: biasIntensity,
                                        sentimentScore: sentimentScore,
                                      ),
                                    ),
                                  );
                                  // Trigger profile reload
                                  widget.onArticleRead();
                                },
                                child: Padding(
                                  padding: const EdgeInsets.all(16),
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      // Title
                                      Text(
                                        title,
                                        style: const TextStyle(
                                          fontWeight: FontWeight.bold,
                                          fontSize: 15,
                                        ),
                                        maxLines: 2,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                      const SizedBox(height: 6),

                                      // Source + date + category
                                      Row(
                                        children: [
                                          Icon(
                                            Icons.source,
                                            size: 14,
                                            color: Colors.grey[600],
                                          ),
                                          const SizedBox(width: 4),
                                          Expanded(
                                            child: Text(
                                              sourceName,
                                              style: TextStyle(
                                                fontSize: 12,
                                                color: Colors.grey[700],
                                              ),
                                              overflow: TextOverflow.ellipsis,
                                            ),
                                          ),
                                          const SizedBox(width: 8),
                                          Text(
                                            _formatDate(publishedAt),
                                            style: TextStyle(
                                              fontSize: 11,
                                              color: Colors.grey[600],
                                            ),
                                          ),
                                        ],
                                      ),

                                      const SizedBox(height: 4),

                                      // Category label
                                      if (category != null &&
                                          category.isNotEmpty)
                                        Text(
                                          _formatCategory(category),
                                          style: TextStyle(
                                            fontSize: 11,
                                            fontWeight: FontWeight.w500,
                                            color: Colors.blueGrey[700],
                                          ),
                                        ),

                                      const SizedBox(height: 12),

                                      // Bias / Sentiment chips
                                      Row(
                                        children: [
                                          if (biasScore != null)
                                            Container(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                horizontal: 8,
                                                vertical: 4,
                                              ),
                                              decoration: BoxDecoration(
                                                color: _getBiasColor(
                                                  biasScore,
                                                ).withAlpha(
                                                    (255 * 0.15).round()),
                                                borderRadius:
                                                    BorderRadius.circular(12),
                                                border: Border.all(
                                                  color: _getBiasColor(
                                                      biasScore),
                                                  width: 1,
                                                ),
                                              ),
                                              child: Text(
                                                _getBiasLabel(biasScore),
                                                style: TextStyle(
                                                  fontSize: 11,
                                                  fontWeight: FontWeight.bold,
                                                  color: _getBiasColor(
                                                      biasScore),
                                                ),
                                              ),
                                            ),
                                          const SizedBox(width: 8),
                                          if (sentimentScore != null)
                                            Container(
                                              padding:
                                                  const EdgeInsets.symmetric(
                                                horizontal: 8,
                                                vertical: 4,
                                              ),
                                              decoration: BoxDecoration(
                                                color: _getSentimentColor(
                                                  sentimentScore,
                                                ).withAlpha(
                                                    (255 * 0.15).round()),
                                                borderRadius:
                                                    BorderRadius.circular(12),
                                                border: Border.all(
                                                  color: _getSentimentColor(
                                                      sentimentScore),
                                                  width: 1,
                                                ),
                                              ),
                                              child: Row(
                                                mainAxisSize: MainAxisSize.min,
                                                children: [
                                                  Icon(
                                                    sentimentScore > 0
                                                        ? Icons
                                                            .sentiment_satisfied
                                                        : Icons
                                                            .sentiment_dissatisfied,
                                                    size: 12,
                                                    color: _getSentimentColor(
                                                        sentimentScore),
                                                  ),
                                                  const SizedBox(width: 4),
                                                  Text(
                                                    _getSentimentLabel(
                                                        sentimentScore),
                                                    style: TextStyle(
                                                      fontSize: 11,
                                                      fontWeight:
                                                          FontWeight.bold,
                                                      color:
                                                          _getSentimentColor(
                                                              sentimentScore),
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            ),
                                          const SizedBox(width: 8),
                                          if (biasIntensity != null)
                                            Text(
                                              '${(biasIntensity * 100).round()}% biased',
                                              style: TextStyle(
                                                fontSize: 11,
                                                color: Colors.grey[600],
                                              ),
                                            ),
                                        ],
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            );
                          }),
                        ],
                      );
                    },
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
