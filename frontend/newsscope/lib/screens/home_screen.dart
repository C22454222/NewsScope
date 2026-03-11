// lib/screens/home_screen.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../models/article.dart';
import '../services/api_service.dart';
import '../screens/article_detail_screen.dart';
import '../screens/compare_screen.dart';
import '../screens/profile_screen.dart';
import '../widgets/article_card.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;
  int _profileKey = 0;

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

  void _handleArticleRead() {
    setState(() {
      _profileKey++;
      _screens[2] = ProfileScreen(key: ValueKey(_profileKey));
    });
  }

  void _onTabTapped(int index) {
    setState(() => _currentIndex = index);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _currentIndex, children: _screens),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
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

  const HomeFeedTab({super.key, required this.onArticleRead});

  @override
  State<HomeFeedTab> createState() => _HomeFeedTabState();
}

class _HomeFeedTabState extends State<HomeFeedTab> {
  final user = FirebaseAuth.instance.currentUser;
  final ApiService _apiService = ApiService();

  Future<List<Article>>? _articlesFuture;

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
  void initState() {
    super.initState();
    Future.delayed(const Duration(milliseconds: 100), () {
      if (mounted) {
        setState(() => _articlesFuture = _apiService.getArticles());
      }
    });
  }

  void _loadArticles() {
    final backendCategory =
        (_selectedCategory == null || _selectedCategory == 'All')
            ? null
            : _selectedCategory!.toLowerCase();
    _articlesFuture = _apiService.getArticles(category: backendCategory);
  }

  Future<void> _refreshArticles() async {
    setState(_loadArticles);
  }

  Future<void> _handleLogout() async {
    await FirebaseAuth.instance.signOut();
    if (!mounted) return;
    Navigator.of(context).popUntil((route) => route.isFirst);
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
                }

                if (snapshot.hasError) {
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
                }

                if (!snapshot.hasData || snapshot.data!.isEmpty) {
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
                final grouped = <String, List<Article>>{};

                for (final article in articles) {
                  final date = article.publishedAt;
                  final key = date != null
                      ? '${date.year}-${date.month}-${date.day}'
                      : 'Unknown Date';
                  grouped.putIfAbsent(key, () => []).add(article);
                }

                final sortedKeys = grouped.keys.toList()
                  ..sort((a, b) {
                    if (a == 'Unknown Date') return 1;
                    if (b == 'Unknown Date') return -1;
                    final pa = a.split('-');
                    final pb = b.split('-');
                    return DateTime(
                      int.parse(pb[0]),
                      int.parse(pb[1]),
                      int.parse(pb[2]),
                    ).compareTo(DateTime(
                      int.parse(pa[0]),
                      int.parse(pa[1]),
                      int.parse(pa[2]),
                    ));
                  });

                return RefreshIndicator(
                  onRefresh: _refreshArticles,
                  child: ListView.builder(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    itemCount: sortedKeys.length,
                    itemBuilder: (context, i) {
                      final dateKey = sortedKeys[i];
                      final sectionArticles = grouped[dateKey]!;

                      final now = DateTime.now();
                      final todayKey = '${now.year}-${now.month}-${now.day}';
                      final yesterday =
                          now.subtract(const Duration(days: 1));
                      final yesterdayKey =
                          '${yesterday.year}-${yesterday.month}-${yesterday.day}';

                      String headerText;
                      if (dateKey == todayKey) {
                        headerText = 'Today';
                      } else if (dateKey == yesterdayKey) {
                        headerText = 'Yesterday';
                      } else if (dateKey == 'Unknown Date') {
                        headerText = 'Unknown Date';
                      } else {
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
                          // ArticleCard widget replaces the inline card — Bug 4 fix
                          // (fromArticle passes all credibility + fact-check fields).
                          ...sectionArticles.map(
                            (article) => ArticleCard(
                              article: article,
                              onTap: () async {
                                await Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (_) =>
                                        ArticleDetailScreen.fromArticle(
                                      article,
                                    ),
                                  ),
                                );
                                widget.onArticleRead();
                              },
                            ),
                          ),
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
