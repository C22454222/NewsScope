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
        selectedItemColor: Colors.blue[700],
        unselectedItemColor: Colors.grey[500],
        backgroundColor: Colors.white,
        elevation: 8,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
          BottomNavigationBarItem(
            icon: Icon(Icons.compare_arrows),
            label: 'Spectrum',
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

// ── Home Feed Tab ─────────────────────────────────────────────────────────────

class HomeFeedTab extends StatefulWidget {
  final VoidCallback onArticleRead;

  const HomeFeedTab({super.key, required this.onArticleRead});

  @override
  State<HomeFeedTab> createState() => _HomeFeedTabState();
}

class _HomeFeedTabState extends State<HomeFeedTab> {
  final user = FirebaseAuth.instance.currentUser;
  final ApiService _apiService = ApiService();

  late Future<List<Article>> _articlesFuture;

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
    _articlesFuture = _apiService.getArticles();
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

  String get _timeOfDayGreeting {
    final hour = DateTime.now().hour;
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  }

  // ── AppBar title — single colour, readable on white AppBar ───────────────
  Widget _buildNewsScopeTitle() {
    return Text(
      'NewsScope',
      style: TextStyle(
        fontSize: 22,
        fontWeight: FontWeight.bold,
        letterSpacing: 0.5,
        color: Colors.blue[800],
      ),
    );
  }

  Widget _buildGreetingCard() {
    final name = user?.displayName ?? 'Reader';
    final initial = name[0].toUpperCase();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [Colors.blue.shade800, Colors.blue.shade500],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.blue.shade200.withAlpha(120),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '$_timeOfDayGreeting, $name! 👋',
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  "Here's the latest from across the spectrum.",
                  style: TextStyle(
                    color: Colors.white.withAlpha(210),
                    fontSize: 13,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          CircleAvatar(
            radius: 26,
            backgroundColor: Colors.white.withAlpha(50),
            child: Text(
              initial,
              style: const TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Category chips ─────────────────────────────────────────────────────────

  Widget _buildCategoryChips() {
    return SizedBox(
      height: 36,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        scrollDirection: Axis.horizontal,
        itemCount: _categories.length,
        separatorBuilder: (_, _) => const SizedBox(width: 8), // ← fixed
        itemBuilder: (context, index) {
          final label = _categories[index];
          final isSelected = (_selectedCategory ?? 'All') == label;
          return GestureDetector(
            onTap: () {
              setState(() {
                _selectedCategory = label == 'All' ? null : label;
                _loadArticles();
              });
            },
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              padding:
                  const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
              decoration: BoxDecoration(
                color: isSelected ? Colors.blue[700] : Colors.grey[100],
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color:
                      isSelected ? Colors.blue[700]! : Colors.grey[300]!,
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
                      fontWeight: isSelected
                          ? FontWeight.w600
                          : FontWeight.normal,
                      color:
                          isSelected ? Colors.white : Colors.grey[700],
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  // ── Date section header ────────────────────────────────────────────────────

  Widget _buildDateHeader(String headerText) {
    return Padding(
      padding: const EdgeInsets.only(top: 16.0, bottom: 8.0),
      child: Row(
        children: [
          Container(
            width: 4,
            height: 20,
            decoration: BoxDecoration(
              color: Colors.blue[700],
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(width: 8),
          Text(
            headerText,
            style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.bold,
              color: Colors.grey[800],
            ),
          ),
        ],
      ),
    );
  }

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: _buildNewsScopeTitle(),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _refreshArticles,
            tooltip: 'Refresh',
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
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 12),
            child: _buildGreetingCard(),
          ),
          _buildCategoryChips(),
          const SizedBox(height: 8),
          Expanded(
            child: FutureBuilder<List<Article>>(
              future: _articlesFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                }

                if (snapshot.hasError) {
                  return Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(Icons.error_outline,
                            size: 64, color: Colors.red),
                        const SizedBox(height: 16),
                        Text(
                          'Error: ${snapshot.error}',
                          textAlign: TextAlign.center,
                        ),
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
                        Icon(Icons.article_outlined,
                            size: 64, color: Colors.grey.shade300),
                        const SizedBox(height: 16),
                        Text(
                          'No articles found.',
                          style: TextStyle(color: Colors.grey[500]),
                        ),
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
                      final todayKey =
                          '${now.year}-${now.month}-${now.day}';
                      final yesterday =
                          now.subtract(const Duration(days: 1));
                      final yesterdayKey =
                          '${yesterday.year}-${yesterday.month}'
                          '-${yesterday.day}';

                      String headerText;
                      if (dateKey == todayKey) {
                        headerText = 'Today';
                      } else if (dateKey == yesterdayKey) {
                        headerText = 'Yesterday';
                      } else if (dateKey == 'Unknown Date') {
                        headerText = 'Unknown Date';
                      } else {
                        final parts = dateKey.split('-');
                        headerText =
                            '${parts[2]}/${parts[1]}/${parts[0]}';
                      }

                      return Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _buildDateHeader(headerText),
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
