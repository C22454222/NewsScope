import 'dart:async';

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
            // RENAMED: Spectrum → Compare
            icon: Icon(Icons.compare_arrows),
            label: 'Compare',
          ),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Profile'),
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
  User? _user;
  StreamSubscription<User?>? _userSub;

  final ApiService _apiService = ApiService();
  late Future<List<Article>> _articlesFuture;

  String? _selectedCategory;
  final List<String> _categories = const [
    'All', 'Politics', 'World', 'Business',
    'Tech', 'Sport', 'Entertainment', 'Health', 'Science',
  ];

  @override
  void initState() {
    super.initState();
    _user = FirebaseAuth.instance.currentUser;
    _userSub = FirebaseAuth.instance.userChanges().listen((u) {
      if (mounted) setState(() => _user = u);
    });
    _articlesFuture = _apiService.getArticles();
  }

  @override
  void dispose() {
    _userSub?.cancel();
    super.dispose();
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

  Widget _buildNewsScopeTitle() {
    return RichText(
      text: TextSpan(
        style: const TextStyle(
          fontSize: 22,
          fontWeight: FontWeight.bold,
          letterSpacing: 0.5,
        ),
        children: [
          const TextSpan(
            text: 'News',
            style: TextStyle(color: Colors.black87),
          ),
          TextSpan(
            text: 'Scope',
            style: TextStyle(color: Colors.blue[800]),
          ),
        ],
      ),
    );
  }

  Widget _buildGreetingCard() {
    final name = _user?.displayName ?? 'Reader';
    final now = DateTime.now();
    final dayNames = [
      'Monday', 'Tuesday', 'Wednesday', 'Thursday',
      'Friday', 'Saturday', 'Sunday'
    ];
    final monthNames = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];
    final dayName = dayNames[now.weekday - 1];
    final monthName = monthNames[now.month - 1];

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 18),
      decoration: BoxDecoration(
        color: const Color(0xFF0D1B3E),
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withAlpha(50),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  // REMOVED: 👋 emoji
                  '$_timeOfDayGreeting, $name',
                  style: const TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 5),
                Text(
                  // UPDATED: subtitle text
                  "Today's news, AI-analysed for bias and sentiment.",
                  style: TextStyle(
                    color: Colors.white.withAlpha(160),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 16),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                dayName.toUpperCase(),
                style: TextStyle(
                  color: Colors.white.withAlpha(130),
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 1.0,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                '${now.day} $monthName',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildCategoryChips() {
    return SizedBox(
      height: 36,
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        scrollDirection: Axis.horizontal,
        itemCount: _categories.length,
        separatorBuilder: (_, _) => const SizedBox(width: 8),
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
                      fontWeight:
                          isSelected ? FontWeight.w600 : FontWeight.normal,
                      color: isSelected ? Colors.white : Colors.grey[700],
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

  String _dateKey(DateTime dt) {
    return '${dt.year}-${dt.month.toString().padLeft(2, '0')}-'
        '${dt.day.toString().padLeft(2, '0')}';
  }

  String _headerText(String key, String todayKey, String yesterdayKey) {
    if (key == todayKey) return 'Today';
    if (key == yesterdayKey) return 'Yesterday';
    if (key == 'Unknown Date') return 'Unknown Date';
    final parts = key.split('-');
    return '${parts[2]}/${parts[1]}/${parts[0]}';
  }

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
                        Text('Error: ${snapshot.error}',
                            textAlign: TextAlign.center),
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
                        Text('No articles found.',
                            style: TextStyle(color: Colors.grey[500])),
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
                final now = DateTime.now();
                final todayKey = _dateKey(now);
                final yesterdayKey =
                    _dateKey(now.subtract(const Duration(days: 1)));

                final grouped = <String, List<Article>>{};
                for (final article in articles) {
                  final key = article.publishedAt != null
                      ? _dateKey(article.publishedAt!)
                      : 'Unknown Date';
                  grouped.putIfAbsent(key, () => []).add(article);
                }

                final sortedKeys = grouped.keys.toList()
                  ..sort((a, b) {
                    if (a == 'Unknown Date') return 1;
                    if (b == 'Unknown Date') return -1;
                    return b.compareTo(a);
                  });

                return RefreshIndicator(
                  onRefresh: _refreshArticles,
                  child: ListView.builder(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    itemCount: sortedKeys.length,
                    itemBuilder: (context, i) {
                      final dateKey = sortedKeys[i];
                      final sectionArticles = grouped[dateKey]!;
                      final header =
                          _headerText(dateKey, todayKey, yesterdayKey);
                      return Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          _buildDateHeader(header),
                          ...sectionArticles.map(
                            (article) => ArticleCard(
                              article: article,
                              onTap: () async {
                                await Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (_) =>
                                        ArticleDetailScreen.fromArticle(
                                            article),
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
