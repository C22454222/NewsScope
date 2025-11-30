// lib/screens/home_screen.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../services/api_service.dart';
import '../screens/article_detail_screen.dart';
import '../screens/placeholders.dart';
import '../models/article.dart'; // IMPORT THE MODEL

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;

  final List<Widget> _screens = [
    const HomeFeedTab(),
    const CompareScreen(),
    const ProfileScreen(),
  ];

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
          BottomNavigationBarItem(icon: Icon(Icons.home), label: "Home"),
          BottomNavigationBarItem(icon: Icon(Icons.compare_arrows), label: "Compare"),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: "Profile"),
        ],
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
      ),
    );
  }
}

class HomeFeedTab extends StatefulWidget {
  const HomeFeedTab({super.key});

  @override
  State<HomeFeedTab> createState() => _HomeFeedTabState();
}

class _HomeFeedTabState extends State<HomeFeedTab> {
  final user = FirebaseAuth.instance.currentUser;
  final ApiService _apiService = ApiService();
  
  // Correctly typed Future
  late Future<List<Article>> _articlesFuture;

  @override
  void initState() {
    super.initState();
    _articlesFuture = _apiService.getArticles();
  }

  Future<void> _refreshArticles() async {
    setState(() {
      _articlesFuture = _apiService.getArticles();
    });
  }

  /// Helper to map bias score to UI colors
  Color _getBiasColor(double? score) {
    if (score == null) return Colors.grey[300]!;
    if (score < -0.1) return Colors.blue[700]!;
    if (score > 0.1) return Colors.red[700]!;
    return Colors.purple[400]!;
  }

  /// Helper to map bias score to text labels
  String _getBiasLabel(double? score) {
    if (score == null) return "Unscored";
    if (score < -0.1) return "Left";
    if (score > 0.1) return "Right";
    return "Center";
  }

  /// Helper to map sentiment score to UI colors
  Color _getSentimentColor(double? score) {
    if (score == null) return Colors.grey;
    if (score > 0.1) return Colors.green[700]!;
    if (score < -0.1) return Colors.orange[800]!;
    return Colors.grey[600]!;
  }

  /// Helper to map sentiment score to text labels
  String _getSentimentLabel(double? score) {
    if (score == null) return "--";
    if (score > 0.1) return "Positive";
    if (score < -0.1) return "Negative";
    return "Neutral";
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("NewsScope"),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _refreshArticles,
          ),
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: () async {
              await FirebaseAuth.instance.signOut();
            },
          ),
        ],
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Welcome Header
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Hello, ${user?.displayName ?? 'Reader'}!",
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
                const SizedBox(height: 4),
                Text(
                  "Latest articles from your feed.",
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
              ],
            ),
          ),

          // Article Feed List (Grouped by Date)
          Expanded(
            child: FutureBuilder<List<Article>>(
              future: _articlesFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                } else if (snapshot.hasError) {
                  return Center(child: Text("Error: ${snapshot.error}"));
                } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
                  return const Center(child: Text("No articles found."));
                }

                final articles = snapshot.data!;

                // --- GROUPING LOGIC START ---
                final groupedArticles = <String, List<Article>>{};
                for (var article in articles) {
                  final date = article.publishedAt;
                  // Create a simple date key YYYY-MM-DD
                  final key = date != null 
                      ? "${date.year}-${date.month}-${date.day}" 
                      : "Unknown Date";
                  
                  if (!groupedArticles.containsKey(key)) {
                    groupedArticles[key] = [];
                  }
                  groupedArticles[key]!.add(article);
                }
                
                // Sort keys so newest dates are first
                final sortedKeys = groupedArticles.keys.toList()
                  ..sort((a, b) {
                    if (a == "Unknown Date") return 1; // Put unknowns at bottom
                    if (b == "Unknown Date") return -1;
                    // Parse back to compare dates roughly
                    // (Simple string compare works for YYYY-M-D if padded, but let's be safe)
                    // Actually, for this prototype, simple string compare is risky if months aren't padded.
                    // Better to just reverse the list since we likely fetch newest first anyway.
                    // Or parse:
                    final partsA = a.split('-');
                    final partsB = b.split('-');
                    final dA = DateTime(int.parse(partsA[0]), int.parse(partsA[1]), int.parse(partsA[2]));
                    final dB = DateTime(int.parse(partsB[0]), int.parse(partsB[1]), int.parse(partsB[2]));
                    return dB.compareTo(dA);
                  });
                // --- GROUPING LOGIC END ---

                return RefreshIndicator(
                  onRefresh: _refreshArticles,
                  child: ListView.builder(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    itemCount: sortedKeys.length,
                    itemBuilder: (context, sectionIndex) {
                      final dateKey = sortedKeys[sectionIndex];
                      final sectionArticles = groupedArticles[dateKey]!;
                      
                      // Format Header Text
                      String headerText = dateKey;
                      final now = DateTime.now();
                      // Match dateKey format YYYY-M-D (no padding usually)
                      final todayKey = "${now.year}-${now.month}-${now.day}";
                      final yesterday = now.subtract(const Duration(days: 1));
                      final yesterdayKey = "${yesterday.year}-${yesterday.month}-${yesterday.day}";

                      if (dateKey == todayKey) {
                        headerText = "Today";
                      } else if (dateKey == yesterdayKey) {
                        headerText = "Yesterday";
                      } else if (dateKey != "Unknown Date") {
                        // Make it look nicer: "30/11/2025"
                        final parts = dateKey.split('-');
                        headerText = "${parts[2]}/${parts[1]}/${parts[0]}";
                      }

                      return Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Date Header
                          Padding(
                            padding: const EdgeInsets.only(top: 16.0, bottom: 8.0),
                            child: Text(
                              headerText,
                              style: TextStyle(
                                fontSize: 18, 
                                fontWeight: FontWeight.bold,
                                color: Colors.grey[800]
                              ),
                            ),
                          ),
                          // List of Cards for this Date
                          ...sectionArticles.map((article) {
                            // Extract Data
                            final biasScore = article.biasScore;
                            final sentimentScore = article.sentimentScore;
                            final sourceName = article.source;
                            final url = article.url;
                            final content = article.content;
                            String title = article.title;

                            // Cleanup Title
                            if ((title.startsWith('http') || title.contains('.html')) && url.isNotEmpty) {
                              final uri = Uri.tryParse(url);
                              if (uri != null && uri.pathSegments.isNotEmpty) {
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
                                onTap: () {
                                  Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                      builder: (_) => ArticleDetailScreen(
                                        title: title,
                                        sourceName: sourceName,
                                        content: content,
                                        url: url,
                                        biasScore: biasScore,
                                        sentimentScore: sentimentScore,
                                      ),
                                    ),
                                  );
                                },
                                child: Padding(
                                  padding: const EdgeInsets.all(16),
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
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
                                      const SizedBox(height: 8),

                                      // Source Name
                                      Row(
                                        children: [
                                          Icon(Icons.source, size: 14, color: Colors.grey[600]),
                                          const SizedBox(width: 4),
                                          Text(
                                            sourceName,
                                            style: TextStyle(fontSize: 12, color: Colors.grey[700]),
                                          ),
                                        ],
                                      ),
                                      const SizedBox(height: 12),

                                      // Badges Row
                                      Row(
                                        children: [
                                          // Bias Badge
                                          if (biasScore != null)
                                            Container(
                                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                              decoration: BoxDecoration(
                                                color: _getBiasColor(biasScore).withValues(alpha: 0.15),
                                                borderRadius: BorderRadius.circular(12),
                                                border: Border.all(color: _getBiasColor(biasScore), width: 1),
                                              ),
                                              child: Text(
                                                _getBiasLabel(biasScore),
                                                style: TextStyle(
                                                  fontSize: 11,
                                                  fontWeight: FontWeight.bold,
                                                  color: _getBiasColor(biasScore),
                                                ),
                                              ),
                                            ),
                                          
                                          const SizedBox(width: 8),

                                          // Sentiment Badge
                                          if (sentimentScore != null)
                                            Container(
                                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                              decoration: BoxDecoration(
                                                color: _getSentimentColor(sentimentScore).withValues(alpha: 0.15),
                                                borderRadius: BorderRadius.circular(12),
                                                border: Border.all(color: _getSentimentColor(sentimentScore), width: 1),
                                              ),
                                              child: Row(
                                                mainAxisSize: MainAxisSize.min,
                                                children: [
                                                  Icon(
                                                    sentimentScore > 0 ? Icons.sentiment_satisfied : Icons.sentiment_dissatisfied,
                                                    size: 12,
                                                    color: _getSentimentColor(sentimentScore),
                                                  ),
                                                  const SizedBox(width: 4),
                                                  Text(
                                                    _getSentimentLabel(sentimentScore),
                                                    style: TextStyle(
                                                      fontSize: 11,
                                                      fontWeight: FontWeight.bold,
                                                      color: _getSentimentColor(sentimentScore),
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            ),
                                        ],
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            );
                          }).toList(), // Convert the map to a List<Widget>
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
