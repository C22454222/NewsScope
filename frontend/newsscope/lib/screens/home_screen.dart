// lib/widgets/home_screen.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../services/api_service.dart';
import '../screens/article_detail_screen.dart';
import '../screens/placeholders.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0; // Tracks the currently active bottom tab

  // List of primary screens for the bottom navigation
  final List<Widget> _screens = [
    const HomeFeedTab(),
    const CompareScreen(),
    const ProfileScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // IndexedStack preserves the state of each tab when switching
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

// Extracted Home Feed logic into a separate widget for modularity
class HomeFeedTab extends StatefulWidget {
  const HomeFeedTab({super.key});

  @override
  State<HomeFeedTab> createState() => _HomeFeedTabState();
}

class _HomeFeedTabState extends State<HomeFeedTab> {
  final user = FirebaseAuth.instance.currentUser;
  final ApiService _apiService = ApiService();
  late Future<List<dynamic>> _articlesFuture;

  @override
  void initState() {
    super.initState();
    // Initialize the fetch operation once when the widget loads
    _articlesFuture = _apiService.getArticles();
  }

  Future<void> _refreshArticles() async {
    setState(() {
      // Re-trigger the API call to refresh data
      _articlesFuture = _apiService.getArticles();
    });
  }

  /// Helper to map bias score to UI colors
  Color _getBiasColor(double? score) {
    if (score == null) return Colors.grey[300]!; // Neutral placeholder
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

          // Article Feed List
          Expanded(
            child: FutureBuilder<List<dynamic>>(
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
                return RefreshIndicator(
                  onRefresh: _refreshArticles,
                  child: ListView.builder(
                    itemCount: articles.length,
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    itemBuilder: (context, index) {
                      final article = articles[index];
                      
                      // Safely parse scores (backend can send int or double)
                      final biasScore = article['bias_score'] != null 
                          ? (article['bias_score'] as num).toDouble() 
                          : null;
                      final sentimentScore = article['sentiment_score'] != null 
                          ? (article['sentiment_score'] as num).toDouble() 
                          : null;
                      
                      final sourceName = article['source'] ?? article['source_name'] ?? 'Unknown Source';
                      final url = article['url'] ?? '';
                      final content = article['content'] ?? 'No content available.';
                      String title = article['title'] ?? 'Article ${index + 1}';

                      // Clean up filenames/URLs appearing as titles
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
