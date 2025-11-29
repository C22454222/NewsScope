// lib/screens/home_screen.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../services/api_service.dart';
import 'screens/article_detail_screen.dart';
import 'screens/placeholders.dart'; // Import the new placeholders

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
    if (score < -0.3) return Colors.blue[300]!;
    if (score > 0.3) return Colors.red[300]!;
    return Colors.purple[200]!;
  }

  /// Helper to map bias score to text labels
  String _getBiasLabel(double? score) {
    if (score == null) return "Pending Analysis";
    if (score < -0.3) return "Left Leaning";
    if (score > 0.3) return "Right Leaning";
    return "Center";
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
                      final biasScore = article['bias_score'] as double?;
                      final sentimentScore = article['sentiment_score'] as double?;
                      final sourceName = article['source'] ?? article['source_name'] ?? 'Unknown Source';
                      final url = article['url'] ?? '';
                      final content = article['content'] ?? 'No content available.';
                      String title = article['title'] ?? 'Article ${index + 1}';

                      // Clean up filenames that sometimes appear as titles from raw scraping
                      if (title == 'Article ${index + 1}' && url.isNotEmpty) {
                        final uri = Uri.tryParse(url);
                        if (uri != null && uri.pathSegments.isNotEmpty) {
                          String pathSegment = uri.pathSegments.last;
                          pathSegment = pathSegment.replaceAll('.html', '')
                                                 .replaceAll('.htm', '')
                                                 .replaceAll('-', ' ')
                                                 .replaceAll('_', ' ');
                          if (pathSegment.length > 60) {
                            title = '${pathSegment.substring(0, 57)}...';
                          } else {
                            title = pathSegment;
                          }
                        }
                      }

                      return Card(
                        elevation: 2,
                        margin: const EdgeInsets.only(bottom: 12),
                        child: ListTile(
                          contentPadding: const EdgeInsets.all(16),
                          title: Text(
                            title,
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 15,
                            ),
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                          ),
                          subtitle: Padding(
                            padding: const EdgeInsets.only(top: 8.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                // Source Icon and Name
                                Row(
                                  children: [
                                    Icon(Icons.source, size: 16, color: Colors.grey[600]),
                                    const SizedBox(width: 4),
                                    Expanded(
                                      child: Text(
                                        sourceName,
                                        overflow: TextOverflow.ellipsis,
                                        style: const TextStyle(fontSize: 13),
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 8),
                                
                                // Bias and Sentiment Status Chips
                                Wrap(
                                  spacing: 8,
                                  children: [
                                    Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                                      decoration: BoxDecoration(
                                        color: _getBiasColor(biasScore).withAlpha((255 * 0.2).round()),
                                        borderRadius: BorderRadius.circular(12),
                                        border: Border.all(color: _getBiasColor(biasScore)),
                                      ),
                                      child: Text(
                                        _getBiasLabel(biasScore),
                                        style: TextStyle(
                                          fontSize: 11, 
                                          fontWeight: FontWeight.w600,
                                          color: biasScore == null ? Colors.grey[700] : Colors.black,
                                        ),
                                      ),
                                    ),
                                    if (sentimentScore == null)
                                      Container(
                                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                                        decoration: BoxDecoration(
                                          color: Colors.grey[200],
                                          borderRadius: BorderRadius.circular(12),
                                          border: Border.all(color: Colors.grey),
                                        ),
                                        child: const Text(
                                          "Sentiment: --",
                                          style: TextStyle(fontSize: 11, color: Colors.grey),
                                        ),
                                      ),
                                  ],
                                )
                              ],
                            ),
                          ),
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
