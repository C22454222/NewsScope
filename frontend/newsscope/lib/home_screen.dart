import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../services/api_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final user = FirebaseAuth.instance.currentUser;
  final ApiService _apiService = ApiService();
  late Future<List<dynamic>> _articlesFuture;

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

  Color _getBiasColor(double? score) {
    if (score == null) return Colors.grey;
    // Assuming score range: -1.0 (Left) to 1.0 (Right)
    if (score < -0.3) return Colors.blue[300]!;
    if (score > 0.3) return Colors.red[300]!;
    return Colors.purple[200]!; // Center
  }

  String _getBiasLabel(double? score) {
    if (score == null) return "Pending";
    if (score < -0.3) return "Left";
    if (score > 0.3) return "Right";
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
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Hello, ${user?.displayName ?? user?.email?.split('@')[0] ?? 'Reader'}!",
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
          Expanded(
            child: FutureBuilder<List<dynamic>>(
              future: _articlesFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                } else if (snapshot.hasError) {
                  return Center(
                    child: Padding(
                      padding: const EdgeInsets.all(24.0),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(Icons.error_outline, size: 64, color: Colors.redAccent),
                          const SizedBox(height: 16),
                          Text(
                            "Error loading articles",
                            style: Theme.of(context).textTheme.titleLarge,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            snapshot.error.toString(),
                            textAlign: TextAlign.center,
                            style: TextStyle(color: Colors.grey[600]),
                          ),
                          const SizedBox(height: 24),
                          ElevatedButton.icon(
                            onPressed: _refreshArticles,
                            icon: const Icon(Icons.refresh),
                            label: const Text("Retry"),
                          ),
                        ],
                      ),
                    ),
                  );
                } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
                  return Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.article_outlined, size: 64, color: Colors.grey[400]),
                        const SizedBox(height: 16),
                        const Text("No articles found."),
                        const SizedBox(height: 8),
                        ElevatedButton(
                          onPressed: _refreshArticles,
                          child: const Text("Refresh"),
                        ),
                      ],
                    ),
                  );
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
                      final sourceName = article['source_name'] ?? 'Unknown Source';
                      final url = article['url'] ?? '';
                      
                      // Extract a simple title from URL if not provided
                      String title = 'Article ${index + 1}';
                      if (url.isNotEmpty) {
                        final uri = Uri.tryParse(url);
                        if (uri != null && uri.pathSegments.isNotEmpty) {
                          title = uri.pathSegments.last
                              .replaceAll('-', ' ')
                              .replaceAll('_', ' ');
                          if (title.length > 60) {
                            title = '${title.substring(0, 57)}...';
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
                            child: Row(
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
                                const SizedBox(width: 8),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 10, vertical: 4),
                                  decoration: BoxDecoration(
                                    color: _getBiasColor(biasScore).withOpacity(0.2),
                                    borderRadius: BorderRadius.circular(12),
                                    border: Border.all(
                                      color: _getBiasColor(biasScore),
                                      width: 1.5,
                                    ),
                                  ),
                                  child: Text(
                                    _getBiasLabel(biasScore),
                                    style: const TextStyle(
                                      color: Colors.black87,
                                      fontSize: 11,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                          onTap: () {
                            // TODO: Open article detail or web view
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(content: Text('Article from $sourceName')),
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
      bottomNavigationBar: BottomNavigationBar(
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: "Home"),
          BottomNavigationBarItem(icon: Icon(Icons.search), label: "Compare"),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: "Profile"),
        ],
        onTap: (index) {
          // TODO: Handle navigation to Compare/Profile screens
        },
      ),
    );
  }
}
