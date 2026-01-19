// lib/screens/placeholders.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../services/api_service.dart';
import '../models/article.dart';
import 'article_detail_screen.dart';
import 'settings_screen.dart';


/// Comparison screen for viewing how different outlets cover the same topic.
class CompareScreen extends StatefulWidget {
  const CompareScreen({super.key});

  @override
  State<CompareScreen> createState() => _CompareScreenState();
}


class _CompareScreenState extends State<CompareScreen> {
  final TextEditingController _searchController = TextEditingController();
  final ApiService _apiService = ApiService();

  List<Article>? _results;
  bool _isLoading = false;
  String? _errorMessage;

  Future<void> _searchTopic() async {
    final topic = _searchController.text.trim();
    if (topic.isEmpty) {
      setState(() {
        _errorMessage = "Please enter a topic";
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _results = null;
    });

    try {
      final articles = await _apiService.compareArticles(topic);
      setState(() {
        _results = articles;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = "Failed to load articles: $e";
        _isLoading = false;
      });
    }
  }

  Color _getBiasColor(double? score) {
    if (score == null) return Colors.grey[300]!;
    if (score < -0.1) return Colors.blue[700]!;
    if (score > 0.1) return Colors.red[700]!;
    return Colors.purple[400]!;
  }

  String _getBiasLabel(double? score) {
    if (score == null) return "Unscored";
    if (score < -0.1) return "Left";
    if (score > 0.1) return "Right";
    return "Center";
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Compare Coverage"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            TextField(
              controller: _searchController,
              decoration: InputDecoration(
                labelText: "Enter a topic to compare",
                hintText: "e.g., climate, housing, election",
                prefixIcon: const Icon(Icons.search),
                border: const OutlineInputBorder(),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () {
                    _searchController.clear();
                    setState(() {
                      _results = null;
                      _errorMessage = null;
                    });
                  },
                ),
              ),
              onSubmitted: (_) => _searchTopic(),
            ),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: _isLoading ? null : _searchTopic,
              icon: const Icon(Icons.compare_arrows),
              label: const Text("Compare Coverage"),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 14),
              ),
            ),
            const SizedBox(height: 16),
            if (_isLoading)
              const Expanded(
                child: Center(child: CircularProgressIndicator()),
              )
            else if (_errorMessage != null)
              Expanded(
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.error_outline,
                          size: 64, color: Colors.red),
                      const SizedBox(height: 16),
                      Text(_errorMessage!, textAlign: TextAlign.center),
                    ],
                  ),
                ),
              )
            else if (_results != null && _results!.isEmpty)
              const Expanded(
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.search_off, size: 64, color: Colors.grey),
                      SizedBox(height: 16),
                      Text("No articles found for this topic."),
                      SizedBox(height: 8),
                      Text(
                        "Try a different keyword.",
                        style: TextStyle(color: Colors.grey),
                      ),
                    ],
                  ),
                ),
              )
            else if (_results != null)
              Expanded(
                child: ListView.builder(
                  itemCount: _results!.length,
                  itemBuilder: (context, index) {
                    final article = _results![index];
                    return Card(
                      margin: const EdgeInsets.only(bottom: 12),
                      child: ListTile(
                        title: Text(
                          article.title,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        subtitle: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const SizedBox(height: 4),
                            Text(article.source),
                            const SizedBox(height: 8),
                            Wrap(
                              spacing: 6,
                              children: [
                                if (article.biasScore != null)
                                  Chip(
                                    label: Text(
                                        _getBiasLabel(article.biasScore)),
                                    backgroundColor: _getBiasColor(
                                            article.biasScore)
                                        .withAlpha((255 * 0.2).round()),
                                    labelStyle: TextStyle(
                                      fontSize: 10,
                                      color: _getBiasColor(article.biasScore),
                                    ),
                                  ),
                              ],
                            ),
                          ],
                        ),
                        isThreeLine: true,
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => ArticleDetailScreen(
                                title: article.title,
                                sourceName: article.source,
                                content: article.content,
                                url: article.url,
                                biasScore: article.biasScore,
                                sentimentScore: article.sentimentScore,
                              ),
                            ),
                          );
                        },
                      ),
                    );
                  },
                ),
              )
            else
              const Expanded(
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.compare_arrows,
                          size: 64, color: Colors.grey),
                      SizedBox(height: 16),
                      Text("Enter a topic above to start comparing."),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }
}


/// Profile screen showing reading history and bias profile.
class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}


class _ProfileScreenState extends State<ProfileScreen> {
  final ApiService _apiService = ApiService();
  final user = FirebaseAuth.instance.currentUser;

  late Future<List<Article>> _readingHistoryFuture;
  late Future<Map<String, int>> _biasProfileFuture;

  @override
  void initState() {
    super.initState();
    final userId = user?.uid ?? 'demo-user';
    _readingHistoryFuture = _apiService.getUserReadingHistory(userId);
    _biasProfileFuture = _apiService.getBiasProfile(userId);
  }

  Color _getBiasColor(String category) {
    switch (category) {
      case 'Left':
        return Colors.blue[700]!;
      case 'Right':
        return Colors.red[700]!;
      case 'Center':
        return Colors.purple[400]!;
      default:
        return Colors.grey;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("My Profile"),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const SettingsScreen()),
              );
            },
            tooltip: "Settings",
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CircleAvatar(
                  radius: 32,
                  backgroundColor: Colors.blue[100],
                  child: Text(
                    (user?.displayName ?? 'U')[0].toUpperCase(),
                    style: const TextStyle(
                        fontSize: 28, fontWeight: FontWeight.bold),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        user?.displayName ?? 'Reader',
                        style:
                            Theme.of(context).textTheme.titleLarge?.copyWith(
                                  fontWeight: FontWeight.bold,
                                ),
                      ),
                      Text(
                        user?.email ?? '',
                        style:
                            Theme.of(context).textTheme.bodyMedium?.copyWith(
                                  color: Colors.grey[600],
                                ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),
            Text(
              "Your Bias Profile",
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 12),
            FutureBuilder<Map<String, int>>(
              future: _biasProfileFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                } else if (snapshot.hasError) {
                  return Text("Error: ${snapshot.error}");
                } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
                  return const Text("No reading history yet.");
                }

                final profile = snapshot.data!;
                final total =
                    profile.values.fold<int>(0, (sum, count) => sum + count);

                return Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        ...profile.entries.map((entry) {
                          final percentage =
                              (entry.value / total * 100).round();
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 12.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  mainAxisAlignment:
                                      MainAxisAlignment.spaceBetween,
                                  children: [
                                    Text(
                                      entry.key,
                                      style: TextStyle(
                                        fontWeight: FontWeight.bold,
                                        color: _getBiasColor(entry.key),
                                      ),
                                    ),
                                    Text(
                                      "${entry.value} articles ($percentage%)",
                                      style: TextStyle(
                                          color: Colors.grey[600]),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 4),
                                LinearProgressIndicator(
                                  value: entry.value / total,
                                  backgroundColor: Colors.grey[200],
                                  color: _getBiasColor(entry.key),
                                  minHeight: 8,
                                ),
                              ],
                            ),
                          );
                        }),
                        const SizedBox(height: 8),
                        Text(
                          "You've read $total articles total",
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey[600],
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
            const SizedBox(height: 24),
            Text(
              "Reading History",
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 12),
            FutureBuilder<List<Article>>(
              future: _readingHistoryFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const Center(child: CircularProgressIndicator());
                } else if (snapshot.hasError) {
                  return Text("Error: ${snapshot.error}");
                } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
                  return const Center(
                      child: Text("No articles read yet."));
                }

                final articles = snapshot.data!;
                return Column(
                  children: articles.map((article) {
                    final biasLabel = article.biasScore != null
                        ? (article.biasScore! < -0.1
                            ? "Left"
                            : article.biasScore! > 0.1
                                ? "Right"
                                : "Center")
                        : "Unscored";
                    final biasColor = article.biasScore != null
                        ? (article.biasScore! < -0.1
                            ? Colors.blue[700]!
                            : article.biasScore! > 0.1
                                ? Colors.red[700]!
                                : Colors.purple[400]!)
                        : Colors.grey;

                    return Card(
                      margin: const EdgeInsets.only(bottom: 8),
                      child: ListTile(
                        title: Text(
                          article.title,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        subtitle: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const SizedBox(height: 4),
                            Text(article.source),
                            const SizedBox(height: 4),
                            Chip(
                              label: Text(biasLabel),
                              backgroundColor:
                                  biasColor.withAlpha((255 * 0.2).round()),
                              labelStyle: TextStyle(
                                  fontSize: 10, color: biasColor),
                            ),
                          ],
                        ),
                        trailing: Text(
                          _formatDate(article.publishedAt),
                          style: TextStyle(
                              fontSize: 12, color: Colors.grey[600]),
                        ),
                        onTap: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => ArticleDetailScreen(
                                title: article.title,
                                sourceName: article.source,
                                content: article.content,
                                url: article.url,
                                biasScore: article.biasScore,
                                sentimentScore: article.sentimentScore,
                              ),
                            ),
                          );
                        },
                      ),
                    );
                  }).toList(),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  String _formatDate(DateTime? date) {
    if (date == null) return '';
    final now = DateTime.now();
    final diff = now.difference(date);
    if (diff.inDays == 0) return 'Today';
    if (diff.inDays == 1) return 'Yesterday';
    return '${diff.inDays}d ago';
  }
}
