// lib/screens/profile_screen.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:fl_chart/fl_chart.dart';
import '../services/api_service.dart';
import 'settings_screen.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final ApiService _apiService = ApiService();
  final user = FirebaseAuth.instance.currentUser;

  Map<String, dynamic>? _profile;
  bool _loading = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _loadProfile();
  }

  Future<void> _loadProfile() async {
    setState(() {
      _loading = true;
      _errorMessage = null;
    });

    try {
      final profile = await _apiService.getBiasProfile();
      setState(() {
        _profile = profile;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to load profile: $e';
        _loading = false;
      });
    }
  }

  Color _getBiasColor(double bias) {
    if (bias < -0.3) return Colors.blue[700]!;
    if (bias > 0.3) return Colors.red[700]!;
    return Colors.purple[400]!;
  }

  String _getBiasLabel(double bias) {
    if (bias < -0.5) return 'Left';
    if (bias < -0.2) return 'Center-Left';
    if (bias < 0.2) return 'Center';
    if (bias < 0.5) return 'Center-Right';
    return 'Right';
  }

  Color _getSentimentColor(double sentiment) {
    if (sentiment > 0.3) return Colors.green[600]!;
    if (sentiment < -0.3) return Colors.orange[600]!;
    return Colors.grey[600]!;
  }

  String _getSentimentLabel(double sentiment) {
    if (sentiment > 0.3) return 'Positive';
    if (sentiment < -0.3) return 'Negative';
    return 'Neutral';
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    if (_errorMessage != null) {
      return Scaffold(
        appBar: AppBar(title: const Text('My Profile')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, size: 64, color: Colors.red),
              const SizedBox(height: 16),
              Text(_errorMessage!),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _loadProfile,
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    if (_profile == null || _profile!['total_articles_read'] == 0) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('My Profile'),
          actions: [
            IconButton(
              icon: const Icon(Icons.settings),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (_) => const SettingsScreen(),
                  ),
                );
              },
            ),
          ],
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.auto_stories,
                size: 80,
                color: Colors.grey.shade400,
              ),
              const SizedBox(height: 16),
              Text(
                'Start reading articles to see your profile!',
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.grey.shade600,
                ),
              ),
              const SizedBox(height: 24),
              ElevatedButton.icon(
                onPressed: () {
                  // Navigate to home/articles tab
                },
                icon: const Icon(Icons.article),
                label: const Text('Browse Articles'),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('My Profile'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadProfile,
            tooltip: 'Refresh',
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => const SettingsScreen(),
                ),
              );
            },
            tooltip: 'Settings',
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _loadProfile,
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          physics: const AlwaysScrollableScrollPhysics(),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildUserHeader(),
              const SizedBox(height: 24),
              _buildStatCards(),
              const SizedBox(height: 24),
              _buildSectionTitle('Reading Distribution'),
              const SizedBox(height: 16),
              _buildPieChart(),
              const SizedBox(height: 24),
              _buildSectionTitle('Detailed Breakdown'),
              const SizedBox(height: 16),
              _buildDetailsCards(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildUserHeader() {
    return Row(
      children: [
        CircleAvatar(
          radius: 32,
          backgroundColor: Colors.blue[100],
          child: Text(
            (user?.displayName ?? user?.email ?? 'U')[0].toUpperCase(),
            style: const TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                user?.displayName ?? 'Reader',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              if (user?.email != null)
                Text(
                  user!.email!,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: Colors.grey[600],
                      ),
                ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildStatCards() {
    final bias = (_profile!['avg_bias'] ?? 0.0) as double;
    final sentiment = (_profile!['avg_sentiment'] ?? 0.0) as double;
    final total = (_profile!['total_articles_read'] ?? 0) as int;
    final minutes = (_profile!['reading_time_total_minutes'] ?? 0) as int;

    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: _buildStatCard(
                'Your Leaning',
                _getBiasLabel(bias),
                _getBiasColor(bias),
                Icons.balance,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildStatCard(
                'Avg Sentiment',
                _getSentimentLabel(sentiment),
                _getSentimentColor(sentiment),
                Icons.sentiment_satisfied,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _buildStatCard(
                'Articles Read',
                '$total',
                Colors.blue,
                Icons.article,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _buildStatCard(
                'Reading Time',
                '${minutes}min',
                Colors.purple,
                Icons.timer,
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildStatCard(
    String title,
    String value,
    Color color,
    IconData icon,
  ) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, size: 16, color: color),
                const SizedBox(width: 4),
                Expanded(
                  child: Text(
                    title,
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey.shade600,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              value,
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPieChart() {
    final dist = _profile!['bias_distribution'] as Map<String, dynamic>;
    final left = ((dist['left'] ?? 0) as num).toDouble();
    final center = ((dist['center'] ?? 0) as num).toDouble();
    final right = ((dist['right'] ?? 0) as num).toDouble();

    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            SizedBox(
              height: 250,
              child: PieChart(
                PieChartData(
                  sectionsSpace: 2,
                  centerSpaceRadius: 60,
                  sections: [
                    if (left > 0)
                      PieChartSectionData(
                        value: left,
                        title: '${left.toInt()}%',
                        color: Colors.blue[700],
                        radius: 80,
                        titleStyle: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                    if (center > 0)
                      PieChartSectionData(
                        value: center,
                        title: '${center.toInt()}%',
                        color: Colors.purple[400],
                        radius: 80,
                        titleStyle: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                    if (right > 0)
                      PieChartSectionData(
                        value: right,
                        title: '${right.toInt()}%',
                        color: Colors.red[700],
                        radius: 80,
                        titleStyle: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildLegendItem('Left', Colors.blue[700]!),
                const SizedBox(width: 16),
                _buildLegendItem('Center', Colors.purple[400]!),
                const SizedBox(width: 16),
                _buildLegendItem('Right', Colors.red[700]!),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLegendItem(String label, Color color) {
    return Row(
      children: [
        Container(
          width: 16,
          height: 16,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(width: 4),
        Text(
          label,
          style: const TextStyle(fontSize: 12),
        ),
      ],
    );
  }

  Widget _buildDetailsCards() {
    final leftCount = (_profile!['left_count'] ?? 0) as int;
    final centerCount = (_profile!['center_count'] ?? 0) as int;
    final rightCount = (_profile!['right_count'] ?? 0) as int;
    final mostRead = (_profile!['most_read_source'] ?? 'N/A') as String;

    return Column(
      children: [
        _buildDetailCard(
          'Left-leaning articles',
          '$leftCount',
          Colors.blue[700]!,
          Icons.trending_down,
        ),
        _buildDetailCard(
          'Center articles',
          '$centerCount',
          Colors.purple[400]!,
          Icons.trending_flat,
        ),
        _buildDetailCard(
          'Right-leaning articles',
          '$rightCount',
          Colors.red[700]!,
          Icons.trending_up,
        ),
        _buildDetailCard(
          'Most read source',
          mostRead,
          Colors.indigo[600]!,
          Icons.bookmark,
        ),
      ],
    );
  }

  Widget _buildDetailCard(
    String label,
    String value,
    Color color,
    IconData icon,
  ) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 1,
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: color.withAlpha((255 * 0.2).round()),
          child: Icon(icon, color: color, size: 20),
        ),
        title: Text(label),
        trailing: Text(
          value,
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontSize: 20,
        fontWeight: FontWeight.bold,
      ),
    );
  }
}
