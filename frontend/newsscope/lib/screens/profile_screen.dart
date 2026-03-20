import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:fl_chart/fl_chart.dart';

import '../models/bias_profile.dart';
import '../services/api_service.dart';
import '../utils/score_helpers.dart';
import 'settings_screen.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final ApiService _apiService = ApiService();
  final user = FirebaseAuth.instance.currentUser;

  BiasProfile? _profile;
  bool _loading = true;
  String? _errorMessage;
  int _touchedBarIndex = -1;

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
      if (!mounted) return;
      setState(() {
        _profile = profile;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Failed to load profile: $e';
        _loading = false;
      });
    }
  }

  Widget _buildBiasProfileTitle() {
    return Text(
      'Bias Profile',
      style: TextStyle(
        fontSize: 20,
        fontWeight: FontWeight.bold,
        letterSpacing: 0.3,
        color: Colors.blue[800],
      ),
    );
  }

  Widget _settingsButton() {
    return IconButton(
      icon: const Icon(Icons.settings),
      tooltip: 'Settings',
      onPressed: () => Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => const SettingsScreen()),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    if (_errorMessage != null) {
      return Scaffold(
        appBar: AppBar(centerTitle: true, title: _buildBiasProfileTitle()),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, size: 64, color: Colors.red),
              const SizedBox(height: 16),
              Text(_errorMessage!),
              const SizedBox(height: 16),
              ElevatedButton(onPressed: _loadProfile, child: const Text('Retry')),
            ],
          ),
        ),
      );
    }

    if (_profile == null || _profile!.isEmpty) {
      return Scaffold(
        appBar: AppBar(
          centerTitle: true,
          title: _buildBiasProfileTitle(),
          actions: [_settingsButton()],
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.auto_stories, size: 80, color: Colors.grey.shade300),
              const SizedBox(height: 20),
              Text(
                'Your Bias Profile',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.grey.shade700,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Start reading articles to build\nyour personal bias breakdown.',
                style: TextStyle(fontSize: 14, color: Colors.grey.shade500, height: 1.5),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: _buildBiasProfileTitle(),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadProfile,
            tooltip: 'Refresh',
          ),
          _settingsButton(),
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
              _buildSectionTitle('Reading Distribution', Icons.pie_chart),
              const SizedBox(height: 12),
              _buildPieChart(),
              const SizedBox(height: 24),
              _buildSectionTitle('Most Read Sources', Icons.bar_chart),
              const SizedBox(height: 12),
              _buildSourceBarChart(),
              const SizedBox(height: 24),
              _buildSectionTitle('Detailed Breakdown', Icons.analytics_outlined),
              const SizedBox(height: 12),
              _buildDetailsCards(),
            ],
          ),
        ),
      ),
    );
  }

  // ── REDESIGNED User header ────────────────────────────────────────────────
  // Replaced blue gradient blob with a structured navy card with inline stats row

  Widget _buildUserHeader() {
    final name = user?.displayName ?? 'Reader';
    final initial = name[0].toUpperCase();
    final p = _profile!;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 18),
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
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  color: Colors.white.withAlpha(18),
                  border: Border.all(color: Colors.white.withAlpha(50), width: 1.5),
                ),
                child: Center(
                  child: Text(
                    initial,
                    style: const TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      name,
                      style: const TextStyle(
                        fontSize: 17,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    if (user?.email != null)
                      Text(
                        user!.email!,
                        style: TextStyle(
                          color: Colors.white.withAlpha(150),
                          fontSize: 12,
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 18),
          Container(height: 1, color: Colors.white.withAlpha(25)),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildHeaderStat('${p.totalArticlesRead}', 'Articles Read'),
              Container(width: 1, height: 32, color: Colors.white.withAlpha(25)),
              _buildHeaderStat('${p.readingTimeTotalMinutes}m', 'Read Time'),
              Container(width: 1, height: 32, color: Colors.white.withAlpha(25)),
              _buildHeaderStat(getBiasLabel(p.avgBias), 'Leaning'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildHeaderStat(String value, String label) {
    return Column(
      children: [
        Text(
          value,
          style: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 3),
        Text(
          label,
          style: TextStyle(fontSize: 11, color: Colors.white.withAlpha(150)),
        ),
      ],
    );
  }

  // ── Stat cards ────────────────────────────────────────────────────────────

  Widget _buildStatCards() {
    final p = _profile!;
    final sentimentColor = p.sentimentLabel == 'Positive'
        ? Colors.green[600]!
        : p.sentimentLabel == 'Negative'
            ? Colors.red[600]!
            : Colors.orange[600]!;

    return Row(
      children: [
        Expanded(
          child: _buildStatCard(
            'Your Leaning',
            getBiasLabel(p.avgBias),
            getBiasColor(p.avgBias),
            Icons.balance,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _buildStatCard(
            'Avg Sentiment',
            p.sentimentLabel,
            sentimentColor,
            Icons.sentiment_satisfied,
          ),
        ),
      ],
    );
  }

  Widget _buildStatCard(String title, String value, Color color, IconData icon) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, size: 16, color: color),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    title,
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey.shade600,
                      fontWeight: FontWeight.w500,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(
              value,
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Pie chart ─────────────────────────────────────────────────────────────
  // FIX: sectionsSpace changed from 3 → 0 to remove gaps between slices

  Widget _buildPieChart() {
    final dist = _profile!.biasDistribution;
    final left = (dist['left'] ?? 0.0).toDouble();
    final center = (dist['center'] ?? 0.0).toDouble();
    final right = (dist['right'] ?? 0.0).toDouble();
    final total = left + center + right;
    if (total == 0) return const SizedBox.shrink();

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 16),
        child: Column(
          children: [
            SizedBox(
              height: 210,
              child: PieChart(
                PieChartData(
                  sectionsSpace: 0, // FIX: was 3, caused visible gaps
                  centerSpaceRadius: 50,
                  startDegreeOffset: -90,
                  sections: [
                    if (left > 0)
                      PieChartSectionData(
                        value: left,
                        title: '${left.toInt()}%',
                        color: Colors.blue[700]!,
                        radius: 55,
                        titleStyle: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                    if (center > 0)
                      PieChartSectionData(
                        value: center,
                        title: '${center.toInt()}%',
                        color: Colors.purple[400]!,
                        radius: 55,
                        titleStyle: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                    if (right > 0)
                      PieChartSectionData(
                        value: right,
                        title: '${right.toInt()}%',
                        color: Colors.red[700]!,
                        radius: 55,
                        titleStyle: const TextStyle(
                          fontSize: 13,
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
                _buildLegendItem('Left Wing', Colors.blue[700]!),
                const SizedBox(width: 20),
                _buildLegendItem('Centre', Colors.purple[400]!),
                const SizedBox(width: 20),
                _buildLegendItem('Right Wing', Colors.red[700]!),
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
          width: 12,
          height: 12,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 6),
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w500,
            color: Colors.grey[700],
          ),
        ),
      ],
    );
  }

  // ── Source bar chart ──────────────────────────────────────────────────────

  Color _sourceBarColor(int index, int total) {
    final palette = [
      Colors.blue[700]!,
      Colors.blue[500]!,
      Colors.cyan[600]!,
      Colors.purple[400]!,
      Colors.purple[600]!,
      Colors.indigo[500]!,
      Colors.teal[500]!,
      Colors.orange[600]!,
      Colors.orange[400]!,
      Colors.red[400]!,
      Colors.red[600]!,
      Colors.red[800]!,
    ];
    return palette[index % palette.length];
  }

  Widget _buildSourceBarChart() {
    final raw = _profile!.sourceBreakdown;

    if (raw == null || raw.isEmpty) {
      return _buildDetailCard(
        'Most read source',
        _profile!.mostReadSource,
        Colors.indigo[600]!,
        Icons.bookmark,
      );
    }

    final sorted = raw.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
    final sources = sorted.take(12).toList();
    final maxVal = sources.first.value.toDouble();

    // FIX: interval calculated to avoid fractional ticks causing duplicate labels
    final interval = (maxVal / 4).ceilToDouble().clamp(1.0, double.infinity);

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 20, 12, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              height: 240,
              child: BarChart(
                BarChartData(
                  alignment: BarChartAlignment.spaceAround,
                  maxY: maxVal * 1.25,
                  barTouchData: BarTouchData(
                    touchCallback: (FlTouchEvent event, BarTouchResponse? response) {
                      setState(() {
                        if (response == null ||
                            response.spot == null ||
                            event is FlTapUpEvent == false) {
                          _touchedBarIndex = -1;
                          return;
                        }
                        _touchedBarIndex = response.spot!.touchedBarGroupIndex;
                      });
                    },
                    touchTooltipData: BarTouchTooltipData(
                      getTooltipColor: (_) => Colors.blueGrey.shade800,
                      getTooltipItem: (group, groupIndex, rod, rodIndex) {
                        final s = sources[groupIndex];
                        return BarTooltipItem(
                          '${s.key}\n',
                          const TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 12,
                          ),
                          children: [
                            TextSpan(
                              text: '${s.value} article${s.value == 1 ? '' : 's'}',
                              style: const TextStyle(
                                color: Colors.white70,
                                fontSize: 11,
                                fontWeight: FontWeight.normal,
                              ),
                            ),
                          ],
                        );
                      },
                    ),
                  ),
                  titlesData: FlTitlesData(
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 44,
                        getTitlesWidget: (value, meta) {
                          final i = value.toInt();
                          if (i < 0 || i >= sources.length) return const SizedBox.shrink();
                          final name = sources[i].key;
                          final short = name.split(' ').first;
                          final abbr = short.length > 7 ? '${short.substring(0, 6)}…' : short;
                          return Padding(
                            padding: const EdgeInsets.only(top: 6),
                            child: Transform.rotate(
                              angle: -0.45,
                              child: Text(
                                abbr,
                                style: TextStyle(
                                  fontSize: 10,
                                  fontWeight: i == _touchedBarIndex
                                      ? FontWeight.bold
                                      : FontWeight.normal,
                                  color: i == _touchedBarIndex
                                      ? Colors.blue[700]
                                      : Colors.grey[600],
                                ),
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 28,
                        interval: interval,
                        getTitlesWidget: (value, meta) {
                          // FIX: skip meta.max (= maxY = maxVal*1.25) which rendered
                          // as the same int as the last real tick, causing the duplicate
                          if (value == meta.max) return const SizedBox.shrink();
                          return Text(
                            value.toInt().toString(),
                            style: TextStyle(fontSize: 10, color: Colors.grey[500]),
                          );
                        },
                      ),
                    ),
                    topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  ),
                  gridData: FlGridData(
                    show: true,
                    drawVerticalLine: false,
                    horizontalInterval: interval,
                    getDrawingHorizontalLine: (_) => FlLine(
                      color: Colors.grey.withValues(alpha: 0.15),
                      strokeWidth: 1,
                    ),
                  ),
                  borderData: FlBorderData(show: false),
                  barGroups: sources.asMap().entries.map((entry) {
                    final i = entry.key;
                    final isTouched = i == _touchedBarIndex;
                    return BarChartGroupData(
                      x: i,
                      barRods: [
                        BarChartRodData(
                          toY: entry.value.value.toDouble(),
                          color: isTouched
                              ? _sourceBarColor(i, sources.length).withValues(alpha: 1.0)
                              : _sourceBarColor(i, sources.length).withValues(alpha: 0.82),
                          width: sources.length > 8 ? 14 : 18,
                          borderRadius: const BorderRadius.vertical(top: Radius.circular(6)),
                          backDrawRodData: BackgroundBarChartRodData(
                            show: true,
                            toY: maxVal * 1.25,
                            color: Colors.grey.withValues(alpha: 0.05),
                          ),
                        ),
                      ],
                    );
                  }).toList(),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Tap a bar for details · showing top ${sources.length} sources',
              style: TextStyle(fontSize: 10, color: Colors.grey[400]),
            ),
          ],
        ),
      ),
    );
  }

  // ── Detailed breakdown cards ──────────────────────────────────────────────

  Widget _buildDetailsCards() {
    final p = _profile!;
    return Column(
      children: [
        _buildDetailCard('Left Wing articles', '${p.leftCount}', Colors.blue[700]!, Icons.trending_down),
        _buildDetailCard('Centre articles', '${p.centerCount}', Colors.purple[400]!, Icons.trending_flat),
        _buildDetailCard('Right Wing articles', '${p.rightCount}', Colors.red[700]!, Icons.trending_up),
        _buildDetailCard('Positive articles', '${p.positiveCount}', Colors.green[600]!, Icons.sentiment_satisfied),
        _buildDetailCard('Negative articles', '${p.negativeCount}', Colors.red[600]!, Icons.sentiment_dissatisfied),
        if (_profile!.sourceBreakdown == null || _profile!.sourceBreakdown!.isEmpty)
          _buildDetailCard('Most read source', p.mostReadSource, Colors.indigo[600]!, Icons.bookmark),
      ],
    );
  }

  Widget _buildDetailCard(String label, String value, Color color, IconData icon) {
    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: color.withAlpha(40),
          child: Icon(icon, color: color, size: 20),
        ),
        title: Text(label, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500)),
        trailing: Text(
          value,
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: color),
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title, IconData icon) {
    return Row(
      children: [
        Icon(icon, size: 20, color: Colors.blue[700]),
        const SizedBox(width: 8),
        Text(
          title,
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.grey[800]),
        ),
      ],
    );
  }
}
