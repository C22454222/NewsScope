import 'dart:async';

import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:fl_chart/fl_chart.dart';

import '../models/bias_profile.dart';
import '../services/api_service.dart';
import '../utils/score_helpers.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final ApiService _apiService = ApiService();

  User? _user;
  StreamSubscription<User?>? _userSub;

  BiasProfile? _profile;
  bool _loading = true;
  String? _errorMessage;
  int _touchedBarIndex = -1;

  @override
  void initState() {
    super.initState();
    _user = FirebaseAuth.instance.currentUser;
    _userSub = FirebaseAuth.instance.userChanges().listen((u) {
      if (mounted) setState(() => _user = u);
    });
    _loadProfile();
  }

  @override
  void dispose() {
    _userSub?.cancel();
    super.dispose();
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

  Widget _buildBiasProfileTitle() => Text(
        'Bias Profile',
        style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            letterSpacing: 0.3,
            color: Colors.blue[800]),
      );

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(
          body: Center(child: CircularProgressIndicator()));
    }

    if (_errorMessage != null) {
      return Scaffold(
        appBar: AppBar(
            centerTitle: true, title: _buildBiasProfileTitle()),
        body: Center(
            child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
              const Icon(Icons.error_outline,
                  size: 64, color: Colors.red),
              const SizedBox(height: 16),
              Text(_errorMessage!),
              const SizedBox(height: 16),
              ElevatedButton(
                  onPressed: _loadProfile,
                  child: const Text('Retry')),
            ])),
      );
    }

    if (_profile == null || _profile!.isEmpty) {
      return Scaffold(
        appBar: AppBar(
            centerTitle: true, title: _buildBiasProfileTitle()),
        body: Center(
            child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
              Icon(Icons.auto_stories,
                  size: 80, color: Colors.grey.shade300),
              const SizedBox(height: 20),
              Text('Your Bias Profile',
                  style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.grey.shade700)),
              const SizedBox(height: 8),
              Text(
                  'Start reading articles to build\n'
                  'your personal bias breakdown.',
                  style: TextStyle(
                      fontSize: 14,
                      color: Colors.grey.shade500,
                      height: 1.5),
                  textAlign: TextAlign.center),
            ])),
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
              tooltip: 'Refresh'),
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
              const SizedBox(height: 20),
              _buildStatCards(),
              const SizedBox(height: 20),
              _buildSectionTitle(
                  'Reading Distribution by Outlet Political Bias',
                  Icons.pie_chart),
              const SizedBox(height: 12),
              _buildOutletPieChart(),
              const SizedBox(height: 20),
              _buildSectionTitle(
                  'Reading Distribution by Article Political Bias',
                  Icons.pie_chart_outline),
              const SizedBox(height: 12),
              _buildArticlePieChart(),
              const SizedBox(height: 20),
              _buildSectionTitle('Most Read Sources', Icons.bar_chart),
              const SizedBox(height: 12),
              _buildSourceBarChart(),
              const SizedBox(height: 20),
              _buildSectionTitle(
                  'Detailed Breakdown', Icons.analytics_outlined),
              const SizedBox(height: 12),
              _buildDetailedBreakdown(),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  // ── User header ────────────────────────────────────────────────────────────

  Widget _buildUserHeader() {
    final name = _user?.displayName ?? 'Reader';
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
              offset: const Offset(0, 4))
        ],
      ),
      child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  color: Colors.white.withAlpha(18),
                  border: Border.all(
                      color: Colors.white.withAlpha(50), width: 1.5),
                ),
                child: Center(
                    child: Text(initial,
                        style: const TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                            color: Colors.white))),
              ),
              const SizedBox(width: 14),
              Expanded(
                  child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                    Text(name,
                        style: const TextStyle(
                            fontSize: 17,
                            fontWeight: FontWeight.bold,
                            color: Colors.white)),
                    if (_user?.email != null)
                      Text(_user!.email!,
                          style: TextStyle(
                              color: Colors.white.withAlpha(150),
                              fontSize: 12)),
                  ])),
            ]),
            const SizedBox(height: 18),
            Container(height: 1, color: Colors.white.withAlpha(25)),
            const SizedBox(height: 16),
            Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _buildHeaderStat(
                      '${p.totalArticlesRead}', 'Articles Read'),
                  Container(
                      width: 1,
                      height: 32,
                      color: Colors.white.withAlpha(25)),
                  _buildHeaderStat(
                      '${p.readingTimeTotalMinutes}m', 'Read Time'),
                  Container(
                      width: 1,
                      height: 32,
                      color: Colors.white.withAlpha(25)),
                  _buildHeaderStat(
                    p.avgCredibility != null
                        ? '${p.avgCredibility!.round()}%'
                        : '—',
                    'Avg Credibility',
                  ),
                ]),
          ]),
    );
  }

  Widget _buildHeaderStat(String value, String label) {
    return Column(children: [
      Text(value,
          style: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.bold,
              color: Colors.white)),
      const SizedBox(height: 3),
      Text(label,
          style: TextStyle(
              fontSize: 11, color: Colors.white.withAlpha(150))),
    ]);
  }

  // ── Four stat cards (2 × 2) ────────────────────────────────────────────────

  Widget _buildStatCards() {
    final p = _profile!;

    // Outlet Political Bias
    final outletBiasColor = getBiasColor(p.avgBias);
    final outletBiasLabel = getBiasLabel(p.avgBias);

    // Article Political Bias
    final articleBiasColor = getBiasColor(p.avgArticleBias);
    final articleBiasLabel = getBiasLabel(p.avgArticleBias);
    final hasArticleBias =
        p.articleLeftCount + p.articleCenterCount + p.articleRightCount >
            0;

    // General Bias
    final totalLeaning = p.leftCount + p.centerCount + p.rightCount;
    final extremeRatio = totalLeaning > 0
        ? (p.leftCount + p.rightCount) / totalLeaning
        : 0.0;
    final generalBiasLabel =
        extremeRatio > 0.6 ? 'Biased' : 'Unbiased';
    final generalBiasColor = extremeRatio > 0.6
        ? Colors.orange[700]!
        : Colors.green[700]!;
    final generalBiasIcon = extremeRatio > 0.6
        ? Icons.warning_amber
        : Icons.check_circle_outline;

    // Sentiment
    final sentimentColor = p.sentimentLabel == 'Positive'
        ? Colors.green[700]!
        : p.sentimentLabel == 'Negative'
            ? Colors.deepOrange[600]!
            : Colors.amber[600]!;
    final sentimentIcon = p.sentimentLabel == 'Positive'
        ? Icons.sentiment_satisfied
        : p.sentimentLabel == 'Negative'
            ? Icons.sentiment_dissatisfied
            : Icons.sentiment_neutral;

    return Column(children: [
      Row(children: [
        Expanded(
            child: _buildStatCard(
                'Outlet Political Bias',
                outletBiasLabel,
                outletBiasColor,
                Icons.source)),
        const SizedBox(width: 8),
        Expanded(
            child: _buildStatCard(
                'Article Political Bias',
                hasArticleBias ? articleBiasLabel : '—',
                hasArticleBias ? articleBiasColor : Colors.grey,
                Icons.article_outlined)),
      ]),
      const SizedBox(height: 8),
      Row(children: [
        Expanded(
            child: _buildStatCard('General Bias', generalBiasLabel,
                generalBiasColor, generalBiasIcon)),
        const SizedBox(width: 8),
        Expanded(
            child: _buildStatCard('Sentiment', p.sentimentLabel,
                sentimentColor, sentimentIcon)),
      ]),
    ]);
  }

  Widget _buildStatCard(
      String title, String value, Color color, IconData icon) {
    return Card(
      elevation: 2,
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(children: [
                Icon(icon, size: 14, color: color),
                const SizedBox(width: 5),
                Expanded(
                  child: Text(title,
                      style: TextStyle(
                          fontSize: 10,
                          color: Colors.grey.shade600,
                          fontWeight: FontWeight.w500),
                      overflow: TextOverflow.ellipsis),
                ),
              ]),
              const SizedBox(height: 8),
              Text(value,
                  style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.bold,
                      color: color)),
            ]),
      ),
    );
  }

  // ── Pie charts ─────────────────────────────────────────────────────────────

  List<int> _roundToHundred(double a, double b, double c) {
    final floors = [a.floor(), b.floor(), c.floor()];
    final remainders = [
      a - floors[0],
      b - floors[1],
      c - floors[2]
    ];
    int remainder = 100 - floors.reduce((x, y) => x + y);
    final indices = [0, 1, 2]
      ..sort((i, j) => remainders[j].compareTo(remainders[i]));
    for (int i = 0; i < remainder; i++) {
      floors[indices[i]]++;
    }
    return floors;
  }

  Widget _buildPieChartFromDist(Map<String, double> dist) {
    final leftRaw = (dist['left'] ?? 0.0).toDouble();
    final centerRaw = (dist['center'] ?? 0.0).toDouble();
    final rightRaw = (dist['right'] ?? 0.0).toDouble();
    final total = leftRaw + centerRaw + rightRaw;

    if (total == 0) {
      return Card(
        elevation: 2,
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 32),
          child: Center(
              child: Text('No data yet',
                  style: TextStyle(
                      color: Colors.grey.shade400, fontSize: 13))),
        ),
      );
    }

    final rounded = _roundToHundred(leftRaw, centerRaw, rightRaw);
    final left = rounded[0];
    final center = rounded[1];
    final right = rounded[2];

    return Card(
      elevation: 2,
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 20, 20, 16),
        child: Column(children: [
          SizedBox(
            height: 210,
            child: PieChart(PieChartData(
              sectionsSpace: 0,
              centerSpaceRadius: 50,
              startDegreeOffset: -90,
              sections: [
                if (left > 0)
                  PieChartSectionData(
                      value: leftRaw,
                      title: '$left%',
                      color: Colors.blue[800]!,
                      radius: 55,
                      titleStyle: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.bold,
                          color: Colors.white)),
                if (center > 0)
                  PieChartSectionData(
                      value: centerRaw,
                      title: '$center%',
                      color: Colors.teal[600]!,
                      radius: 55,
                      titleStyle: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.bold,
                          color: Colors.white)),
                if (right > 0)
                  PieChartSectionData(
                      value: rightRaw,
                      title: '$right%',
                      color: Colors.red[800]!,
                      radius: 55,
                      titleStyle: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.bold,
                          color: Colors.white)),
              ],
            )),
          ),
          const SizedBox(height: 16),
          Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            _buildLegendItem('Left Wing', Colors.blue[800]!),
            const SizedBox(width: 20),
            _buildLegendItem('Centre', Colors.teal[600]!),
            const SizedBox(width: 20),
            _buildLegendItem('Right Wing', Colors.red[800]!),
          ]),
        ]),
      ),
    );
  }

  Widget _buildOutletPieChart() =>
      _buildPieChartFromDist(_profile!.biasDistribution);

  Widget _buildArticlePieChart() =>
      _buildPieChartFromDist(_profile!.articleBiasDistribution);

  Widget _buildLegendItem(String label, Color color) {
    return Row(children: [
      Container(
          width: 12,
          height: 12,
          decoration:
              BoxDecoration(color: color, shape: BoxShape.circle)),
      const SizedBox(width: 6),
      Text(label,
          style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w500,
              color: Colors.grey[700])),
    ]);
  }

  // ── Source bar chart ───────────────────────────────────────────────────────

  Color _sourceBarColor(int index) {
    final palette = [
      Colors.blue[800]!,
      Colors.blue[600]!,
      Colors.cyan[600]!,
      Colors.teal[500]!,
      Colors.teal[700]!,
      Colors.indigo[500]!,
      Colors.purple[400]!,
      Colors.orange[600]!,
      Colors.orange[400]!,
      Colors.red[400]!,
      Colors.red[700]!,
      Colors.red[900]!,
    ];
    return palette[index % palette.length];
  }

  String _wrapSourceName(String name) {
    final words = name.split(' ');
    if (words.length <= 2) return words.join('\n');
    final mid = (words.length / 2).ceil();
    return '${words.sublist(0, mid).join(' ')}\n'
        '${words.sublist(mid).join(' ')}';
  }

  Widget _buildSourceBarChart() {
    final raw = _profile!.sourceBreakdown;
    if (raw == null || raw.isEmpty) {
      return _buildDetailCard(
          'Most read source',
          _profile!.mostReadSource,
          Colors.indigo[600]!,
          Icons.bookmark);
    }

    final sorted = raw.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final sources = sorted.take(12).toList();
    final maxVal = sources.first.value.toDouble();
    final interval =
        (maxVal / 4).ceilToDouble().clamp(1.0, double.infinity);
    const double barAreaWidth = 64.0;
    final double chartWidth = sources.length * barAreaWidth;

    return Card(
      elevation: 2,
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 20, 12, 12),
        child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(
                height: 270,
                child: Row(children: [
                  SizedBox(
                    width: 28,
                    child: BarChart(BarChartData(
                      maxY: maxVal * 1.25,
                      titlesData: FlTitlesData(
                        leftTitles: AxisTitles(
                            sideTitles: SideTitles(
                          showTitles: true,
                          reservedSize: 28,
                          interval: interval,
                          getTitlesWidget: (value, meta) {
                            if (value == meta.max) {
                              return const SizedBox.shrink();
                            }
                            return Text(value.toInt().toString(),
                                style: TextStyle(
                                    fontSize: 10,
                                    color: Colors.grey[500]));
                          },
                        )),
                        bottomTitles: const AxisTitles(
                            sideTitles:
                                SideTitles(showTitles: false)),
                        topTitles: const AxisTitles(
                            sideTitles:
                                SideTitles(showTitles: false)),
                        rightTitles: const AxisTitles(
                            sideTitles:
                                SideTitles(showTitles: false)),
                      ),
                      borderData: FlBorderData(show: false),
                      gridData: const FlGridData(show: false),
                      barGroups: const [],
                    )),
                  ),
                  Expanded(
                    child: SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: SizedBox(
                        width: chartWidth,
                        child: BarChart(BarChartData(
                          alignment: BarChartAlignment.spaceAround,
                          maxY: maxVal * 1.25,
                          barTouchData: BarTouchData(
                            touchCallback: (FlTouchEvent event,
                                BarTouchResponse? response) {
                              setState(() {
                                if (response == null ||
                                    response.spot == null ||
                                    event is FlTapUpEvent == false) {
                                  _touchedBarIndex = -1;
                                  return;
                                }
                                _touchedBarIndex = response
                                    .spot!.touchedBarGroupIndex;
                              });
                            },
                            touchTooltipData: BarTouchTooltipData(
                              getTooltipColor: (_) =>
                                  Colors.blueGrey.shade800,
                              fitInsideHorizontally: true,
                              fitInsideVertically: true,
                              getTooltipItem: (group, groupIndex,
                                  rod, rodIndex) {
                                final s = sources[groupIndex];
                                return BarTooltipItem(
                                  '${s.key}\n',
                                  const TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.bold,
                                      fontSize: 12),
                                  children: [
                                    TextSpan(
                                      text:
                                          '${s.value} article'
                                          '${s.value == 1 ? '' : 's'}',
                                      style: const TextStyle(
                                          color: Colors.white70,
                                          fontSize: 11,
                                          fontWeight:
                                              FontWeight.normal),
                                    )
                                  ],
                                );
                              },
                            ),
                          ),
                          titlesData: FlTitlesData(
                            bottomTitles: AxisTitles(
                                sideTitles: SideTitles(
                              showTitles: true,
                              reservedSize: 72,
                              getTitlesWidget: (value, meta) {
                                final i = value.toInt();
                                if (i < 0 || i >= sources.length) {
                                  return const SizedBox.shrink();
                                }
                                final isSelected =
                                    i == _touchedBarIndex;
                                return Padding(
                                  padding:
                                      const EdgeInsets.only(top: 6),
                                  child: Column(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Container(
                                          width: 18,
                                          height: 18,
                                          decoration: BoxDecoration(
                                            color: isSelected
                                                ? Colors.blue[700]
                                                : Colors.grey[300],
                                            shape: BoxShape.circle,
                                          ),
                                          child: Center(
                                              child: Text('${i + 1}',
                                                  style: TextStyle(
                                                      fontSize: 9,
                                                      fontWeight:
                                                          FontWeight
                                                              .bold,
                                                      color: isSelected
                                                          ? Colors
                                                              .white
                                                          : Colors
                                                              .grey[700]))),
                                        ),
                                        const SizedBox(height: 3),
                                        Text(
                                          _wrapSourceName(
                                              sources[i].key),
                                          textAlign: TextAlign.center,
                                          style: TextStyle(
                                            fontSize: 9,
                                            fontWeight: isSelected
                                                ? FontWeight.bold
                                                : FontWeight.normal,
                                            color: isSelected
                                                ? Colors.blue[700]
                                                : Colors.grey[600],
                                            height: 1.3,
                                          ),
                                        ),
                                      ]),
                                );
                              },
                            )),
                            leftTitles: const AxisTitles(
                                sideTitles:
                                    SideTitles(showTitles: false)),
                            topTitles: const AxisTitles(
                                sideTitles:
                                    SideTitles(showTitles: false)),
                            rightTitles: const AxisTitles(
                                sideTitles:
                                    SideTitles(showTitles: false)),
                          ),
                          gridData: FlGridData(
                            show: true,
                            drawVerticalLine: false,
                            horizontalInterval: interval,
                            getDrawingHorizontalLine: (_) => FlLine(
                                color: Colors.grey
                                    .withValues(alpha: 0.15),
                                strokeWidth: 1),
                          ),
                          borderData: FlBorderData(show: false),
                          barGroups:
                              sources.asMap().entries.map((entry) {
                            final i = entry.key;
                            final isTouched = i == _touchedBarIndex;
                            return BarChartGroupData(x: i, barRods: [
                              BarChartRodData(
                                toY: entry.value.value.toDouble(),
                                color: isTouched
                                    ? _sourceBarColor(i)
                                        .withValues(alpha: 1.0)
                                    : _sourceBarColor(i)
                                        .withValues(alpha: 0.82),
                                width: 22,
                                borderRadius:
                                    const BorderRadius.vertical(
                                        top: Radius.circular(6)),
                                backDrawRodData:
                                    BackgroundBarChartRodData(
                                  show: true,
                                  toY: maxVal * 1.25,
                                  color: Colors.grey
                                      .withValues(alpha: 0.05),
                                ),
                              ),
                            ]);
                          }).toList(),
                        )),
                      ),
                    ),
                  ),
                ]),
              ),
              const SizedBox(height: 4),
              Text(
                'Scroll to see all · tap a bar for details · '
                'top ${sources.length} sources',
                style:
                    TextStyle(fontSize: 10, color: Colors.grey[400]),
              ),
            ]),
      ),
    );
  }

  // ── Detailed breakdown ─────────────────────────────────────────────────────

  Widget _buildDetailedBreakdown() {
    final p = _profile!;
    final outletTotal = p.leftCount + p.centerCount + p.rightCount;
    final articleTotal =
        p.articleLeftCount + p.articleCenterCount + p.articleRightCount;
    final sentimentTotal = p.positiveCount + p.negativeCount;

    return Column(children: [
      // ── Outlet Political Bias ──────────────────────────────────────
      Card(
        elevation: 2,
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Icon(Icons.source, size: 16, color: Colors.blue[700]),
                  const SizedBox(width: 8),
                  Text('Outlet Political Bias',
                      style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey[800])),
                ]),
                const SizedBox(height: 14),
                Row(children: [
                  Expanded(
                      child: _buildLeaningTile(
                          label: 'Left Wing',
                          count: p.leftCount,
                          total: outletTotal,
                          color: Colors.blue[800]!,
                          icon: Icons.west)),
                  const SizedBox(width: 10),
                  Expanded(
                      child: _buildLeaningTile(
                          label: 'Centre',
                          count: p.centerCount,
                          total: outletTotal,
                          color: Colors.teal[600]!,
                          icon: Icons.remove)),
                  const SizedBox(width: 10),
                  Expanded(
                      child: _buildLeaningTile(
                          label: 'Right Wing',
                          count: p.rightCount,
                          total: outletTotal,
                          color: Colors.red[800]!,
                          icon: Icons.east)),
                ]),
                if (outletTotal > 0) ...[
                  const SizedBox(height: 14),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: Row(children: [
                      if (p.leftCount > 0)
                        Expanded(
                            flex: p.leftCount,
                            child: Container(
                                height: 8,
                                color: Colors.blue[800])),
                      if (p.centerCount > 0)
                        Expanded(
                            flex: p.centerCount,
                            child: Container(
                                height: 8,
                                color: Colors.teal[600])),
                      if (p.rightCount > 0)
                        Expanded(
                            flex: p.rightCount,
                            child: Container(
                                height: 8,
                                color: Colors.red[800])),
                    ]),
                  ),
                ],
              ]),
        ),
      ),
      const SizedBox(height: 12),

      // ── Article Political Bias ─────────────────────────────────────
      Card(
        elevation: 2,
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Icon(Icons.article_outlined,
                      size: 16, color: Colors.indigo[600]),
                  const SizedBox(width: 8),
                  Text('Article Political Bias',
                      style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey[800])),
                ]),
                const SizedBox(height: 14),
                if (articleTotal == 0)
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    child: Text(
                      'No per-article classifications yet.\n'
                      'Article bias appears once the RoBERTa model '
                      'scores new articles.',
                      style: TextStyle(
                          fontSize: 12, color: Colors.grey[500]),
                    ),
                  )
                else ...[
                  Row(children: [
                    Expanded(
                        child: _buildLeaningTile(
                            label: 'Left Wing',
                            count: p.articleLeftCount,
                            total: articleTotal,
                            color: Colors.blue[800]!,
                            icon: Icons.west)),
                    const SizedBox(width: 10),
                    Expanded(
                        child: _buildLeaningTile(
                            label: 'Centre',
                            count: p.articleCenterCount,
                            total: articleTotal,
                            color: Colors.teal[600]!,
                            icon: Icons.remove)),
                    const SizedBox(width: 10),
                    Expanded(
                        child: _buildLeaningTile(
                            label: 'Right Wing',
                            count: p.articleRightCount,
                            total: articleTotal,
                            color: Colors.red[800]!,
                            icon: Icons.east)),
                  ]),
                  const SizedBox(height: 14),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: Row(children: [
                      if (p.articleLeftCount > 0)
                        Expanded(
                            flex: p.articleLeftCount,
                            child: Container(
                                height: 8,
                                color: Colors.blue[800])),
                      if (p.articleCenterCount > 0)
                        Expanded(
                            flex: p.articleCenterCount,
                            child: Container(
                                height: 8,
                                color: Colors.teal[600])),
                      if (p.articleRightCount > 0)
                        Expanded(
                            flex: p.articleRightCount,
                            child: Container(
                                height: 8,
                                color: Colors.red[800])),
                    ]),
                  ),
                ],
              ]),
        ),
      ),
      const SizedBox(height: 12),

      // ── Sentiment ──────────────────────────────────────────────────
      Card(
        elevation: 2,
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Icon(Icons.sentiment_satisfied,
                      size: 16, color: Colors.green[700]),
                  const SizedBox(width: 8),
                  Text('Sentiment',
                      style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Colors.grey[800])),
                ]),
                const SizedBox(height: 14),
                Row(children: [
                  Expanded(
                      child: _buildSentimentTile(
                          label: 'Positive',
                          count: _profile!.positiveCount,
                          total: sentimentTotal,
                          color: Colors.green[700]!,
                          icon: Icons.sentiment_satisfied)),
                  const SizedBox(width: 10),
                  Expanded(
                      child: _buildSentimentTile(
                          label: 'Negative',
                          count: _profile!.negativeCount,
                          total: sentimentTotal,
                          color: Colors.deepOrange[600]!,
                          icon: Icons.sentiment_dissatisfied)),
                ]),
                if (sentimentTotal > 0) ...[
                  const SizedBox(height: 14),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: Row(children: [
                      if (_profile!.positiveCount > 0)
                        Expanded(
                            flex: _profile!.positiveCount,
                            child: Container(
                                height: 8,
                                color: Colors.green[700])),
                      if (_profile!.negativeCount > 0)
                        Expanded(
                            flex: _profile!.negativeCount,
                            child: Container(
                                height: 8,
                                color: Colors.deepOrange[600])),
                    ]),
                  ),
                ],
              ]),
        ),
      ),
      const SizedBox(height: 12),

      // ── Summary row ────────────────────────────────────────────────
      Card(
        elevation: 2,
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding:
              const EdgeInsets.symmetric(vertical: 14, horizontal: 12),
          child: Row(children: [
            Expanded(
                child: _buildSummaryItem(
                    label: 'Articles\nRead',
                    value: '${_profile!.totalArticlesRead}',
                    color: Colors.blue[700]!,
                    icon: Icons.article_outlined)),
            _buildVerticalDivider(),
            Expanded(
                child: _buildSummaryItem(
                    label: 'Read\nTime',
                    value: '${_profile!.readingTimeTotalMinutes}m',
                    color: Colors.indigo[600]!,
                    icon: Icons.timer_outlined)),
            _buildVerticalDivider(),
            Expanded(
                child: _buildSummaryItem(
                    label: 'Avg Outlet\nBias',
                    value: getBiasLabel(_profile!.avgBias),
                    color: getBiasColor(_profile!.avgBias),
                    icon: Icons.source)),
            _buildVerticalDivider(),
            Expanded(
                child: _buildSummaryItem(
                    label: 'Avg Article\nBias',
                    value: (articleTotal > 0)
                        ? getBiasLabel(_profile!.avgArticleBias)
                        : '—',
                    color: (articleTotal > 0)
                        ? getBiasColor(_profile!.avgArticleBias)
                        : Colors.grey,
                    icon: Icons.article_outlined)),
          ]),
        ),
      ),
    ]);
  }

  Widget _buildLeaningTile({
    required String label,
    required int count,
    required int total,
    required Color color,
    required IconData icon,
  }) {
    final pct = total > 0 ? (count / total * 100).round() : 0;
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
      decoration: BoxDecoration(
        color: color.withAlpha(18),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withAlpha(50)),
      ),
      child: Column(children: [
        Icon(icon, color: color, size: 18),
        const SizedBox(height: 6),
        Text('$count',
            style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: color)),
        Text('$pct%',
            style:
                TextStyle(fontSize: 11, color: color.withAlpha(180))),
        const SizedBox(height: 4),
        Text(label,
            textAlign: TextAlign.center,
            style: TextStyle(
                fontSize: 10,
                color: Colors.grey[600],
                fontWeight: FontWeight.w500)),
      ]),
    );
  }

  Widget _buildSentimentTile({
    required String label,
    required int count,
    required int total,
    required Color color,
    required IconData icon,
  }) {
    final pct = total > 0 ? (count / total * 100).round() : 0;
    return Container(
      padding:
          const EdgeInsets.symmetric(vertical: 12, horizontal: 12),
      decoration: BoxDecoration(
        color: color.withAlpha(18),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withAlpha(50)),
      ),
      child: Row(children: [
        Icon(icon, color: color, size: 22),
        const SizedBox(width: 10),
        Expanded(
            child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
              Text(label,
                  style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: Colors.grey[700])),
              const SizedBox(height: 2),
              Text('$count articles',
                  style: TextStyle(
                      fontSize: 11, color: Colors.grey[500])),
            ])),
        Text('$pct%',
            style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: color)),
      ]),
    );
  }

  Widget _buildSummaryItem({
    required String label,
    required String value,
    required Color color,
    required IconData icon,
  }) {
    return Column(mainAxisSize: MainAxisSize.min, children: [
      Icon(icon, size: 18, color: color),
      const SizedBox(height: 6),
      Text(value,
          style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.bold,
              color: color),
          textAlign: TextAlign.center,
          maxLines: 1,
          overflow: TextOverflow.ellipsis),
      const SizedBox(height: 3),
      Text(label,
          style: TextStyle(fontSize: 10, color: Colors.grey[500]),
          textAlign: TextAlign.center),
    ]);
  }

  Widget _buildVerticalDivider() {
    return Container(
        width: 1,
        height: 52,
        color: Colors.grey[200],
        margin: const EdgeInsets.symmetric(horizontal: 4));
  }

  Widget _buildDetailCard(
      String label, String value, Color color, IconData icon) {
    return Card(
      margin: const EdgeInsets.only(bottom: 10),
      elevation: 1,
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: color.withAlpha(40),
          child: Icon(icon, color: color, size: 20),
        ),
        title: Text(label,
            style: const TextStyle(
                fontSize: 14, fontWeight: FontWeight.w500)),
        trailing: Text(value,
            style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: color)),
      ),
    );
  }

  Widget _buildSectionTitle(String title, IconData icon) {
    return Row(children: [
      Icon(icon, size: 20, color: Colors.blue[700]),
      const SizedBox(width: 8),
      Flexible(
        child: Text(title,
            style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Colors.grey[800])),
      ),
    ]);
  }
}
