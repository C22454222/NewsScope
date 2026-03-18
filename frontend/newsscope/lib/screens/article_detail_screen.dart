import 'dart:async';
import 'dart:developer' as developer;

import 'package:flutter/material.dart';

import '../models/article.dart';
import '../services/api_service.dart';
import '../utils/score_helpers.dart';

class ArticleDetailScreen extends StatefulWidget {
  final String id;
  final String title;
  final String sourceName;
  final String? content;
  final String url;
  final double? biasScore;
  final double? biasIntensity;
  final double? sentimentScore;
  final double? credibilityScore;
  final Map<String, dynamic>? factChecks;
  final int? claimsChecked;
  final String? credibilityReason;
  final String? generalBias;

  const ArticleDetailScreen({
    super.key,
    required this.id,
    required this.title,
    required this.sourceName,
    this.content,
    required this.url,
    this.biasScore,
    this.biasIntensity,
    this.sentimentScore,
    this.credibilityScore,
    this.factChecks,
    this.claimsChecked,
    this.credibilityReason,
    this.generalBias,
  });

  factory ArticleDetailScreen.fromArticle(Article article) {
    return ArticleDetailScreen(
      id: article.id,
      title: article.title,
      sourceName: article.source,
      content: article.content,
      url: article.url,
      biasScore: article.biasScore,
      biasIntensity: article.biasIntensity,
      sentimentScore: article.sentimentScore,
      credibilityScore: article.credibilityScore,
      factChecks: article.factChecks,
      claimsChecked: article.claimsChecked,
      credibilityReason: article.credibilityReason,
      generalBias: article.generalBias,
    );
  }

  @override
  State<ArticleDetailScreen> createState() => _ArticleDetailScreenState();
}

class _ArticleDetailScreenState extends State<ArticleDetailScreen> {
  final ApiService _api = ApiService();
  Timer? _trackingTimer;
  int _secondsSpent = 0;
  bool _hasTracked = false;

  bool _factCheckExpanded = false;
  bool _factCheckLoading = false;
  Map<String, dynamic>? _loadedFactChecks;

  @override
  void initState() {
    super.initState();
    _loadedFactChecks = widget.factChecks;
    _trackingTimer = Timer.periodic(
      const Duration(seconds: 5),
      (_) => _secondsSpent += 5,
    );
  }

  @override
  void dispose() {
    _trackingTimer?.cancel();
    _trackReadingTime();
    super.dispose();
  }

  // ── Reading tracking ──────────────────────────────────────────────────────

  Future<void> _trackReadingTime() async {
    if (_secondsSpent < 3 || _hasTracked) return;
    _hasTracked = true;
    try {
      await _api.trackReading(
        articleId: widget.id,
        timeSpentSeconds: _secondsSpent,
      );
    } catch (e) {
      developer.log('Failed to track reading: $e',
          name: 'ArticleDetailScreen', error: e);
    }
  }

  // ── Fact-check loading ────────────────────────────────────────────────────

  Future<void> _loadFactChecks() async {
    if (_loadedFactChecks != null && _loadedFactChecks!.isNotEmpty) return;
    setState(() => _factCheckLoading = true);
    try {
      final result = await _api.triggerFactCheck(widget.id);
      if (result != null && mounted) {
        setState(() {
          _loadedFactChecks = result['fact_checks'] as Map<String, dynamic>?;
        });
      }
    } catch (e) {
      developer.log('Failed to load fact-checks: $e',
          name: 'ArticleDetailScreen', error: e);
    } finally {
      if (mounted) setState(() => _factCheckLoading = false);
    }
  }

  // ── Fact-check URL launch ─────────────────────────────────────────────────

  void _launchFactCheckURL(String url) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Source: $url')),
    );
  }

  // ── Spectrum bar ──────────────────────────────────────────────────────────

  double _biasScoreToPosition(double? score) {
    if (score == null) return 0.5;
    return ((score.clamp(-1.0, 1.0) + 1.0) / 2.0);
  }

  Color _positionToColor(double position) {
    if (position < 0.2) return Colors.blue[700]!;
    if (position < 0.4) return Colors.cyan[600]!;
    if (position < 0.6) return Colors.purple[400]!;
    if (position < 0.8) return Colors.orange[600]!;
    return Colors.red[700]!;
  }

  Widget _buildSpectrumBar() {
    final pos = _biasScoreToPosition(widget.biasScore);
    final markerColor = _positionToColor(pos);
    final label = getBiasLabel(widget.biasScore);

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
                Icon(Icons.balance, size: 18, color: Colors.blue[700]),
                const SizedBox(width: 8),
                const Text(
                  'Ideological Spectrum',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                ),
                const Spacer(),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: markerColor.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: markerColor, width: 1),
                  ),
                  child: Text(
                    label,
                    style: TextStyle(
                      color: markerColor,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            LayoutBuilder(
              builder: (context, constraints) {
                final barWidth = constraints.maxWidth;
                const markerSize = 20.0;

                return Column(
                  children: [
                    Stack(
                      clipBehavior: Clip.none,
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Container(
                            height: 16,
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  Colors.blue[700]!,
                                  Colors.cyan[600]!,
                                  Colors.purple[400]!,
                                  Colors.orange[600]!,
                                  Colors.red[700]!,
                                ],
                              ),
                            ),
                          ),
                        ),
                        Positioned(
                          left: (barWidth * pos - markerSize / 2)
                              .clamp(0, barWidth - markerSize),
                          top: -4,
                          child: Container(
                            width: markerSize,
                            height: markerSize,
                            decoration: BoxDecoration(
                              color: Colors.white,
                              shape: BoxShape.circle,
                              border: Border.all(
                                  color: markerColor, width: 2.5),
                              boxShadow: const [
                                BoxShadow(
                                  color: Colors.black26,
                                  blurRadius: 4,
                                  offset: Offset(0, 2),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('Left',
                            style: TextStyle(
                                fontSize: 10, color: Colors.blue[700])),
                        Text('C-Left',
                            style: TextStyle(
                                fontSize: 10, color: Colors.cyan[600])),
                        Text('Centre',
                            style: TextStyle(
                                fontSize: 10, color: Colors.purple[400])),
                        Text('C-Right',
                            style: TextStyle(
                                fontSize: 10, color: Colors.orange[600])),
                        Text('Right',
                            style: TextStyle(
                                fontSize: 10, color: Colors.red[700])),
                      ],
                    ),
                    if (widget.biasIntensity != null) ...[
                      const SizedBox(height: 8),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: widget.biasIntensity!.clamp(0.0, 1.0),
                          minHeight: 6,
                          backgroundColor: Colors.grey.shade200,
                          valueColor:
                              AlwaysStoppedAnimation<Color>(markerColor),
                        ),
                      ),
                      const SizedBox(height: 4),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            'Bias intensity',
                            style: TextStyle(
                                fontSize: 10, color: Colors.grey[500]),
                          ),
                          Text(
                            '${(widget.biasIntensity! * 100).round()}%',
                            style: TextStyle(
                              fontSize: 10,
                              fontWeight: FontWeight.bold,
                              color: markerColor,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ],
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  // ── Credibility card ──────────────────────────────────────────────────────

  Widget _buildCredibilityCard() {
    final score = widget.credibilityScore;
    final color = getCredibilityColor(score);
    final label = getCredibilityLabel(score);

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            CircleAvatar(
              radius: 28,
              backgroundColor: color.withAlpha((255 * 0.15).round()),
              child: Text(
                score != null ? '${score.round()}' : '?',
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.bold,
                  fontSize: 18,
                ),
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Credibility: $label',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: color,
                      fontSize: 16,
                    ),
                  ),
                  if (widget.claimsChecked != null &&
                      widget.claimsChecked! > 0)
                    Text(
                      '${widget.claimsChecked} claim(s) verified',
                      style: TextStyle(
                          color: Colors.grey[600], fontSize: 13),
                    ),
                  if (widget.credibilityReason != null)
                    Text(
                      widget.credibilityReason!,
                      style: TextStyle(
                          color: Colors.grey[500], fontSize: 12),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Fact-check section ────────────────────────────────────────────────────

  Widget _buildFactCheckSection() {
    return ExpansionTile(
      leading: const Icon(Icons.fact_check_outlined),
      title: const Text(
        'Fact Checks',
        style: TextStyle(fontWeight: FontWeight.w600),
      ),
      subtitle: Text(
        _loadedFactChecks != null && _loadedFactChecks!.isNotEmpty
            ? '${_loadedFactChecks!.length} claim(s) checked'
            : 'Tap to load',
        style: TextStyle(color: Colors.grey[600], fontSize: 12),
      ),
      initiallyExpanded: _factCheckExpanded,
      onExpansionChanged: (expanded) {
        setState(() => _factCheckExpanded = expanded);
        if (expanded) _loadFactChecks();
      },
      children: [
        if (_factCheckLoading)
          const Padding(
            padding: EdgeInsets.all(16),
            child: Center(child: CircularProgressIndicator()),
          )
        else if (_loadedFactChecks == null || _loadedFactChecks!.isEmpty)
          Padding(
            padding: const EdgeInsets.all(16),
            child: Text(
              'No fact-check data available for this article.',
              style: TextStyle(color: Colors.grey[600]),
            ),
          )
        else
          ..._loadedFactChecks!.entries.map(
            (entry) => _buildFactCheckTile(entry.key, entry.value),
          ),
      ],
    );
  }

  Widget _buildFactCheckTile(String claim, dynamic data) {
    final ruling = data is Map ? (data['ruling'] ?? 'Unknown') : 'Unknown';
    final url = data is Map ? data['url'] as String? : null;
    final speaker = data is Map ? data['speaker'] as String? : null;
    final emoji = getRulingEmoji(ruling.toString());

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            claim,
            style:
                const TextStyle(fontStyle: FontStyle.italic, fontSize: 13),
            maxLines: 3,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              Text(
                '$emoji $ruling',
                style: const TextStyle(fontWeight: FontWeight.w600),
              ),
              if (speaker != null && speaker != 'N/A') ...[
                const SizedBox(width: 8),
                Text(
                  '— $speaker',
                  style:
                      TextStyle(color: Colors.grey[600], fontSize: 12),
                ),
              ],
              if (url != null) ...[
                const Spacer(),
                GestureDetector(
                  onTap: () => _launchFactCheckURL(url),
                  child: Text(
                    'Source',
                    style: TextStyle(
                      color: Colors.blue[700],
                      fontSize: 12,
                      decoration: TextDecoration.underline,
                    ),
                  ),
                ),
              ],
            ],
          ),
          const Divider(),
        ],
      ),
    );
  }

  // ── Article content card ──────────────────────────────────────────────────

  Widget _buildArticleContentCard() {
    final wordCount = widget.content!.trim().split(RegExp(r'\s+')).length;
    final readingMins = (wordCount / 200).ceil();

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          border: Border(
            left: BorderSide(color: Colors.blue[700]!, width: 4),
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ── Card header ───────────────────────────────────────────────
              Row(
                children: [
                  Icon(Icons.article, size: 16, color: Colors.blue[700]),
                  const SizedBox(width: 6),
                  Text(
                    'ARTICLE CONTENT',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: Colors.blue[700],
                      letterSpacing: 1.0,
                    ),
                  ),
                  const Spacer(),
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: Colors.grey[100],
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      '$wordCount words · $readingMins min read',
                      style: TextStyle(
                        fontSize: 10,
                        color: Colors.grey[500],
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              const Divider(height: 1),
              const SizedBox(height: 18),

              // ── Article body — full content, no truncation ────────────────
              SelectableText(
                widget.content!,
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      height: 1.8,
                      fontSize: 15.5,
                      color: Colors.grey[850],
                      letterSpacing: 0.1,
                    ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: true,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop && !_hasTracked) await _trackReadingTime();
      },
      child: Scaffold(
        // No actions — browser button removed.
        appBar: AppBar(
          title: const Text('Article'),
        ),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ── Source badge ──────────────────────────────────────────────
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.blue.shade50,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Text(
                  widget.sourceName,
                  style: TextStyle(
                    color: Colors.blue.shade700,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              const SizedBox(height: 12),

              // ── Title ─────────────────────────────────────────────────────
              Text(
                widget.title,
                style: Theme.of(context)
                    .textTheme
                    .headlineSmall
                    ?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),

              // ── Chips ─────────────────────────────────────────────────────
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  Chip(
                    avatar: Icon(
                      Icons.balance,
                      size: 16,
                      color: getBiasColor(widget.biasScore),
                    ),
                    label: Text(getBiasLabel(widget.biasScore)),
                    backgroundColor: getBiasColor(widget.biasScore)
                        .withAlpha((255 * 0.2).round()),
                  ),
                  if (widget.sentimentScore != null)
                    Chip(
                      avatar: Icon(
                        widget.sentimentScore! > 0
                            ? Icons.sentiment_satisfied
                            : widget.sentimentScore! < 0
                                ? Icons.sentiment_dissatisfied
                                : Icons.sentiment_neutral,
                        size: 16,
                        color: getSentimentColor(widget.sentimentScore),
                      ),
                      label:
                          Text(getSentimentLabel(widget.sentimentScore)),
                      backgroundColor:
                          getSentimentColor(widget.sentimentScore)
                              .withAlpha((255 * 0.2).round()),
                    ),
                  if (widget.generalBias != null)
                    Chip(
                      avatar: Icon(
                        widget.generalBias == 'BIASED'
                            ? Icons.warning_amber
                            : Icons.check_circle_outline,
                        size: 16,
                        color: widget.generalBias == 'BIASED'
                            ? Colors.orange[700]
                            : Colors.green[700],
                      ),
                      label: Text(widget.generalBias!),
                      backgroundColor: widget.generalBias == 'BIASED'
                          ? Colors.orange.withAlpha(40)
                          : Colors.green.withAlpha(40),
                    ),
                ],
              ),

              const SizedBox(height: 16),
              _buildSpectrumBar(),
              const SizedBox(height: 12),
              _buildCredibilityCard(),
              const SizedBox(height: 8),
              _buildFactCheckSection(),
              const SizedBox(height: 16),

              // ── Article content ───────────────────────────────────────────
              if (widget.content != null && widget.content!.isNotEmpty)
                _buildArticleContentCard()
              else
                Center(
                  child: Column(
                    children: [
                      const Icon(Icons.article_outlined,
                          size: 64, color: Colors.grey),
                      const SizedBox(height: 16),
                      Text(
                        'Content not yet available.',
                        style: TextStyle(color: Colors.grey[500]),
                      ),
                    ],
                  ),
                ),

              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}
