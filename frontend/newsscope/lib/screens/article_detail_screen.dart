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
  final List<Map<String, dynamic>>? biasExplanation;
  final String? category;
  final String? politicalBias;
  final double? politicalBiasScore;

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
    this.biasExplanation,
    this.category,
    this.politicalBias,
    this.politicalBiasScore,
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
      biasExplanation: article.biasExplanation,
      category: article.category,
      politicalBias: article.politicalBias,
      politicalBiasScore: article.politicalBiasScore,
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

  Future<void> _loadFactChecks() async {
    if (_loadedFactChecks != null && _loadedFactChecks!.isNotEmpty) {
      return;
    }
    setState(() => _factCheckLoading = true);
    try {
      final result = await _api.triggerFactCheck(widget.id);
      if (result != null && mounted) {
        setState(() {
          _loadedFactChecks =
              result['fact_checks'] as Map<String, dynamic>?;
        });
      }
    } catch (e) {
      developer.log('Failed to load fact-checks: $e',
          name: 'ArticleDetailScreen', error: e);
    } finally {
      if (mounted) setState(() => _factCheckLoading = false);
    }
  }

  void _launchFactCheckURL(String url) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Source: $url')),
    );
  }

  double _biasScoreToPosition(double? score) {
    if (score == null) return 0.5;
    return ((score.clamp(-1.0, 1.0) + 1.0) / 2.0);
  }

  Color _positionToColor(double position) {
    if (position < 0.2) return Colors.blue[800]!;
    if (position < 0.4) return Colors.cyan[600]!;
    if (position < 0.6) return Colors.teal[600]!;
    if (position < 0.8) return Colors.orange[600]!;
    return Colors.red[800]!;
  }

  Color _biasLabelColor() {
    final label = (widget.generalBias ?? '').toUpperCase();
    if (label.contains('LEFT')) return Colors.blue[700]!;
    if (label.contains('RIGHT')) return Colors.red[700]!;
    return Colors.teal[600]!;
  }

  String _formatGeneralBias(String? raw) {
    if (raw == null || raw.isEmpty) return 'Unbiased';
    final lower = raw.toLowerCase();
    return lower[0].toUpperCase() + lower.substring(1);
  }

  // ── Bias Breakdown card ────────────────────────────────────────────────────

  Widget _buildBiasBreakdownCard() {
    final sourceLabel = getBiasLabelShort(widget.biasScore);
    final sourceColor = getBiasColor(widget.biasScore);

    final articleLabel = getPoliticalBiasLabel(widget.politicalBias);
    final articleColor = getPoliticalBiasColor(widget.politicalBias);

    bool diverges = false;
    if (widget.politicalBias != null && widget.biasScore != null) {
      diverges =
          sourceLabel.toLowerCase() != articleLabel.toLowerCase();
    }

    final confidenceText = widget.politicalBiasScore != null
        ? '${(widget.politicalBiasScore! * 100).round()}% confidence'
        : '';

    return Card(
      elevation: 2,
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.compare_arrows,
                    size: 18, color: Colors.blue[700]),
                const SizedBox(width: 8),
                const Text(
                  'Bias Breakdown',
                  style: TextStyle(
                      fontWeight: FontWeight.bold, fontSize: 14),
                ),
                const Spacer(),
                if (diverges)
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: Colors.orange.withAlpha(35),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.orange[700]!),
                    ),
                    child: Text(
                      'Diverges',
                      style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: Colors.orange[800],
                      ),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 14),
            Row(
              children: [
                Expanded(
                  child: _buildBreakdownTile(
                    title: 'Outlet Political Bias',
                    subtitle: widget.sourceName,
                    label: sourceLabel,
                    color: sourceColor,
                    icon: Icons.source,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _buildBreakdownTile(
                    title: 'Article Political Bias',
                    subtitle: confidenceText.isEmpty
                        ? 'RoBERTa classifier'
                        : confidenceText,
                    label: articleLabel,
                    color: articleColor,
                    icon: Icons.article_outlined,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              diverges
                  ? 'The article\'s text was classified independently '
                      'of its outlet — this piece reads differently to '
                      'the outlet\'s usual leaning.'
                  : 'The outlet rating reflects the publisher\'s '
                      'typical leaning; the article-level label is '
                      'produced by RoBERTa on this article\'s own text.',
              style: TextStyle(
                  color: Colors.grey[600], fontSize: 11, height: 1.4),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBreakdownTile({
    required String title,
    required String subtitle,
    required String label,
    required Color color,
    required IconData icon,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 10),
      decoration: BoxDecoration(
        color: color.withAlpha(18),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withAlpha(50)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 13, color: color),
              const SizedBox(width: 5),
              Expanded(
                child: Text(
                  title.toUpperCase(),
                  style: TextStyle(
                    fontSize: 9,
                    fontWeight: FontWeight.w700,
                    color: color,
                    letterSpacing: 0.8,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            label,
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            subtitle,
            style: TextStyle(fontSize: 10, color: Colors.grey[600]),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }

  Widget _buildBiasExplanationSection() {
    final explanation = widget.biasExplanation;
    if (explanation == null || explanation.isEmpty) {
      return const SizedBox.shrink();
    }

    final labelColor = _biasLabelColor();
    final labelText = _formatGeneralBias(widget.generalBias);

    return Card(
      elevation: 2,
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.psychology_outlined,
                    size: 18, color: labelColor),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Why "$labelText" was flagged',
                    style: const TextStyle(
                        fontWeight: FontWeight.bold, fontSize: 14),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              'Top words driving this classification',
              style:
                  TextStyle(color: Colors.grey[500], fontSize: 12),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: explanation.map((e) {
                final word = e['word']?.toString() ?? '';
                final isTowards = e['direction'] == 'towards';
                final weight =
                    (e['weight'] as num?)?.toDouble() ?? 0.0;
                final opacity =
                    (0.4 + weight.clamp(0.0, 1.0) * 0.6);

                return Tooltip(
                  message: isTowards
                      ? 'Pushes toward $labelText'
                      : 'Pushes against $labelText',
                  child: Chip(
                    label: Text(
                      word,
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                        color: isTowards
                            ? Colors.white
                            : Colors.grey[700],
                      ),
                    ),
                    backgroundColor: isTowards
                        ? labelColor.withValues(alpha: opacity)
                        : Colors.grey[200],
                    side: BorderSide(
                      color: isTowards
                          ? labelColor.withValues(alpha: 0.3)
                          : Colors.grey[300]!,
                      width: 1,
                    ),
                    padding: EdgeInsets.zero,
                    materialTapTargetSize:
                        MaterialTapTargetSize.shrinkWrap,
                  ),
                );
              }).toList(),
            ),
            const SizedBox(height: 8),
            Text(
              'Filled = pushes toward label  ·  Grey = pushes against',
              style:
                  TextStyle(color: Colors.grey[400], fontSize: 10),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSpectrumBar() {
    final pos = _biasScoreToPosition(widget.biasScore);
    final markerColor = _positionToColor(pos);
    final label = getBiasLabel(widget.biasScore);

    return Card(
      elevation: 2,
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
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
                  'Outlet Political Bias Spectrum',
                  style: TextStyle(
                      fontWeight: FontWeight.bold, fontSize: 14),
                ),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 10, vertical: 4),
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
                                  Colors.blue[800]!,
                                  Colors.cyan[600]!,
                                  Colors.teal[600]!,
                                  Colors.orange[600]!,
                                  Colors.red[800]!,
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
                      mainAxisAlignment:
                          MainAxisAlignment.spaceBetween,
                      children: [
                        Text('Left',
                            style: TextStyle(
                                fontSize: 10,
                                color: Colors.blue[800])),
                        Text('C-Left',
                            style: TextStyle(
                                fontSize: 10,
                                color: Colors.cyan[600])),
                        Text('Centre',
                            style: TextStyle(
                                fontSize: 10,
                                color: Colors.teal[600])),
                        Text('C-Right',
                            style: TextStyle(
                                fontSize: 10,
                                color: Colors.orange[600])),
                        Text('Right',
                            style: TextStyle(
                                fontSize: 10,
                                color: Colors.red[800])),
                      ],
                    ),
                  ],
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCredibilityCard() {
    final score = widget.credibilityScore;
    final color = getCredibilityColor(score);
    final label = getCredibilityLabel(score);

    return Card(
      elevation: 2,
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            CircleAvatar(
              radius: 28,
              backgroundColor:
                  color.withAlpha((255 * 0.15).round()),
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
                  if (widget.credibilityReason != null &&
                      widget.credibilityReason!.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(top: 4),
                      child: Text(
                        widget.credibilityReason!,
                        style: TextStyle(
                            color: Colors.grey[500], fontSize: 12),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFactCheckSection() {
    return Card(
      elevation: 2,
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      clipBehavior: Clip.antiAlias,
      child: ExpansionTile(
        leading:
            Icon(Icons.fact_check_outlined, color: Colors.blue[700]),
        title: const Text('Fact Checks',
            style:
                TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
        subtitle: Text(
          _loadedFactChecks != null && _loadedFactChecks!.isNotEmpty
              ? '${_loadedFactChecks!.length} claim(s) checked'
              : '${widget.claimsChecked ?? 0} claim(s) checked',
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
          else if (_loadedFactChecks == null ||
              _loadedFactChecks!.isEmpty)
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
      ),
    );
  }

  Widget _buildFactCheckTile(String claim, dynamic data) {
    final ruling =
        data is Map ? (data['ruling'] ?? 'Unknown') : 'Unknown';
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
            style: const TextStyle(
                fontStyle: FontStyle.italic, fontSize: 13),
            maxLines: 3,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 4),
          Row(
            children: [
              Text('$emoji $ruling',
                  style:
                      const TextStyle(fontWeight: FontWeight.w600)),
              if (speaker != null && speaker != 'N/A') ...[
                const SizedBox(width: 8),
                Text('— $speaker',
                    style: TextStyle(
                        color: Colors.grey[600], fontSize: 12)),
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

  Widget _buildArticleContentCard() {
    final wordCount =
        widget.content!.trim().split(RegExp(r'\s+')).length;
    final readingMins = (wordCount / 200).ceil();

    return Card(
      elevation: 2,
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
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

  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: true,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop && !_hasTracked) await _trackReadingTime();
      },
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Article'),
        ),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Source badge + category
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
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
                  if (widget.category != null &&
                      widget.category!.isNotEmpty)
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: Colors.blue[50],
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: Colors.blue[200]!),
                      ),
                      child: Text(
                        formatCategory(widget.category),
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: Colors.blue[700],
                        ),
                      ),
                    ),
                ],
              ),
              const SizedBox(height: 12),

              // Title
              Text(
                widget.title,
                style: Theme.of(context)
                    .textTheme
                    .headlineSmall
                    ?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),

              // Summary chips — full label names in detail screen
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  Chip(
                    avatar: Icon(Icons.source,
                        size: 16,
                        color: getBiasColor(widget.biasScore)),
                    label: Text(
                        'Outlet Political Bias: '
                        '${getBiasLabel(widget.biasScore)}'),
                    backgroundColor: getBiasColor(widget.biasScore)
                        .withAlpha((255 * 0.2).round()),
                  ),
                  if (widget.politicalBias != null)
                    Chip(
                      avatar: Icon(Icons.article_outlined,
                          size: 16,
                          color: getPoliticalBiasColor(
                              widget.politicalBias)),
                      label: Text(
                          'Article Political Bias: '
                          '${getPoliticalBiasLabel(widget.politicalBias)}'),
                      backgroundColor:
                          getPoliticalBiasColor(widget.politicalBias)
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
                      label: Text(
                          'Sentiment: '
                          '${getSentimentLabel(widget.sentimentScore)}'),
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
                      label:
                          Text(_formatGeneralBias(widget.generalBias)),
                      backgroundColor: widget.generalBias == 'BIASED'
                          ? Colors.orange.withAlpha(40)
                          : Colors.green.withAlpha(40),
                    ),
                ],
              ),
              const SizedBox(height: 16),

              _buildBiasBreakdownCard(),
              const SizedBox(height: 12),
              _buildSpectrumBar(),
              const SizedBox(height: 12),
              _buildBiasExplanationSection(),
              if (widget.biasExplanation != null &&
                  widget.biasExplanation!.isNotEmpty)
                const SizedBox(height: 12),
              _buildCredibilityCard(),
              const SizedBox(height: 8),
              _buildFactCheckSection(),
              const SizedBox(height: 16),

              if (widget.content != null && widget.content!.isNotEmpty)
                _buildArticleContentCard()
              else
                Center(
                  child: Column(
                    children: [
                      const Icon(Icons.article_outlined,
                          size: 64, color: Colors.grey),
                      const SizedBox(height: 16),
                      Text('Content not yet available.',
                          style:
                              TextStyle(color: Colors.grey[500])),
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
