import 'dart:async';
import 'dart:developer' as developer;

import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../models/article.dart';
import '../services/api_service.dart';

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

  /// Convenience constructor — build from an Article model directly.
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

  // Fact-check state — loaded lazily on expand
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

  // ── Reading tracking ────────────────────────────────────────────────────────

  Future<void> _trackReadingTime() async {
    if (_secondsSpent < 3 || _hasTracked) return;
    _hasTracked = true;
    try {
      await _api.trackReading(
        articleId: widget.id,
        timeSpentSeconds: _secondsSpent,
      );
      developer.log(
        'Reading tracked: ${widget.id}, ${_secondsSpent}s',
        name: 'ArticleDetailScreen',
      );
    } catch (e) {
      developer.log(
        'Failed to track reading: $e',
        name: 'ArticleDetailScreen',
        error: e,
      );
    }
  }

  // ── Fact-check loading ──────────────────────────────────────────────────────

  Future<void> _loadFactChecks() async {
    if (_loadedFactChecks != null && _loadedFactChecks!.isNotEmpty) return;

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
      developer.log(
        'Failed to load fact-checks: $e',
        name: 'ArticleDetailScreen',
        error: e,
      );
    } finally {
      if (mounted) setState(() => _factCheckLoading = false);
    }
  }

  // ── Colour/label helpers ────────────────────────────────────────────────────

  Color _getBiasColor(double? score) {
    if (score == null) return Colors.grey;
    if (score < -0.3) return Colors.blue[700]!;
    if (score > 0.3) return Colors.red[700]!;
    return Colors.purple[400]!;
  }

  String _getBiasLabel(double? score) {
    if (score == null) return 'Pending';
    if (score < -0.5) return 'Left';
    if (score < -0.2) return 'Center-Left';
    if (score < 0.2) return 'Center';
    if (score < 0.5) return 'Center-Right';
    return 'Right';
  }

  Color _getSentimentColor(double? score) {
    if (score == null) return Colors.grey;
    if (score > 0.1) return Colors.green[700]!;
    if (score < -0.1) return Colors.orange[700]!;
    return Colors.grey[600]!;
  }

  String _getSentimentLabel(double? score) {
    if (score == null) return 'Pending';
    if (score > 0.3) return 'Positive';
    if (score < -0.3) return 'Negative';
    return 'Neutral';
  }

  Color _getCredibilityColor(double? score) {
    if (score == null) return Colors.grey;
    if (score >= 75) return Colors.green[700]!;
    if (score >= 50) return Colors.orange[700]!;
    return Colors.red[700]!;
  }

  String _getCredibilityLabel(double? score) {
    if (score == null) return 'Unverified';
    if (score >= 75) return 'Credible';
    if (score >= 50) return 'Mixed';
    return 'Questionable';
  }

  String _getRulingEmoji(String ruling) {
    final r = ruling.toLowerCase();
    if (r.contains('true') && !r.contains('mostly')) return '✅';
    if (r.contains('mostly true')) return '🟢';
    if (r.contains('half')) return '🟡';
    if (r.contains('mostly false')) return '🟠';
    if (r.contains('false') || r.contains('pants')) return '❌';
    return '❓';
  }

  // ── URL launcher ────────────────────────────────────────────────────────────

  Future<void> _launchURL() async {
    final uri = Uri.parse(widget.url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  Future<void> _launchFactCheckURL(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  // ── Widgets ─────────────────────────────────────────────────────────────────

  Widget _buildCredibilityCard() {
    final score = widget.credibilityScore;
    final color = _getCredibilityColor(score);
    final label = _getCredibilityLabel(score);

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
                        color: Colors.grey[600],
                        fontSize: 13,
                      ),
                    ),
                  if (widget.credibilityReason != null)
                    Text(
                      widget.credibilityReason!,
                      style: TextStyle(
                        color: Colors.grey[500],
                        fontSize: 12,
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
    final emoji = _getRulingEmoji(ruling.toString());

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            claim,
            style: const TextStyle(
              fontStyle: FontStyle.italic,
              fontSize: 13,
            ),
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
                  style: TextStyle(
                    color: Colors.grey[600],
                    fontSize: 12,
                  ),
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

  // ── Build ───────────────────────────────────────────────────────────────────

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
          actions: [
            IconButton(
              icon: const Icon(Icons.open_in_browser),
              tooltip: 'Open in browser',
              onPressed: _launchURL,
            ),
          ],
        ),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Source badge
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 6,
                ),
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

              // Title
              Text(
                widget.title,
                style: Theme.of(context)
                    .textTheme
                    .headlineSmall
                    ?.copyWith(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),

              // Analysis chips
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  Chip(
                    avatar: Icon(
                      Icons.balance,
                      size: 16,
                      color: _getBiasColor(widget.biasScore),
                    ),
                    label: Text(_getBiasLabel(widget.biasScore)),
                    backgroundColor: _getBiasColor(widget.biasScore)
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
                        color: _getSentimentColor(widget.sentimentScore),
                      ),
                      label: Text(_getSentimentLabel(widget.sentimentScore)),
                      backgroundColor:
                          _getSentimentColor(widget.sentimentScore)
                              .withAlpha((255 * 0.2).round()),
                    ),
                  if (widget.biasIntensity != null)
                    Chip(
                      label: Text(
                        '${(widget.biasIntensity! * 100).round()}% Intensity',
                      ),
                      backgroundColor: Colors.grey.shade200,
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

              // Credibility card
              _buildCredibilityCard(),

              const SizedBox(height: 8),

              // Fact-check expandable section
              _buildFactCheckSection(),

              const Divider(height: 32),

              // Content
              if (widget.content != null && widget.content!.isNotEmpty)
                Text(
                  widget.content!,
                  style: Theme.of(context)
                      .textTheme
                      .bodyLarge
                      ?.copyWith(height: 1.6),
                )
              else
                Center(
                  child: Column(
                    children: [
                      const Icon(
                        Icons.article_outlined,
                        size: 64,
                        color: Colors.grey,
                      ),
                      const SizedBox(height: 16),
                      const Text('Content not yet available.'),
                      const SizedBox(height: 16),
                      ElevatedButton.icon(
                        onPressed: _launchURL,
                        icon: const Icon(Icons.open_in_browser),
                        label: const Text('Read on source website'),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
