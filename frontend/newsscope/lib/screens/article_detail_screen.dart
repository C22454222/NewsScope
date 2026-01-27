// lib/screens/article_detail_screen.dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
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
  });

  @override
  State<ArticleDetailScreen> createState() => _ArticleDetailScreenState();
}

class _ArticleDetailScreenState extends State<ArticleDetailScreen> {
  final ApiService _api = ApiService();
  Timer? _trackingTimer;
  int _secondsSpent = 0;

  @override
  void initState() {
    super.initState();

    // Track time every 5 seconds
    _trackingTimer = Timer.periodic(
      const Duration(seconds: 5),
      (_) {
        _secondsSpent += 5;
      },
    );
  }

  @override
  void dispose() {
    _trackingTimer?.cancel();
    _trackReadingTime();
    super.dispose();
  }

  Future<void> _trackReadingTime() async {
    if (_secondsSpent < 3) return; // Ignore quick exits

    await _api.trackReading(
      articleId: widget.id,
      timeSpentSeconds: _secondsSpent,
    );
  }

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

  Future<void> _launchURL() async {
    final uri = Uri.parse(widget.url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Article'),
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: () {
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
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
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 16),

            // Analysis chips
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                // Bias chip
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

                // Sentiment chip
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
                    backgroundColor: _getSentimentColor(widget.sentimentScore)
                        .withAlpha((255 * 0.2).round()),
                  ),

                // Intensity chip
                if (widget.biasIntensity != null)
                  Chip(
                    label: Text(
                      '${(widget.biasIntensity! * 100).round()}% Biased',
                    ),
                    backgroundColor: Colors.grey.shade200,
                  ),
              ],
            ),

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
    );
  }
}
