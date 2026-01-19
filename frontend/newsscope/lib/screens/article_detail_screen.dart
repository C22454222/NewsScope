// lib/screens/article_detail_screen.dart
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';


class ArticleDetailScreen extends StatelessWidget {
  final String title;
  final String sourceName;
  final String? content;
  final String url;
  final double? biasScore;
  final double? sentimentScore;

  const ArticleDetailScreen({
    super.key,
    required this.title,
    required this.sourceName,
    this.content,
    required this.url,
    this.biasScore,
    this.sentimentScore,
  });

  /// Determines the color of the bias chip based on the score range.
  Color _getBiasColor(double? score) {
    if (score == null) return Colors.grey;
    if (score < -0.3) return Colors.blue[300]!;
    if (score > 0.3) return Colors.red[300]!;
    return Colors.purple[200]!;
  }

  /// Maps the numerical bias score to a human-readable label.
  String _getBiasLabel(double? score) {
    if (score == null) return "Pending";
    if (score < -0.3) return "Left";
    if (score > 0.3) return "Right";
    return "Center";
  }

  /// Determines sentiment chip color.
  Color _getSentimentColor(double? score) {
    if (score == null) return Colors.grey;
    if (score > 0.1) return Colors.green[300]!;
    if (score < -0.1) return Colors.orange[300]!;
    return Colors.grey[300]!;
  }

  /// Maps sentiment score to human-readable label.
  String _getSentimentLabel(double? score) {
    if (score == null) return "Pending";
    if (score > 0.1) return "Positive";
    if (score < -0.1) return "Negative";
    return "Neutral";
  }

  /// Opens the full article in the default external browser.
  Future<void> _launchURL() async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  @override
  Widget build(BuildContext context) {
    final sentiment = sentimentScore;

    return Scaffold(
      appBar: AppBar(
        title: const Text("Article"),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Article Title
            Text(
              title,
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 12),

            // Metadata Chips (Source, Bias, Sentiment)
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                Chip(
                  avatar: const Icon(Icons.source, size: 18),
                  label: Text(sourceName),
                ),
                Chip(
                  label: Text(_getBiasLabel(biasScore)),
                  backgroundColor:
                      _getBiasColor(biasScore).withAlpha((255 * 0.3).round()),
                ),
                if (sentiment != null)
                  Chip(
                    avatar: Icon(
                      sentiment > 0
                          ? Icons.sentiment_satisfied
                          : Icons.sentiment_dissatisfied,
                      size: 18,
                    ),
                    label: Text(_getSentimentLabel(sentiment)),
                    backgroundColor: _getSentimentColor(sentiment)
                        .withAlpha((255 * 0.3).round()),
                  ),
              ],
            ),

            const Divider(height: 32),

            // Article Content or Fallback
            if (content != null && content!.isNotEmpty)
              Text(
                content!,
                style: Theme.of(context)
                    .textTheme
                    .bodyLarge
                    ?.copyWith(height: 1.6),
              )
            else
              Column(
                children: [
                  const Icon(Icons.article_outlined,
                      size: 64, color: Colors.grey),
                  const SizedBox(height: 16),
                  const Text("Content not yet available."),
                  const SizedBox(height: 8),
                  ElevatedButton.icon(
                    onPressed: _launchURL,
                    icon: const Icon(Icons.open_in_browser),
                    label: const Text("Read on source website"),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}
