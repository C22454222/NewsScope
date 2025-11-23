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

  Color _getBiasColor(double? score) {
    if (score == null) return Colors.grey;
    if (score < -0.3) return Colors.blue[300]!;
    if (score > 0.3) return Colors.red[300]!;
    return Colors.purple[200]!;
  }

  String _getBiasLabel(double? score) {
    if (score == null) return "Pending";
    if (score < -0.3) return "Left";
    if (score > 0.3) return "Right";
    return "Center";
  }

  Future<void> _launchURL() async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  @override
  Widget build(BuildContext context) {
    // Capture sentiment score locally to promote it to non-nullable if checked
    final sentiment = sentimentScore;

    return Scaffold(
      appBar: AppBar(
        title: const Text("Article"),
        actions: [
          IconButton(
            icon: const Icon(Icons.open_in_browser),
            onPressed: _launchURL,
            tooltip: "Open in browser",
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Title
            Text(
              title,
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 12),

            // Source & Bias chips
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
                  backgroundColor: _getBiasColor(biasScore).withAlpha((255 * 0.3).round()),
                ),
                // Null check logic fixed here
                if (sentiment != null)
                  Chip(
                    label: Text("Sentiment: ${sentiment.toStringAsFixed(2)}"),
                    backgroundColor: Colors.amber[100],
                  ),
              ],
            ),

            const Divider(height: 32),

            // Content
            if (content != null && content!.isNotEmpty)
              Text(
                content!,
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(height: 1.6),
              )
            else
              Column(
                children: [
                  const Icon(Icons.article_outlined, size: 64, color: Colors.grey),
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
