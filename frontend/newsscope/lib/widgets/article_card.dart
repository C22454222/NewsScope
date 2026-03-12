// lib/widgets/article_card.dart
import 'package:flutter/material.dart';

import '../models/article.dart';
import '../utils/score_helpers.dart';

/// Reusable article card used by HomeFeedTab and CompareScreen.
///
/// Displays title, source, date, category, bias chip, sentiment chip,
/// and credibility chip. Calls [onTap] with the article when tapped.
/// Stateless — all data comes from the Article model.
class ArticleCard extends StatelessWidget {
  final Article article;
  final VoidCallback onTap;

  const ArticleCard({
    super.key,
    required this.article,
    required this.onTap,
  });

  String _formatDate(DateTime? dt) {
    if (dt == null) return 'Unknown date';
    return '${dt.day.toString().padLeft(2, '0')}/'
        '${dt.month.toString().padLeft(2, '0')}/'
        '${dt.year}';
  }

  String _sanitiseTitle(String title, String url) {
    if (!title.startsWith('http') && !title.contains('.html')) return title;
    final uri = Uri.tryParse(url);
    if (uri == null || uri.pathSegments.isEmpty) return title;
    return uri.pathSegments.last
        .replaceAll('.html', '')
        .replaceAll('.htm', '')
        .replaceAll('-', ' ')
        .replaceAll('_', ' ');
  }

  @override
  Widget build(BuildContext context) {
    final displayTitle = _sanitiseTitle(article.title, article.url);
    final biasColor = getBiasColor(article.biasScore);
    final sentimentColor = getSentimentColor(article.sentimentScore);
    final credibilityColor = getCredibilityColor(article.credibilityScore);

    return Card(
      elevation: 2,
      margin: const EdgeInsets.only(bottom: 12),
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Title
              Text(
                displayTitle,
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 15,
                ),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 6),

              // Source + date row
              Row(
                children: [
                  Icon(Icons.source, size: 14, color: Colors.grey[600]),
                  const SizedBox(width: 4),
                  Expanded(
                    child: Text(
                      article.source,
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey[700],
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    _formatDate(article.publishedAt),
                    style: TextStyle(
                      fontSize: 11,
                      color: Colors.grey[600],
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 4),

              // Category label
              if (article.category != null && article.category!.isNotEmpty)
                Text(
                  formatCategory(article.category),
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w500,
                    color: Colors.blueGrey[700],
                  ),
                ),

              const SizedBox(height: 12),

              // Bias + sentiment + credibility chips
              Wrap(
                spacing: 8,
                runSpacing: 6,
                children: [
                  if (article.biasScore != null)
                    _ScorePill(
                      label: getBiasLabelShort(article.biasScore),
                      color: biasColor,
                      icon: null,
                    ),
                  if (article.sentimentScore != null)
                    _ScorePill(
                      label: getSentimentLabel(article.sentimentScore),
                      color: sentimentColor,
                      icon: (article.sentimentScore ?? 0) > 0
                          ? Icons.sentiment_satisfied
                          : Icons.sentiment_dissatisfied,
                    ),
                  if (article.credibilityScore != null)
                    _ScorePill(
                      label:
                          '${article.credibilityScore!.round()}% credible',
                      color: credibilityColor,
                      icon: Icons.fact_check_outlined,
                    ),
                  if (article.biasIntensity != null)
                    _ScorePill(
                      label:
                          '${(article.biasIntensity! * 100).round()}% biased',
                      color: Colors.grey,
                      icon: null,
                    ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Internal pill badge used by ArticleCard only.
class _ScorePill extends StatelessWidget {
  final String label;
  final Color color;
  final IconData? icon;

  const _ScorePill({
    required this.label,
    required this.color,
    required this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withAlpha((255 * 0.15).round()),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 12, color: color),
            const SizedBox(width: 4),
          ],
          Text(
            label,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.bold,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}
