import 'package:flutter/material.dart';
import '../models/article.dart';
import '../utils/score_helpers.dart';

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
    final looksLikeUrl = title.startsWith('http');
    final looksLikePath = title.endsWith('.html') || title.endsWith('.htm');
    if (!looksLikeUrl && !looksLikePath) return title;
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

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Theme.of(context).cardColor,
        borderRadius: BorderRadius.circular(12),
        border: Border(left: BorderSide(color: biasColor, width: 4)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.07),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        borderRadius: BorderRadius.circular(12),
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  displayTitle,
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 15,
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Icon(Icons.source, size: 13, color: Colors.grey[500]),
                    const SizedBox(width: 4),
                    Expanded(
                      child: Text(
                        article.source,
                        style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    if (article.category != null && article.category!.isNotEmpty) ...[
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                        decoration: BoxDecoration(
                          color: Colors.blue[50],
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(
                          formatCategory(article.category),
                          style: TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.w600,
                            color: Colors.blue[700],
                          ),
                        ),
                      ),
                    ],
                    const SizedBox(width: 8),
                    Text(
                      _formatDate(article.publishedAt),
                      style: TextStyle(fontSize: 11, color: Colors.grey[500]),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: [
                    if (article.biasScore != null)
                      _ScorePill(
                        label: 'Political Leaning: ${getBiasLabelShort(article.biasScore)}',
                        color: biasColor,
                        icon: Icons.balance,
                      ),
                    if (article.sentimentScore != null)
                      _ScorePill(
                        label: 'Sentiment: ${getSentimentLabel(article.sentimentScore)}',
                        color: sentimentColor,
                        icon: (article.sentimentScore ?? 0) > 0
                            ? Icons.sentiment_satisfied
                            : Icons.sentiment_dissatisfied,
                      ),
                    if (article.credibilityScore != null)
                      _ScorePill(
                        label: '${article.credibilityScore!.round()}% credible',
                        color: credibilityColor,
                        icon: Icons.fact_check_outlined,
                      ),
                    if (article.generalBias != null)
                      _ScorePill(
                        label: article.generalBias == 'BIASED' ? 'Biased' : 'Unbiased',
                        color: article.generalBias == 'BIASED'
                            ? Colors.orange[700]!
                            : Colors.green[700]!,
                        icon: article.generalBias == 'BIASED'
                            ? Icons.warning_amber
                            : Icons.check_circle_outline,
                      ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

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
        color: color.withAlpha((255 * 0.12).round()),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withAlpha(180)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 11, color: color),
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
