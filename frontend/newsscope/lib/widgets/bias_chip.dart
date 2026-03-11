// lib/widgets/bias_chip.dart
import 'package:flutter/material.dart';
import '../utils/score_helpers.dart';

/// Standalone bias label chip for use in list tiles and detail views.
///
/// Used in CompareScreen article lists and any future widget that needs
/// a compact bias indicator without the full ArticleCard layout.
class BiasChip extends StatelessWidget {
  final double? biasScore;
  final double? biasIntensity;

  const BiasChip({
    super.key,
    required this.biasScore,
    this.biasIntensity,
  });

  @override
  Widget build(BuildContext context) {
    final color = getBiasColor(biasScore);
    final label = getBiasLabelShort(biasScore);

    return Wrap(
      spacing: 8,
      crossAxisAlignment: WrapCrossAlignment.center,
      children: [
        Chip(
          label: Text(label),
          backgroundColor: color.withAlpha((255 * 0.2).round()),
          labelStyle: TextStyle(
            fontSize: 10,
            color: color,
            fontWeight: FontWeight.w600,
          ),
          side: BorderSide(color: color.withAlpha((255 * 0.4).round())),
          padding: const EdgeInsets.symmetric(horizontal: 4),
          materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
          visualDensity: VisualDensity.compact,
        ),
        if (biasIntensity != null)
          Text(
            '${(biasIntensity! * 100).round()}% biased',
            style: const TextStyle(fontSize: 12, color: Colors.grey),
          ),
      ],
    );
  }
}
