import 'package:flutter/material.dart';
import '../utils/score_helpers.dart';

/// Standalone bias label chip for list tiles and detail views.
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
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: color.withAlpha((255 * 0.12).round()),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: color.withAlpha(180)),
          ),
          child: Text(
            label,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: color,
            ),
          ),
        ),
        if (biasIntensity != null)
          Text(
            '${(biasIntensity! * 100).round()}% biased',
            style: TextStyle(fontSize: 12, color: Colors.grey[500]),
          ),
      ],
    );
  }
}
