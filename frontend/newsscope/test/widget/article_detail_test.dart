import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:newsscope/screens/article_detail_screen.dart';

void main() {
  testWidgets('ArticleDetailScreen renders title and source', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: ArticleDetailScreen(
          id: '1',
          title: 'Test Article',
          sourceName: 'BBC',
          url: 'https://example.com',
          content: 'body text',
          biasScore: 0.0,
          sentimentScore: 0.1,
          credibilityScore: 85,
        ),
      ),
    );
    await tester.pump();
    expect(find.text('Test Article'), findsOneWidget);
    expect(find.text('BBC'), findsOneWidget);
  });
}
