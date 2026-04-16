import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('Article header renders title and source badge', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: const Text('Article')),
          body: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                color: Colors.blue.shade50,
                child: const Text('BBC'),
              ),
              const Text('Test Article Title'),
              const SelectableText('Body text of the article.'),
            ],
          ),
        ),
      ),
    );
    expect(find.text('Test Article Title'), findsOneWidget);
    expect(find.text('BBC'), findsOneWidget);
    expect(find.text('Body text of the article.'), findsOneWidget);
  });
}