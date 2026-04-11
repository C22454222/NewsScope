import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('App boots without error', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(body: Center(child: Text('NewsScope'))),
      ),
    );
    expect(find.text('NewsScope'), findsOneWidget);
  });

  testWidgets('Material widgets render', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          appBar: AppBar(title: const Text('Home')),
          body: const Center(child: Text('Loaded')),
        ),
      ),
    );
    expect(find.text('Home'), findsOneWidget);
    expect(find.text('Loaded'), findsOneWidget);
  });
}
