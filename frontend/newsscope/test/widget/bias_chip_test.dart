import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('Chip widget renders Left text', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(body: Chip(label: Text('Left'))),
      ),
    );
    expect(find.text('Left'), findsOneWidget);
  });

  testWidgets('Chip widget renders Centre text', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(body: Chip(label: Text('Centre'))),
      ),
    );
    expect(find.text('Centre'), findsOneWidget);
  });

  testWidgets('Chip widget renders Right text', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(body: Chip(label: Text('Right'))),
      ),
    );
    expect(find.text('Right'), findsOneWidget);
  });
}