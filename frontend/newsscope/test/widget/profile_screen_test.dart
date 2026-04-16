import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('Profile placeholder scaffold renders', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: Center(child: Text('No reading history yet')),
        ),
      ),
    );
    expect(find.byType(Scaffold), findsOneWidget);
    expect(find.text('No reading history yet'), findsOneWidget);
  });
}