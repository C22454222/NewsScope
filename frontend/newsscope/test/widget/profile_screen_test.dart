import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:newsscope/screens/profile_screen.dart';

void main() {
  testWidgets('ProfileScreen renders placeholder when no history', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: ProfileScreen()));
    await tester.pump();
    expect(find.byType(Scaffold), findsOneWidget);
  });
}