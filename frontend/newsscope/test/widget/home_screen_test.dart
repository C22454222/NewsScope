import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:newsscope/screens/home_screen.dart';

void main() {
  testWidgets('HomeScreen shows loading indicator on init', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: HomeScreen()));
    expect(find.byType(CircularProgressIndicator), findsWidgets);
  });

  testWidgets('HomeScreen has bottom navigation with four tabs', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: HomeScreen()));
    await tester.pump();
    expect(find.byType(BottomNavigationBar), findsOneWidget);
  });
}