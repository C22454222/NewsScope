import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:newsscope/screens/settings_screen.dart';

void main() {
  testWidgets('SettingsScreen renders dark mode toggle', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: SettingsScreen()));
    await tester.pump();
    expect(find.byType(SwitchListTile), findsWidgets);
  });

  testWidgets('SettingsScreen shows glossary tile', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: SettingsScreen()));
    await tester.pump();
    expect(find.text('Glossary'), findsOneWidget);
  });
}