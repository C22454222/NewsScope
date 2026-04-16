import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('SwitchListTile toggles value', (tester) async {
    bool value = false;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: StatefulBuilder(
            builder: (context, setState) => SwitchListTile(
              title: const Text('Dark Mode'),
              value: value,
              onChanged: (v) => setState(() => value = v),
            ),
          ),
        ),
      ),
    );
    expect(find.byType(SwitchListTile), findsOneWidget);
    expect(find.text('Dark Mode'), findsOneWidget);

    await tester.tap(find.byType(Switch));
    await tester.pump();
    expect(value, true);
  });

  testWidgets('Glossary ListTile renders', (tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: ListTile(
            leading: Icon(Icons.menu_book),
            title: Text('Glossary'),
          ),
        ),
      ),
    );
    expect(find.text('Glossary'), findsOneWidget);
    expect(find.byIcon(Icons.menu_book), findsOneWidget);
  });
}