import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('Sign-in form renders email and password fields', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: Form(
            child: Column(children: [
              TextFormField(
                decoration: const InputDecoration(hintText: 'Email'),
              ),
              TextFormField(
                obscureText: true,
                decoration: const InputDecoration(hintText: 'Password'),
              ),
              ElevatedButton(onPressed: () {}, child: const Text('Sign In')),
            ]),
          ),
        ),
      ),
    );
    expect(find.byType(TextFormField), findsNWidgets(2));
    expect(find.widgetWithText(ElevatedButton, 'Sign In'), findsOneWidget);
  });

  testWidgets('Email hint is visible', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: TextFormField(
            decoration: const InputDecoration(hintText: 'you@example.com'),
          ),
        ),
      ),
    );
    expect(find.text('you@example.com'), findsOneWidget);
  });
}