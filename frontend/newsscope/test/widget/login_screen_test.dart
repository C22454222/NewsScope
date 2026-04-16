import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:newsscope/screens/sign_in_screen.dart';

void main() {
  testWidgets('SignInScreen shows email and password fields', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: SignInScreen()));
    await tester.pump();
    expect(find.byType(TextFormField), findsNWidgets(2));
  });

  testWidgets('SignInScreen has sign-in button', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: SignInScreen()));
    await tester.pump();
    expect(find.widgetWithText(ElevatedButton, 'Sign In'), findsOneWidget);
  });
}