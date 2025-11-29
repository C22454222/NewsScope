// lib/main.dart
import 'package:flutter/material.dart' show BuildContext, Colors, MaterialApp, StatelessWidget, ThemeData, Widget, WidgetsFlutterBinding, runApp;
import 'package:firebase_core/firebase_core.dart';
import 'auth_gate.dart';

void main() async {
  // Ensure framework binding is initialized before calling native code (Firebase)
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(const NewsScopeApp());
}

class NewsScopeApp extends StatelessWidget {
  const NewsScopeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NewsScope',
      theme: ThemeData(primarySwatch: Colors.blue),
      // Use AuthGate to decide initial screen based on login status
      home: const AuthGate(),
    );
  }
}
