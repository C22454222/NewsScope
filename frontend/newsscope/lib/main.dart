// lib/main.dart
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'screens/auth_gate.dart';


void main() async {
  // Ensure framework binding is initialized before calling native code (Firebase)
  WidgetsFlutterBinding.ensureInitialized();
  
  // Disable debug output in release mode for cleaner logs
  if (kReleaseMode) {
    debugPrint = (String? message, {int? wrapWidth}) {};
  }
  
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
      debugShowCheckedModeBanner: false,
      home: const AuthGate(),
    );
  }
}
