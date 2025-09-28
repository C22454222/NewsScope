import 'package:flutter/material.dart';
import 'screens/feed_screen.dart';

void main() {
  runApp(const NewsScopeApp());
}

class NewsScopeApp extends StatelessWidget {
  const NewsScopeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NewsScope',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true, // keeps your theme modern
      ),
      home: const FeedScreen(), // 👈 Replaces the counter demo
    );
  }
}