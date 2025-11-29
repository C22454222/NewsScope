// lib/screens/placeholders.dart
import 'package:flutter/material.dart';

/// Placeholder screen for future comparison features.
/// Will allow users to view articles side-by-side.
class CompareScreen extends StatelessWidget {
  const CompareScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Compare Coverage")),
      body: const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.compare_arrows, size: 64, color: Colors.grey),
            SizedBox(height: 16),
            Text("Comparison tools coming soon!"),
          ],
        ),
      ),
    );
  }
}

/// Placeholder screen for user profile settings.
/// Will include bias tracking and reading history preferences.
class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("My Profile")),
      body: const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.person_outline, size: 64, color: Colors.grey),
            SizedBox(height: 16),
            Text("User preferences and bias profile."),
          ],
        ),
      ),
    );
  }
}
