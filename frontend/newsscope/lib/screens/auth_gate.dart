// lib/screens/auth_gate.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../services/api_service.dart';
import 'home_screen.dart';
import 'sign_in_screen.dart';

/// Controls navigation based on Firebase authentication state.
///
/// Calls POST /users on every login to upsert the user record in
/// Supabase. Required before any reading history can be written —
/// reading_history.user_id has a FK constraint to users.id.
/// Upsert is idempotent so repeated calls on rebuild are safe,
/// but the UID guard prevents redundant network calls.
class AuthGate extends StatefulWidget {
  const AuthGate({super.key});

  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  final ApiService _apiService = ApiService();

  // Track the last registered UID to avoid duplicate POST /users calls
  // across StreamBuilder rebuilds for the same authenticated session.
  String? _lastRegisteredUid;

  Future<void> _ensureUserRegistered(User user) async {
    if (_lastRegisteredUid == user.uid) return;
    _lastRegisteredUid = user.uid;
    try {
      await _apiService.registerUser(
        uid: user.uid,
        email: user.email ?? '',
      );
      debugPrint('User registered/verified: ${user.uid}');
    } catch (e) {
      debugPrint('Failed to register user: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        if (snapshot.hasData) {
          _ensureUserRegistered(snapshot.data!);
          return const HomeScreen();
        }
        return const SignInScreen();
      },
    );
  }
}
