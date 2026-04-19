// lib/screens/auth_gate.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../services/api_service.dart';
import 'home_screen.dart';
import 'sign_in_screen.dart';

/// Routes the user to [HomeScreen] or [SignInScreen] based on Firebase auth state.
///
/// Calls POST /users on every login to upsert the user record in Supabase.
/// This must complete before reading history can be written because
/// reading_history.user_id has a FK constraint to users.id.
///
/// A static [_registeredUids] set persists across widget rebuilds and multiple
/// authStateChanges events (token refreshes, profile updates) within the same
/// app session. Without static, a widget rebuild would recreate [_AuthGateState]
/// and clear the set, causing a redundant POST /users on every rebuild.
class AuthGate extends StatefulWidget {
  const AuthGate({super.key});

  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  final ApiService _apiService = ApiService();

  // Persists across _AuthGateState rebuilds within the same app process.
  // Cleared on sign-out when snapshot.data is null.
  static final Set<String> _registeredUids = {};

  Future<void> _ensureUserRegistered(User user) async {
    // Guard: already registered this UID in this app session.
    if (_registeredUids.contains(user.uid)) return;

    // Log a warning in debug builds when a Google user registers, as
    // signing in after clearing app data can generate a new UID for the
    // same Google account. The backend must handle deduplication on email.
    assert(() {
      final isGoogle =
          user.providerData.any((p) => p.providerId == 'google.com');
      if (isGoogle) {
        debugPrint(
          '[AuthGate] Google user registering -- UID: \${user.uid}, '
          'email: \${user.email}. If duplicates appear in Supabase, '
          'the backend must deduplicate on email.',
        );
      }
      return true;
    }());

    // Mark before the await to prevent races on rapid auth events.
    _registeredUids.add(user.uid);

    try {
      await _apiService.registerUser(
        uid: user.uid,
        email: user.email ?? '',
        displayName: user.displayName,
      );
      debugPrint('[AuthGate] User registered/verified: \${user.uid}');
    } catch (e) {
      // Remove so a retry is possible on the next auth event.
      _registeredUids.remove(user.uid);
      debugPrint('[AuthGate] Failed to register user: \$e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      // authStateChanges fires only on sign-in and sign-out, not on profile
      // updates, which avoids re-triggering _ensureUserRegistered needlessly.
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          // Prevents a flash of SignInScreen while Firebase restores
          // the persisted auth session on cold start.
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }

        if (snapshot.hasData) {
          _ensureUserRegistered(snapshot.data!);
          return const HomeScreen();
        }

        // User signed out; clear so the next sign-in triggers registration.
        _registeredUids.clear();
        return const SignInScreen();
      },
    );
  }
}
