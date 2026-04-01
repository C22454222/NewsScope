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
///
/// Duplicate-user root cause notes
/// ────────────────────────────────
/// The 3-row duplicate is a Firebase-side issue: signing in with Google
/// after clearing app data (or on a new device) can cause an anonymous
/// session UID to be linked to the Google credential, producing a brand-new
/// UID for the same Google account. `on_conflict="id"` on the backend
/// happily inserts all 3 because the UIDs are genuinely distinct.
///
/// Frontend guard: a *static* set means the guard survives widget
/// disposal/remounting and multiple `authStateChanges` events (token
/// refreshes, profile updates) within the same app session. Without
/// static, a widget rebuild recreates _AuthGateState and clears
/// _lastRegisteredUid, causing a redundant POST /users on every rebuild.
class AuthGate extends StatefulWidget {
  const AuthGate({super.key});

  @override
  State<AuthGate> createState() => _AuthGateState();
}

class _AuthGateState extends State<AuthGate> {
  final ApiService _apiService = ApiService();

  // Static so it persists across _AuthGateState rebuilds/disposals within
  // the same app process. Cleared on sign-out (when snapshot.data == null).
  static final Set<String> _registeredUids = {};

  Future<void> _ensureUserRegistered(User user) async {
    // Guard 1: already registered this UID in this app session.
    if (_registeredUids.contains(user.uid)) return;

    // Guard 2: if the user has a Google provider, check whether the email
    // is already linked to a *different* UID. This can't be detected on the
    // frontend alone — the backend must handle it — but we log it so you can
    // diagnose it in debug builds.
    assert(() {
      final isGoogle =
          user.providerData.any((p) => p.providerId == 'google.com');
      if (isGoogle) {
        debugPrint(
          '[AuthGate] Google user registering — UID: ${user.uid}, '
          'email: ${user.email}. If duplicates appear in Supabase, '
          'the backend must deduplicate on email (on_conflict="id,email" '
          'or a unique email constraint with upsert merge).',
        );
      }
      return true;
    }());

    _registeredUids.add(user.uid); // mark before the await to prevent races

    try {
      await _apiService.registerUser(
        uid: user.uid,
        email: user.email ?? '',
        displayName: user.displayName,
      );
      debugPrint('[AuthGate] User registered/verified: ${user.uid}');
    } catch (e) {
      // Remove from the set so a retry is possible on next auth event.
      _registeredUids.remove(user.uid);
      debugPrint('[AuthGate] Failed to register user: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      // userChanges() is a superset of authStateChanges() — it also fires on
      // displayName/email/photoURL updates. authStateChanges() is correct here
      // because we only want to react to actual sign-in / sign-out events, not
      // profile updates (which would re-trigger _ensureUserRegistered).
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          // Avoid a flash of SignInScreen on cold start while Firebase
          // is restoring the persisted auth session.
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }

        if (snapshot.hasData) {
          _ensureUserRegistered(snapshot.data!);
          return const HomeScreen();
        }

        // User signed out — clear the set so the next sign-in re-registers.
        _registeredUids.clear();
        return const SignInScreen();
      },
    );
  }
}
