// lib/main.dart
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';

import 'screens/auth_gate.dart';
import 'screens/settings_screen.dart'; // for AppTheme and AppNotifications
import 'core/app_prefs.dart';

// ── FCM background message handler ────────────────────────────────────────────
// Must be a top-level function (not a class method). Flutter runs this in a
// separate isolate on Android when the app is terminated or in the background.
@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  // Firebase must be initialised in the background isolate too.
  await Firebase.initializeApp();
  debugPrint('Background FCM message: ${message.messageId}');
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  if (kReleaseMode) {
    debugPrint = (String? message, {int? wrapWidth}) {};
  }

  await Firebase.initializeApp();

  // Register the background message handler BEFORE any other FCM calls.
  FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);

  // Initialise the local notifications plugin and create the Android
  // channel. Must run before any notification can be shown — on Android 8+
  // notifications without a channel are silently dropped.
  await AppNotifications.init();

  // Load the persisted theme preference before the first frame is painted.
  await AppTheme.load();
  await AppPrefs.load();

  runApp(const NewsScopeApp());
}

class NewsScopeApp extends StatelessWidget {
  const NewsScopeApp({super.key});

  @override
  Widget build(BuildContext context) {
    // ValueListenableBuilder rebuilds MaterialApp whenever AppTheme.notifier
    // changes, giving instant dark/light switching without a full restart.
    return ValueListenableBuilder<ThemeMode>(
      valueListenable: AppTheme.notifier,
      builder: (_, themeMode, _) {
        return MaterialApp(
          title: 'NewsScope',
          debugShowCheckedModeBanner: false,

          // ── Light theme ──────────────────────────────────────────────────
          theme: ThemeData(
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.blue,
              brightness: Brightness.light,
            ),
            scaffoldBackgroundColor: const Color(0xFFF0F2F5),
            cardColor: Colors.white,
            useMaterial3: true,
          ),

          // ── Dark theme ───────────────────────────────────────────────────
          darkTheme: ThemeData(
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.blue,
              brightness: Brightness.dark,
            ),
            scaffoldBackgroundColor: const Color(0xFF121212),
            cardColor: const Color(0xFF1E1E1E),
            useMaterial3: true,
          ),

          themeMode: themeMode,
          home: const AuthGate(),
        );
      },
    );
  }
}
