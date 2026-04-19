import 'dart:io';

import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../core/config.dart';
import '../core/app_prefs.dart';

// Local notifications plugin singleton. Initialised once in
// AppNotifications.init(); safe to call multiple times.
final FlutterLocalNotificationsPlugin _localNotifications =
    FlutterLocalNotificationsPlugin();

/// App-wide notification helper handling local + FCM setup.
class AppNotifications {
  AppNotifications._();

  /// Call once from main() after Firebase.initializeApp().
  static Future<void> init() async {
    // Android channel. On Android 8+ (Oreo) a NotificationChannel must
    // exist before any notification can be shown. Samsung devices honour
    // the same channel API.
    const androidChannel = AndroidNotificationChannel(
      'news_updates',           // id, must match FCM payload channel_id
      'News Updates',           // human-readable name in device settings
      description: 'Breaking news alerts from NewsScope',
      importance: Importance.high,
      playSound: true,
    );

    final androidPlugin = _localNotifications
        .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>();
    await androidPlugin?.createNotificationChannel(androidChannel);

    // Plugin initialisation. iOS permissions are requested via FCM below
    // rather than here to keep the prompt flow consistent.
    const initSettings = InitializationSettings(
      android: AndroidInitializationSettings('@mipmap/ic_launcher'),
      iOS: DarwinInitializationSettings(
        requestAlertPermission: false,
        requestBadgePermission: false,
        requestSoundPermission: false,
      ),
    );
    await _localNotifications.initialize(initSettings);

    // Foreground FCM messages. By default FCM does NOT show a heads-up
    // notification when the app is foregrounded on Android, so we show
    // one manually via flutter_local_notifications.
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      final notification = message.notification;
      if (notification == null) return;

      _localNotifications.show(
        notification.hashCode,
        notification.title,
        notification.body,
        NotificationDetails(
          android: AndroidNotificationDetails(
            'news_updates',
            'News Updates',
            channelDescription: 'Breaking news alerts from NewsScope',
            importance: Importance.high,
            priority: Priority.high,
            icon: '@mipmap/ic_launcher',
            // Ensures the small icon renders correctly on Samsung's status bar.
            styleInformation: const DefaultStyleInformation(true, true),
          ),
          iOS: const DarwinNotificationDetails(),
        ),
      );
    });
  }

  /// Shows a local test notification, used to verify the channel works.
  static Future<void> showTest() async {
    await _localNotifications.show(
      0,
      'NewsScope',
      'Notifications are working! 🎉',
      const NotificationDetails(
        android: AndroidNotificationDetails(
          'news_updates',
          'News Updates',
          importance: Importance.high,
          priority: Priority.high,
          icon: '@mipmap/ic_launcher',
        ),
        iOS: DarwinNotificationDetails(),
      ),
    );
  }
}

/// App-wide theme notifier. Persists the user's light/dark choice to
/// SharedPreferences and exposes a [ValueNotifier] for MaterialApp.
class AppTheme {
  AppTheme._();

  static final ValueNotifier<ThemeMode> notifier =
      ValueNotifier(ThemeMode.light);

  static Future<void> load() async {
    final prefs = await SharedPreferences.getInstance();
    final isDark = prefs.getBool('dark_mode') ?? false;
    notifier.value = isDark ? ThemeMode.dark : ThemeMode.light;
  }

  static Future<void> set(bool dark) async {
    notifier.value = dark ? ThemeMode.dark : ThemeMode.light;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('dark_mode', dark);
  }

  static bool get isDark => notifier.value == ThemeMode.dark;
}

/// Settings screen exposing account, notification, display, data and
/// account-action controls.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  User? get user => FirebaseAuth.instance.currentUser;

  // Persisted preferences, mirrored locally for switch state.
  bool _notificationsEnabled = false;
  bool _compactCards = false;
  bool _showCredibility = true;
  bool _showSentiment = true;
  bool _darkMode = false;

  String? _displayName;

  @override
  void initState() {
    super.initState();
    _displayName = user?.displayName;
    _loadPreferences();
    // Keep local _darkMode in sync with the global notifier so external
    // theme changes (e.g. from another entry point) are reflected here.
    AppTheme.notifier.addListener(_onThemeChanged);
  }

  @override
  void dispose() {
    AppTheme.notifier.removeListener(_onThemeChanged);
    super.dispose();
  }

  void _onThemeChanged() {
    if (!mounted) return;
    setState(() => _darkMode = AppTheme.isDark);
  }

  Future<void> _loadPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      _notificationsEnabled = prefs.getBool('notifications_enabled') ?? false;
      _compactCards = prefs.getBool('compact_cards') ?? false;
      _showCredibility = prefs.getBool('show_credibility') ?? true;
      _showSentiment = prefs.getBool('show_sentiment') ?? true;
      _darkMode = prefs.getBool('dark_mode') ?? false;
    });
  }

  Future<void> _saveBool(String key, bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(key, value);
  }

  // Notifications.

  Future<void> _setNotifications(bool value) async {
    if (value) {
      // Step 1: request OS-level permission. Android 13+ (API 33)
      // treats POST_NOTIFICATIONS as a runtime permission. Older Android
      // and Samsung One UI grant it automatically, but asking is harmless.
      bool osGranted = true;
      if (Platform.isAndroid) {
        final status = await Permission.notification.request();
        osGranted = status.isGranted;
      }

      // Step 2: request FCM permission (required on iOS, no-op on Android).
      final settings = await FirebaseMessaging.instance.requestPermission(
        alert: true, badge: true, sound: true,
      );
      final fcmGranted =
          settings.authorizationStatus == AuthorizationStatus.authorized ||
              settings.authorizationStatus == AuthorizationStatus.provisional;

      if (!osGranted || !fcmGranted) {
        if (!mounted) return;
        // User denied; offer to jump to system settings.
        final openSettings = await showDialog<bool>(
          context: context,
          builder: (context) => AlertDialog(
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16)),
            title: const Text('Notifications Blocked'),
            content: const Text(
              'NewsScope needs notification permission.\n\n'
              'On Samsung devices you may also need to enable '
              '"Allow notifications" inside the app\'s system settings.',
              style: TextStyle(height: 1.5),
            ),
            actions: [
              TextButton(
                  onPressed: () => Navigator.pop(context, false),
                  child: const Text('Cancel')),
              TextButton(
                onPressed: () => Navigator.pop(context, true),
                child: Text('Open Settings',
                    style: TextStyle(color: Colors.blue[700])),
              ),
            ],
          ),
        );
        if (openSettings == true) await openAppSettings();
        return; // Leave the switch off.
      }

      // Step 3: subscribe to the FCM topic used by the backend.
      await FirebaseMessaging.instance.subscribeToTopic('news_updates');

      // Step 4: send a test so the user can see notifications work.
      await AppNotifications.showTest();
    } else {
      await FirebaseMessaging.instance.unsubscribeFromTopic('news_updates');
    }

    await _saveBool('notifications_enabled', value);
    if (!mounted) return;
    setState(() => _notificationsEnabled = value);
    _showSnackBar(
      value ? 'Notifications enabled' : 'Notifications disabled',
      color: Colors.green[700],
    );
  }

  // Theme toggle.

  Future<void> _setDarkMode(bool value) async {
    // Updates the notifier and SharedPreferences; _onThemeChanged()
    // takes care of calling setState for _darkMode.
    await AppTheme.set(value);
    _showSnackBar(
      value ? 'Dark mode enabled' : 'Light mode enabled',
      color: Colors.green[700],
    );
  }

  // Clear reading history.

  Future<void> _handleClearHistory() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16)),
        title: const Text('Clear Reading History'),
        content: const Text(
          'This will permanently delete all your reading history and reset '
          'your Bias Profile to zero. Your account will not be deleted.\n\n'
          'This cannot be undone.',
          style: TextStyle(height: 1.5),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style:
                TextButton.styleFrom(foregroundColor: Colors.red[700]),
            child: const Text('Clear History',
                style: TextStyle(fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    try {
      final idToken = await user?.getIdToken();
      final uid = user?.uid;
      if (idToken != null && uid != null) {
        await http.delete(
          Uri.parse('${AppConfig.baseUrl}/users/$uid/history'),
          headers: {'Authorization': 'Bearer $idToken'},
        );
      }
      if (!mounted) return;
      _showSnackBar('Reading history cleared', color: Colors.green[700]);
    } catch (e) {
      if (!mounted) return;
      _showSnackBar('Failed to clear history: $e', color: Colors.red);
    }
  }

  // Display name.

  Future<void> _handleEditDisplayName() async {
    final currentName = (_displayName ?? '').trim();
    final result = await showDialog<String>(
      context: context,
      barrierDismissible: true,
      builder: (_) =>
          _EditDisplayNameDialog(initialValue: currentName),
    );
    if (!mounted || result == null) return;
    final newName = result.trim();
    if (newName.isEmpty || newName == currentName) return;
    try {
      await user?.updateDisplayName(newName);
      await FirebaseAuth.instance.currentUser?.reload();
      final refreshedName =
          FirebaseAuth.instance.currentUser?.displayName ?? newName;
      if (!mounted) return;
      setState(() => _displayName = refreshedName.trim());
      _showSnackBar('Display name updated', color: Colors.green[700]);
    } catch (e) {
      if (!mounted) return;
      _showSnackBar('Failed to update name: $e', color: Colors.red);
    }
  }

  // Change password (email/password users only).

  Future<void> _handleChangePassword() async {
    final email = user?.email;
    if (email == null) return;
    try {
      await FirebaseAuth.instance.sendPasswordResetEmail(email: email);
      if (!mounted) return;
      _showSnackBar('Password reset email sent to $email',
          color: Colors.green[700]);
    } catch (e) {
      if (!mounted) return;
      _showSnackBar('Failed to send reset email: $e',
          color: Colors.red);
    }
  }

  // Logout.

  Future<void> _handleLogout() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16)),
        title: const Text('Sign Out'),
        content: const Text('Are you sure you want to sign out?'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style:
                TextButton.styleFrom(foregroundColor: Colors.red[700]),
            child: const Text('Sign Out'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    await FirebaseAuth.instance.signOut();
    if (!mounted) return;
    Navigator.of(context).popUntil((route) => route.isFirst);
  }

  // Delete account.

  Future<void> _handleDeleteAccount() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16)),
        title: const Text('Delete Account',
            style: TextStyle(color: Colors.red)),
        content: const Text(
          'This will permanently delete your account and all reading '
          'history. This cannot be undone.',
          style: TextStyle(height: 1.5),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete',
                style: TextStyle(
                    color: Colors.red,
                    fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    final uid = user?.uid;
    if (uid == null) return;
    await _attemptDeleteAccount(uid);
  }

  /// Performs the account deletion flow. Firebase requires a recent
  /// re-auth for deletion; we re-authenticate and retry on
  /// `requires-recent-login`.
  Future<void> _attemptDeleteAccount(String uid) async {
    try {
      final isGoogleUser =
          user?.providerData.any((p) => p.providerId == 'google.com') ??
              false;
      if (isGoogleUser) {
        final success = await _reauthWithGoogle();
        if (!success) return;
      }
      final idToken = await user?.getIdToken();
      await user?.delete();
      // Tell the backend to purge user data too.
      if (idToken != null) {
        await http.delete(
          Uri.parse('${AppConfig.baseUrl}/users/$uid'),
          headers: {'Authorization': 'Bearer $idToken'},
        );
      }
      if (!mounted) return;
      Navigator.of(context).popUntil((route) => route.isFirst);
    } on FirebaseAuthException catch (e) {
      if (!mounted) return;
      if (e.code == 'requires-recent-login') {
        final isGoogleUser =
            user?.providerData.any((p) => p.providerId == 'google.com') ??
                false;
        if (isGoogleUser) {
          final success = await _reauthWithGoogle();
          if (!success || !mounted) return;
        } else {
          final success = await _reauthWithPassword();
          if (!success || !mounted) return;
        }
        await _attemptDeleteAccount(uid);
      } else {
        _showSnackBar('Failed to delete account: ${e.message}',
            color: Colors.red);
      }
    } catch (e) {
      if (!mounted) return;
      _showSnackBar('Failed to delete account: $e', color: Colors.red);
    }
  }

  Future<bool> _reauthWithGoogle() async {
    try {
      final googleSignIn = GoogleSignIn();
      final googleUser = await googleSignIn.signIn();
      if (googleUser == null) return false;
      final googleAuth = await googleUser.authentication;
      final credential = GoogleAuthProvider.credential(
          idToken: googleAuth.idToken,
          accessToken: googleAuth.accessToken);
      await user?.reauthenticateWithCredential(credential);
      return true;
    } catch (_) {
      if (mounted) {
        _showSnackBar('Google re-authentication failed.',
            color: Colors.red);
      }
      return false;
    }
  }

  Future<bool> _reauthWithPassword() async {
    if (!mounted) return false;
    final password = await showDialog<String>(
      context: context,
      barrierDismissible: false,
      builder: (_) =>
          _ReauthPasswordDialog(email: user?.email ?? ''),
    );
    if (password == null || password.isEmpty) return false;
    try {
      final credential = EmailAuthProvider.credential(
          email: user!.email!, password: password);
      await user?.reauthenticateWithCredential(credential);
      return true;
    } on FirebaseAuthException catch (e) {
      if (mounted) {
        _showSnackBar(
          e.code == 'wrong-password'
              ? 'Incorrect password.'
              : 'Re-authentication failed: ${e.message}',
          color: Colors.red,
        );
      }
      return false;
    }
  }

  // Privacy policy bottom sheet.

  void _showPrivacyPolicy() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
          borderRadius:
              BorderRadius.vertical(top: Radius.circular(20))),
      builder: (context) => DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.7,
        maxChildSize: 0.95,
        builder: (_, scrollController) => Column(
          children: [
            const SizedBox(height: 12),
            Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                    color: Colors.grey[300],
                    borderRadius: BorderRadius.circular(2))),
            Padding(
              padding:
                  const EdgeInsets.fromLTRB(20, 16, 20, 8),
              child: Row(children: [
                Icon(Icons.privacy_tip_outlined,
                    color: Colors.blue[700]),
                const SizedBox(width: 10),
                Text('Privacy Policy & GDPR',
                    style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.grey[800])),
              ]),
            ),
            const Divider(),
            Expanded(
              child: ListView(
                controller: scrollController,
                padding:
                    const EdgeInsets.fromLTRB(20, 8, 20, 40),
                children: [
                  _privacySection('Overview',
                      'NewsScope ("we", "us", "our") is a news analysis application developed as a Final Year Project at Technological University Dublin. We are committed to protecting your personal data in compliance with GDPR (EU) 2016/679.'),
                  _privacySection('Data Controller',
                      'The data controller is the developer at Technological University Dublin. For data protection queries, contact the developer through the University.'),
                  _privacySection('What Personal Data We Collect',
                      'Email address, display name, article reading history, and Firebase UID. We do not collect payment information or precise location data.'),
                  _privacySection('Your Rights Under GDPR',
                      'You have the right to access, rectify, erase, restrict, and port your data. To exercise these rights or withdraw consent for notifications, use Settings or contact the developer.'),
                  _privacySection('Data Sharing',
                      'We use Firebase (Google LLC) and Supabase. We do not sell, rent, or trade your personal data to any third party.'),
                  _privacySection('Data Retention',
                      'Your data is retained while your account is active. Delete your account via Settings → Account Actions → Delete Account.'),
                  const SizedBox(height: 12),
                  Text(
                    'Last updated: March 2026 · NewsScope v1.0.0 Beta\n'
                    'Technological University Dublin — Final Year Project',
                    style: TextStyle(
                        fontSize: 11, color: Colors.grey[400]),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _privacySection(String title, String body) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title,
                style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: Colors.grey[800])),
            const SizedBox(height: 6),
            Text(body,
                style: TextStyle(
                    fontSize: 13,
                    color: Colors.grey[600],
                    height: 1.5)),
          ]),
    );
  }

  // Snackbar helper.

  void _showSnackBar(String message, {Color? color}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(message),
      backgroundColor: color,
      behavior: SnackBarBehavior.floating,
      duration: const Duration(seconds: 3),
    ));
  }

  // Widget builders.

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 24, 16, 8),
      child: Text(title.toUpperCase(),
          style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.bold,
              color: Colors.blue[700],
              letterSpacing: 1.2)),
    );
  }

  Widget _buildTile({
    required IconData icon,
    required String title,
    String? subtitle,
    Widget? trailing,
    VoidCallback? onTap,
    Color? iconColor,
    Color? titleColor,
  }) {
    return ListTile(
      leading: CircleAvatar(
        radius: 18,
        backgroundColor:
            (iconColor ?? Colors.blue[700]!).withAlpha(25),
        child: Icon(icon,
            size: 18, color: iconColor ?? Colors.blue[700]),
      ),
      title: Text(title,
          style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: titleColor)),
      subtitle: subtitle != null
          ? Text(subtitle,
              style: TextStyle(
                  fontSize: 12, color: Colors.grey[500]))
          : null,
      trailing: trailing ??
          (onTap != null
              ? Icon(Icons.chevron_right,
                  color: Colors.grey[400], size: 20)
              : null),
      onTap: onTap,
    );
  }

  Widget _buildSwitch({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool value,
    required ValueChanged<bool> onChanged,
    Color? iconColor,
  }) {
    return SwitchListTile(
      secondary: CircleAvatar(
        radius: 18,
        backgroundColor:
            (iconColor ?? Colors.blue[700]!).withAlpha(25),
        child: Icon(icon,
            size: 18, color: iconColor ?? Colors.blue[700]),
      ),
      title: Text(title,
          style: const TextStyle(
              fontSize: 14, fontWeight: FontWeight.w500)),
      subtitle: Text(subtitle,
          style:
              TextStyle(fontSize: 12, color: Colors.grey[500])),
      activeThumbColor: Colors.blue[700],
      activeTrackColor: Colors.blue[300],
      value: value,
      onChanged: onChanged,
    );
  }

  // Glossary data used by the expandable "Glossary of Terms" card.
  static const _glossaryTerms = [
    (
      term: 'Political Leaning',
      icon: Icons.balance,
      color: Color(0xFF1565C0),
      definition:
          'Whether an article\'s language aligns with left-wing, centre, or right-wing political perspectives, as classified by a machine learning model trained on political text.'
    ),
    (
      term: 'Left Wing',
      icon: Icons.arrow_back,
      color: Color(0xFF1565C0),
      definition:
          'Content favouring progressive values such as social equality, expanded public services, workers\' rights, environmental regulation, and government intervention in the economy.'
    ),
    (
      term: 'Centre',
      icon: Icons.remove,
      color: Color(0xFF00796B),
      definition:
          'Content that does not lean strongly in either direction, typically presenting multiple viewpoints without strongly favouring one side of the political spectrum.'
    ),
    (
      term: 'Right Wing',
      icon: Icons.arrow_forward,
      color: Color(0xFFC62828),
      definition:
          'Content favouring conservative values such as free markets, lower taxation, traditional institutions, national sovereignty, and limited government intervention.'
    ),
    (
      term: 'Sentiment',
      icon: Icons.sentiment_satisfied,
      color: Color(0xFF388E3C),
      definition:
          'The overall emotional tone of an article. Positive means optimistic or favourable language. Negative means critical or alarming language. Neutral falls in between.'
    ),
    (
      term: 'Credibility Score',
      icon: Icons.fact_check_outlined,
      color: Color(0xFF388E3C),
      definition:
          'A 0–100 score estimating how reliable an article is. Above 70 = Reliable. 40–70 = Mixed. Below 40 = Low.'
    ),
    (
      term: 'Bias Profile',
      icon: Icons.pie_chart,
      color: Color(0xFF1565C0),
      definition:
          'Your personal reading summary showing the distribution of political leanings, sentiment, and sources across every article you have read.'
    ),
    (
      term: 'Story Comparison',
      icon: Icons.compare_arrows,
      color: Color(0xFF00796B),
      definition:
          'The Compare tab groups articles on the same topic by political leaning (Left, Centre, Right), letting you see how different outlets frame the same story.'
    ),
    (
      term: 'LIME Explanation',
      icon: Icons.psychology_outlined,
      color: Color(0xFF6A1B9A),
      definition:
          'LIME (Local Interpretable Model-Agnostic Explanations) shows which specific words most strongly influenced the political bias classification for an article.'
    ),
  ];

  Widget _buildGlossarySection() {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      elevation: 1,
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12)),
      child: ExpansionTile(
        leading: CircleAvatar(
          radius: 18,
          backgroundColor: Colors.blue[700]!.withAlpha(25),
          child: Icon(Icons.menu_book_outlined,
              size: 18, color: Colors.blue[700]),
        ),
        title: Text('Glossary of Terms',
            style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: Colors.grey[800])),
        subtitle: Text('${_glossaryTerms.length} terms explained',
            style:
                TextStyle(fontSize: 12, color: Colors.grey[500])),
        childrenPadding:
            const EdgeInsets.fromLTRB(16, 0, 16, 16),
        children: _glossaryTerms.map((t) {
          return Padding(
            padding: const EdgeInsets.only(top: 14),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                      color: t.color.withAlpha(25),
                      borderRadius: BorderRadius.circular(8)),
                  child:
                      Icon(t.icon, size: 18, color: t.color),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment:
                        CrossAxisAlignment.start,
                    children: [
                      Text(t.term,
                          style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.bold,
                              color: t.color)),
                      const SizedBox(height: 3),
                      Text(t.definition,
                          style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[600],
                              height: 1.45)),
                    ],
                  ),
                ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  // Build.

  @override
  Widget build(BuildContext context) {
    // Google-auth users don't have a password, so the change-password tile
    // and password reauth dialog are skipped for them.
    final isGoogleUser =
        user?.providerData.any((p) => p.providerId == 'google.com') ??
            false;

    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: Text('Settings',
            style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.blue[800])),
      ),
      body: ListView(
        children: [
          // Account section.
          _buildSectionHeader('Account'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12)),
            child: Column(children: [
              _buildTile(
                icon: Icons.person_outline,
                title: 'Display Name',
                subtitle: (_displayName?.isNotEmpty == true)
                    ? _displayName!
                    : 'Tap to set',
                onTap: _handleEditDisplayName,
              ),
              const Divider(height: 1, indent: 56),
              _buildTile(
                icon: Icons.email_outlined,
                title: 'Email',
                subtitle: user?.email ?? 'Not signed in',
                trailing: const SizedBox.shrink(),
              ),
              if (!isGoogleUser) ...[
                const Divider(height: 1, indent: 56),
                _buildTile(
                  icon: Icons.lock_reset_outlined,
                  title: 'Change Password',
                  subtitle: 'Send a reset link to your email',
                  onTap: _handleChangePassword,
                ),
              ],
              const Divider(height: 1, indent: 56),
              _buildTile(
                icon: Icons.badge_outlined,
                title: 'Account Type',
                subtitle: isGoogleUser
                    ? 'Google Sign-In'
                    : 'Email & Password',
                trailing: const SizedBox.shrink(),
              ),
            ]),
          ),

          // Notifications section.
          _buildSectionHeader('Notifications'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12)),
            child: Column(children: [
              _buildSwitch(
                icon: Icons.notifications_outlined,
                title: 'Breaking News Alerts',
                subtitle: _notificationsEnabled
                    ? 'Tap to disable push notifications'
                    : 'Tap to enable push notifications',
                value: _notificationsEnabled,
                onChanged: _setNotifications,
              ),
              if (_notificationsEnabled) ...[
                const Divider(height: 1, indent: 56),
                // Shortcut for Samsung users who may need to manage
                // channels, sounds and importance in system settings.
                _buildTile(
                  icon: Icons.settings_outlined,
                  title: 'Notification Settings',
                  subtitle:
                      'Manage channels, sounds & importance in system settings',
                  iconColor: Colors.orange[700],
                  onTap: openAppSettings,
                ),
              ],
            ]),
          ),

          // Display preferences section.
          _buildSectionHeader('Display Preferences'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12)),
            child: Column(children: [
              _buildSwitch(
                icon: _darkMode
                    ? Icons.dark_mode_outlined
                    : Icons.light_mode_outlined,
                title: 'Dark Mode',
                subtitle: _darkMode
                    ? 'Switch to light theme'
                    : 'Switch to dark theme',
                value: _darkMode,
                onChanged: _setDarkMode,
                iconColor: _darkMode
                    ? Colors.indigo[700]
                    : Colors.orange[700],
              ),
              const Divider(height: 1, indent: 56),
              _buildSwitch(
                icon: Icons.fact_check_outlined,
                title: 'Show Credibility Score',
                subtitle: 'Display credibility badge on article cards',
                value: _showCredibility,
                onChanged: (v) async {
                  setState(() => _showCredibility = v);
                  await _saveBool('show_credibility', v);
                  // Reload AppPrefs so other screens see the change.
                  await AppPrefs.load();
                },
                iconColor: Colors.green[700],
              ),
              const Divider(height: 1, indent: 56),
              _buildSwitch(
                icon: Icons.sentiment_satisfied_outlined,
                title: 'Show Sentiment',
                subtitle: 'Display sentiment badge on article cards',
                value: _showSentiment,
                onChanged: (v) async {
                  setState(() => _showSentiment = v);
                  await _saveBool('show_sentiment', v);
                  await AppPrefs.load();
                },
                iconColor: Colors.teal[600],
              ),
              const Divider(height: 1, indent: 56),
              _buildSwitch(
                icon: Icons.view_agenda_outlined,
                title: 'Compact Article Cards',
                subtitle: 'Show smaller cards in the news feed',
                value: _compactCards,
                onChanged: (v) async {
                  setState(() => _compactCards = v);
                  await _saveBool('compact_cards', v);
                  await AppPrefs.load();
                },
                iconColor: Colors.indigo[600],
              ),
            ]),
          ),

          // Data & privacy section.
          _buildSectionHeader('Data & Privacy'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12)),
            child: Column(children: [
              _buildTile(
                icon: Icons.delete_sweep_outlined,
                title: 'Clear Reading History',
                subtitle:
                    'Reset your Bias Profile and all reading data',
                iconColor: Colors.orange[700],
                onTap: _handleClearHistory,
              ),
              const Divider(height: 1, indent: 56),
              _buildTile(
                icon: Icons.privacy_tip_outlined,
                title: 'Privacy Policy & GDPR',
                subtitle:
                    'How we collect, use, and protect your data',
                trailing: Icon(Icons.open_in_new,
                    size: 16, color: Colors.grey[400]),
                onTap: _showPrivacyPolicy,
              ),
            ]),
          ),

          // About section.
          _buildSectionHeader('About'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12)),
            child: Column(children: [
              _buildTile(
                icon: Icons.info_outline,
                title: 'App Version',
                subtitle: '1.0.0 Beta — Final Year Project',
                trailing: const SizedBox.shrink(),
              ),
              const Divider(height: 1, indent: 56),
              _buildTile(
                icon: Icons.school_outlined,
                title: 'Institution',
                subtitle: 'Technological University Dublin',
                trailing: const SizedBox.shrink(),
              ),
            ]),
          ),

          // Glossary section.
          _buildSectionHeader('Glossary'),
          _buildGlossarySection(),

          // Account actions section.
          _buildSectionHeader('Account Actions'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12)),
            child: Column(children: [
              _buildTile(
                icon: Icons.logout,
                title: 'Sign Out',
                iconColor: Colors.red[600],
                titleColor: Colors.red[600],
                trailing: const SizedBox.shrink(),
                onTap: _handleLogout,
              ),
              const Divider(height: 1, indent: 56),
              _buildTile(
                icon: Icons.delete_outline,
                title: 'Delete Account',
                subtitle:
                    'Permanently remove your account and all data',
                iconColor: Colors.red[800],
                titleColor: Colors.red[800],
                trailing: const SizedBox.shrink(),
                onTap: _handleDeleteAccount,
              ),
            ]),
          ),

          const SizedBox(height: 32),
          Center(
            child: Text(
              'NewsScope v1.0.0 Beta · TU Dublin · 2026',
              style:
                  TextStyle(fontSize: 11, color: Colors.grey[400]),
            ),
          ),
          const SizedBox(height: 24),
        ],
      ),
    );
  }
}

// Re-auth password dialog used before sensitive actions (delete account).
class _ReauthPasswordDialog extends StatefulWidget {
  final String email;
  const _ReauthPasswordDialog({required this.email});
  @override
  State<_ReauthPasswordDialog> createState() =>
      _ReauthPasswordDialogState();
}

class _ReauthPasswordDialogState
    extends State<_ReauthPasswordDialog> {
  late final TextEditingController _controller;
  bool _obscure = true;
  @override
  void initState() {
    super.initState();
    _controller = TextEditingController();
  }
  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16)),
      title: const Text('Confirm Your Password'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Enter your password for ${widget.email} to confirm deletion.',
            style: TextStyle(
                fontSize: 13,
                color: Colors.grey[600],
                height: 1.4),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _controller,
            obscureText: _obscure,
            textInputAction: TextInputAction.done,
            onSubmitted: (_) =>
                Navigator.of(context).pop(_controller.text.trim()),
            decoration: InputDecoration(
              labelText: 'Password',
              border: const OutlineInputBorder(),
              suffixIcon: IconButton(
                icon: Icon(_obscure
                    ? Icons.visibility_off
                    : Icons.visibility),
                onPressed: () =>
                    setState(() => _obscure = !_obscure),
              ),
            ),
          ),
        ],
      ),
      actions: [
        TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel')),
        TextButton(
          onPressed: () =>
              Navigator.of(context).pop(_controller.text.trim()),
          style: TextButton.styleFrom(
              foregroundColor: Colors.red[700]),
          child: const Text('Confirm Delete',
              style: TextStyle(fontWeight: FontWeight.bold)),
        ),
      ],
    );
  }
}

// Edit display name dialog. Focuses the field and selects all text on
// open so the user can overwrite their existing name quickly.
class _EditDisplayNameDialog extends StatefulWidget {
  final String initialValue;
  const _EditDisplayNameDialog({required this.initialValue});
  @override
  State<_EditDisplayNameDialog> createState() =>
      _EditDisplayNameDialogState();
}

class _EditDisplayNameDialogState
    extends State<_EditDisplayNameDialog> {
  late final TextEditingController _controller;
  late final FocusNode _focusNode;
  @override
  void initState() {
    super.initState();
    _controller =
        TextEditingController(text: widget.initialValue);
    _focusNode = FocusNode();
    // Focus and select-all after the first frame so the dialog is
    // fully built before we request focus.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      _focusNode.requestFocus();
      _controller.selection = TextSelection(
          baseOffset: 0,
          extentOffset: _controller.text.length);
    });
  }
  @override
  void dispose() {
    _focusNode.dispose();
    _controller.dispose();
    super.dispose();
  }
  void _save() {
    FocusScope.of(context).unfocus();
    Navigator.of(context).pop(_controller.text.trim());
  }
  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16)),
      title: const Text('Edit Display Name'),
      content: TextField(
        controller: _controller,
        focusNode: _focusNode,
        maxLength: 40,
        textInputAction: TextInputAction.done,
        textCapitalization: TextCapitalization.words,
        onSubmitted: (_) => _save(),
        onTapOutside: (_) => _focusNode.unfocus(),
        decoration: const InputDecoration(
            labelText: 'Display name',
            border: OutlineInputBorder()),
      ),
      actions: [
        TextButton(
          onPressed: () {
            FocusScope.of(context).unfocus();
            Navigator.of(context).pop();
          },
          child: const Text('Cancel'),
        ),
        TextButton(onPressed: _save, child: const Text('Save')),
      ],
    );
  }
}
