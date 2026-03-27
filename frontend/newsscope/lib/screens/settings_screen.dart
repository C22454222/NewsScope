import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../core/config.dart';

// ── App-wide theme notifier ────────────────────────────────────────────────
// In your main.dart, call AppTheme.load() before runApp(), then wrap
// MaterialApp with ValueListenableBuilder<ThemeMode>:
//
//   ValueListenableBuilder<ThemeMode>(
//     valueListenable: AppTheme.notifier,
//     builder: (_, mode, __) => MaterialApp(
//       themeMode: mode,
//       theme: ThemeData.light(),
//       darkTheme: ThemeData.dark(),
//       ...
//     ),
//   )
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

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  static const Color _scaffoldBg = Color(0xFFF0F2F5);

  User? get user => FirebaseAuth.instance.currentUser;

  // ── Preferences ────────────────────────────────────────────────────────────
  bool _notificationsEnabled = false;
  bool _compactCards         = false;
  bool _showCredibility      = true;
  bool _showSentiment        = true;
  bool _darkMode             = false;
  int  _dailyGoal            = 3;

  static const List<int> _goalOptions = [1, 3, 5, 10, 20];

  String? _displayName;

  @override
  void initState() {
    super.initState();
    _displayName = user?.displayName;
    _loadPreferences();
  }

  Future<void> _loadPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      _notificationsEnabled = prefs.getBool('notifications_enabled') ?? false;
      _compactCards         = prefs.getBool('compact_cards')         ?? false;
      _showCredibility      = prefs.getBool('show_credibility')      ?? true;
      _showSentiment        = prefs.getBool('show_sentiment')        ?? true;
      _darkMode             = prefs.getBool('dark_mode')             ?? false;
      _dailyGoal            = prefs.getInt('daily_goal')             ?? 3;
    });
  }

  Future<void> _saveBool(String key, bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(key, value);
  }

  Future<void> _saveInt(String key, int value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(key, value);
  }

  // ── Notifications ──────────────────────────────────────────────────────────

  Future<void> _setNotifications(bool value) async {
    if (value) {
      final settings = await FirebaseMessaging.instance.requestPermission(
        alert: true, badge: true, sound: true,
      );
      final granted =
          settings.authorizationStatus == AuthorizationStatus.authorized ||
          settings.authorizationStatus == AuthorizationStatus.provisional;
      if (!granted) {
        if (!mounted) return;
        final openSettings = await showDialog<bool>(
          context: context,
          builder: (context) => AlertDialog(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            title: const Text('Notifications Blocked'),
            content: const Text(
              'NewsScope needs notification permission. Open Settings to enable it.',
              style: TextStyle(height: 1.5),
            ),
            actions: [
              TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
              TextButton(
                onPressed: () => Navigator.pop(context, true),
                child: Text('Open Settings', style: TextStyle(color: Colors.blue[700])),
              ),
            ],
          ),
        );
        if (openSettings == true) await openAppSettings();
        return;
      }
      await FirebaseMessaging.instance.subscribeToTopic('news_updates');
    } else {
      await FirebaseMessaging.instance.unsubscribeFromTopic('news_updates');
    }
    await _saveBool('notifications_enabled', value);
    if (!mounted) return;
    setState(() => _notificationsEnabled = value);
    _showSnackBar(
        value ? 'Notifications enabled' : 'Notifications disabled',
        color: Colors.green[700]);
  }

  // ── Theme toggle ───────────────────────────────────────────────────────────

  Future<void> _setDarkMode(bool value) async {
    await AppTheme.set(value);
    if (!mounted) return;
    setState(() => _darkMode = value);
    _showSnackBar(
        value ? 'Dark mode enabled' : 'Light mode enabled',
        color: Colors.green[700]);
  }

  // ── Reading goal ───────────────────────────────────────────────────────────

  Future<void> _handleReadingGoal() async {
    final selected = await showDialog<int>(
      context: context,
      builder: (context) => SimpleDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Text('Daily Reading Goal'),
        children: _goalOptions.map((goal) {
          final isCurrent = goal == _dailyGoal;
          return SimpleDialogOption(
            onPressed: () => Navigator.pop(context, goal),
            child: Row(children: [
              Icon(
                isCurrent ? Icons.radio_button_checked : Icons.radio_button_unchecked,
                color: isCurrent ? Colors.blue[700] : Colors.grey[400],
                size: 20,
              ),
              const SizedBox(width: 12),
              Text(
                '$goal article${goal == 1 ? '' : 's'} per day',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: isCurrent ? FontWeight.bold : FontWeight.normal,
                  color: isCurrent ? Colors.blue[700] : Colors.grey[800],
                ),
              ),
            ]),
          );
        }).toList(),
      ),
    );
    if (selected == null || !mounted) return;
    setState(() => _dailyGoal = selected);
    await _saveInt('daily_goal', selected);
    _showSnackBar(
        'Goal set to $selected article${selected == 1 ? '' : 's'} per day',
        color: Colors.green[700]);
  }

  // ── Clear reading history ──────────────────────────────────────────────────

  Future<void> _handleClearHistory() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
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
            style: TextButton.styleFrom(foregroundColor: Colors.red[700]),
            child: const Text('Clear History',
                style: TextStyle(fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    try {
      final idToken = await user?.getIdToken();
      final uid    = user?.uid;
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

  // ── Display name ───────────────────────────────────────────────────────────

  Future<void> _handleEditDisplayName() async {
    final currentName = (_displayName ?? '').trim();
    final result = await showDialog<String>(
      context: context,
      barrierDismissible: true,
      builder: (_) => _EditDisplayNameDialog(initialValue: currentName),
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

  // ── Change password ────────────────────────────────────────────────────────

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
      _showSnackBar('Failed to send reset email: $e', color: Colors.red);
    }
  }

  // ── Privacy policy ─────────────────────────────────────────────────────────

  void _showPrivacyPolicy() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
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
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
              child: Row(children: [
                Icon(Icons.privacy_tip_outlined, color: Colors.blue[700]),
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
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 40),
                children: [
                  _privacySection('Overview',
                      'NewsScope ("we", "us", "our") is a news analysis application developed as a Final Year Project at Technological University Dublin. We are committed to protecting your personal data and processing it in compliance with the General Data Protection Regulation (GDPR) (EU) 2016/679 and applicable Irish data protection law.'),
                  _privacySection('Data Controller',
                      'The data controller for NewsScope is the developer at Technological University Dublin. For data protection queries, contact the developer through the University.'),
                  _privacySection('What Personal Data We Collect',
                      'We collect the following personal data:\n\n'
                      '• Email address — for authentication purposes.\n'
                      '• Display name — the name you choose to show in the app.\n'
                      '• Article reading history — which articles you read and for how long, used solely to generate your Bias Profile.\n'
                      '• Firebase UID — a unique identifier used to link your data securely.\n\n'
                      'We do not collect your full name, phone number, payment information, or location data.'),
                  _privacySection('Legal Basis for Processing (GDPR Article 6)',
                      'We process your personal data on the following legal bases:\n\n'
                      '• Contractual necessity — to provide the NewsScope service you signed up for.\n'
                      '• Legitimate interests — to improve the service and detect misuse.\n'
                      '• Consent — for optional push notifications, which you may withdraw at any time in Settings.'),
                  _privacySection('How We Use Your Data',
                      'Your data is used exclusively to:\n\n'
                      '• Authenticate your account and maintain a secure session.\n'
                      '• Calculate and display your personal Bias Profile and reading statistics.\n'
                      '• Send push notifications about new articles (only if you have opted in).\n\n'
                      'We do not use your data for advertising, commercial profiling, or automated decision-making that produces legal effects.'),
                  _privacySection('Data Sharing & Third Parties',
                      'We use the following third-party services. Each has its own privacy policy:\n\n'
                      '• Firebase (Google LLC) — authentication and push notifications, processed under Google\'s Standard Contractual Clauses for GDPR compliance.\n'
                      '• Supabase — secure database storage for reading history and bias profile data, with servers within the EU.\n'
                      '• News providers (BBC, RTÉ, The Guardian, etc.) — article content is sourced from these providers; your identity is never shared with them.\n\n'
                      'We do not sell, rent, or trade your personal data to any third party under any circumstances.'),
                  _privacySection('Data Retention',
                      'Your personal data is retained for as long as your account is active. You may delete your account at any time via Settings → Account Actions → Delete Account. Upon deletion, your reading history, Bias Profile, and credentials are permanently removed within 30 days.'),
                  _privacySection('International Data Transfers',
                      'Firebase may process data outside the European Economic Area (EEA). Google LLC participates in the EU–U.S. Data Privacy Framework and provides Standard Contractual Clauses to ensure GDPR-compliant transfers.'),
                  _privacySection('Your Rights Under GDPR',
                      'As a data subject you have the right to:\n\n'
                      '• Access — request a copy of your personal data.\n'
                      '• Rectification — correct inaccurate data (e.g. display name) in Settings.\n'
                      '• Erasure ("right to be forgotten") — delete your account and all data via Settings.\n'
                      '• Restriction — request that we limit processing of your data.\n'
                      '• Data portability — receive your data in a structured, machine-readable format.\n'
                      '• Object — object to processing based on legitimate interests.\n'
                      '• Withdraw consent — for notifications, withdraw at any time in Settings.\n\n'
                      'To exercise these rights, contact the developer at Technological University Dublin.'),
                  _privacySection('Cookies & Tracking',
                      'The NewsScope mobile app does not use cookies. Firebase may use device identifiers for authentication token management. No third-party advertising trackers are used.'),
                  _privacySection('Security',
                      'We implement appropriate technical and organisational measures to protect your data, including:\n\n'
                      '• Firebase Authentication with industry-standard token management.\n'
                      '• HTTPS encryption for all API communications.\n'
                      '• Row-level security policies on our Supabase database so users can only access their own data.'),
                  _privacySection('Children\'s Privacy',
                      'NewsScope is not directed at children under 13. We do not knowingly collect personal data from children. If you believe a child has provided data, contact us immediately for removal.'),
                  _privacySection('Complaints',
                      'If you believe we have not handled your personal data appropriately, you have the right to lodge a complaint with the Irish Data Protection Commission (DPC) at www.dataprotection.ie.'),
                  const SizedBox(height: 12),
                  Text(
                    'Last updated: March 2026 · NewsScope v1.0.0 Beta\nTechnological University Dublin — Final Year Project',
                    style: TextStyle(fontSize: 11, color: Colors.grey[400]),
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
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(title,
            style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
                color: Colors.blue[800])),
        const SizedBox(height: 6),
        Text(body,
            style: TextStyle(
                fontSize: 13, color: Colors.grey[700], height: 1.55)),
      ]),
    );
  }

  // ── Logout ─────────────────────────────────────────────────────────────────

  Future<void> _handleLogout() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Text('Sign Out'),
        content: const Text('Are you sure you want to sign out?'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Sign Out', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
    if (confirmed == true && mounted) {
      await FirebaseAuth.instance.signOut();
      if (!mounted) return;
      Navigator.of(context).popUntil((route) => route.isFirst);
    }
  }

  // ── Delete account ─────────────────────────────────────────────────────────

  Future<void> _handleDeleteAccount() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Text('Delete Account',
            style: TextStyle(color: Colors.red)),
        content: const Text(
          'This will permanently delete your account and all reading history. This cannot be undone.',
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
                    color: Colors.red, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;
    final uid = user?.uid;
    if (uid == null) return;
    await _attemptDeleteAccount(uid);
  }

  Future<void> _attemptDeleteAccount(String uid) async {
    try {
      final isGoogleUser =
          user?.providerData.any((p) => p.providerId == 'google.com') ?? false;
      if (isGoogleUser) {
        final success = await _reauthWithGoogle();
        if (!success) return;
      }
      final idToken = await user?.getIdToken();
      await user?.delete();
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
      final googleUser   = await googleSignIn.signIn();
      if (googleUser == null) return false;
      final googleAuth = await googleUser.authentication;
      final credential = GoogleAuthProvider.credential(
          idToken: googleAuth.idToken,
          accessToken: googleAuth.accessToken);
      await user?.reauthenticateWithCredential(credential);
      return true;
    } catch (_) {
      if (mounted) {
        _showSnackBar('Google re-authentication failed.', color: Colors.red);
      }
      return false;
    }
  }

  Future<bool> _reauthWithPassword() async {
    if (!mounted) return false;
    final password = await showDialog<String>(
      context: context,
      barrierDismissible: false,
      builder: (_) => _ReauthPasswordDialog(email: user?.email ?? ''),
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

  // ── Snackbar ───────────────────────────────────────────────────────────────

  void _showSnackBar(String message, {Color? color}) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(message),
      backgroundColor: color,
      behavior: SnackBarBehavior.floating,
      duration: const Duration(seconds: 3),
    ));
  }

  // ── Shared widget builders ─────────────────────────────────────────────────

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
        backgroundColor: (iconColor ?? Colors.blue[700]!).withAlpha(25),
        child: Icon(icon, size: 18, color: iconColor ?? Colors.blue[700]),
      ),
      title: Text(title,
          style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: titleColor ?? Colors.grey[800])),
      subtitle: subtitle != null
          ? Text(subtitle,
              style: TextStyle(fontSize: 12, color: Colors.grey[500]))
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
        backgroundColor: (iconColor ?? Colors.blue[700]!).withAlpha(25),
        child: Icon(icon, size: 18, color: iconColor ?? Colors.blue[700]),
      ),
      title: Text(title,
          style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: Colors.grey[800])),
      subtitle: Text(subtitle,
          style: TextStyle(fontSize: 12, color: Colors.grey[500])),
      activeThumbColor: Colors.blue[700],
      activeTrackColor: Colors.blue[300],
      value: value,
      onChanged: onChanged,
    );
  }

  // ── Goal progress tile ─────────────────────────────────────────────────────
  // Reads today's article count from SharedPreferences key 'articles_today'.
  // Increment that key from your article-open logic and reset it daily.

  Widget _buildGoalProgressTile(int todayCount) {
    final pct =
        (_dailyGoal > 0 ? (todayCount / _dailyGoal).clamp(0.0, 1.0) : 0.0);
    final done = todayCount >= _dailyGoal;

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
      child: Row(children: [
        CircleAvatar(
          radius: 18,
          backgroundColor:
              (done ? Colors.green[700]! : Colors.purple[600]!).withAlpha(25),
          child: Icon(
            done ? Icons.check_circle_outline : Icons.today_outlined,
            size: 18,
            color: done ? Colors.green[700] : Colors.purple[600],
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              Text(
                done ? 'Goal reached today! 🎉' : 'Today\'s progress',
                style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                    color: Colors.grey[800]),
              ),
              Text(
                '$todayCount / $_dailyGoal',
                style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                    color: done ? Colors.green[700] : Colors.purple[600]),
              ),
            ]),
            const SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: pct,
                minHeight: 6,
                backgroundColor: Colors.grey[200],
                valueColor: AlwaysStoppedAnimation(
                    done ? Colors.green[600]! : Colors.purple[400]!),
              ),
            ),
          ]),
        ),
      ]),
    );
  }

  // ── Glossary ───────────────────────────────────────────────────────────────

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
          'A 0–100 score estimating how reliable an article is, based on source reputation and fact-check results. Above 70 = Reliable. 40–70 = Mixed. Below 40 = Low.'
    ),
    (
      term: 'Biased',
      icon: Icons.warning_amber,
      color: Color(0xFFE64A19),
      definition:
          'Indicates the article uses one-sided framing, emotionally charged language, or selective facts beyond what is typical for its leaning. An article can lean Left but still be Unbiased if it presents information fairly.'
    ),
    (
      term: 'Unbiased',
      icon: Icons.check_circle_outline,
      color: Color(0xFF388E3C),
      definition:
          'The article presents information without significant one-sided framing. Unbiased does not mean neutral in leaning — it means the information is fairly presented.'
    ),
    (
      term: 'Bias Profile',
      icon: Icons.pie_chart,
      color: Color(0xFF1565C0),
      definition:
          'Your personal reading summary showing the distribution of political leanings, sentiment, and sources across every article you have read. Updated automatically as you read.'
    ),
    (
      term: 'Ideological Spectrum',
      icon: Icons.linear_scale,
      color: Color(0xFF1565C0),
      definition:
          'The colour gradient bar on each article page. The marker position reflects the article\'s political bias score on a continuous scale from Far Left to Far Right.'
    ),
    (
      term: 'Story Comparison',
      icon: Icons.compare_arrows,
      color: Color(0xFF00796B),
      definition:
          'The Compare tab groups articles on the same topic or category by political leaning (Left, Centre, Right), letting you see how different outlets frame the same story.'
    ),
    (
      term: 'Fact Check',
      icon: Icons.verified_outlined,
      color: Color(0xFF388E3C),
      definition:
          'NewsScope cross-references key claims in articles against a fact-checking database. Each claim receives a ruling: True, Mostly True, Half True, Mostly False, or False.'
    ),
    (
      term: 'LIME Explanation',
      icon: Icons.psychology_outlined,
      color: Color(0xFF6A1B9A),
      definition:
          'LIME (Local Interpretable Model-Agnostic Explanations) is the AI technique used to show which specific words most strongly influenced the political bias classification for an article.'
    ),
  ];

  Widget _buildGlossarySection() {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16),
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
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
            style: TextStyle(fontSize: 12, color: Colors.grey[500])),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
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
                  child: Icon(t.icon, size: 18, color: t.color),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
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

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final isGoogleUser =
        user?.providerData.any((p) => p.providerId == 'google.com') ?? false;

    // Read today's article count — replace 0 with your cached prefs value
    // e.g. SharedPreferences.getInstance().then((p) => p.getInt('articles_today') ?? 0)
    // For now we pass 0 synchronously; load it in initState if you need live count.
    const int todayCount = 0;

    return Scaffold(
      backgroundColor: _scaffoldBg,
      appBar: AppBar(
        backgroundColor: _scaffoldBg,
        centerTitle: true,
        title: Text('Settings',
            style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.blue[800])),
      ),
      body: ListView(
        children: [

          // ── Account ──────────────────────────────────────────────────
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

          // ── Notifications ─────────────────────────────────────────────
          _buildSectionHeader('Notifications'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12)),
            child: _buildSwitch(
              icon: Icons.notifications_outlined,
              title: 'Breaking News Alerts',
              subtitle: _notificationsEnabled
                  ? 'You\'ll be notified when new articles are analysed'
                  : 'Tap to enable push notifications',
              value: _notificationsEnabled,
              onChanged: _setNotifications,
            ),
          ),

          // ── Display Preferences ───────────────────────────────────────
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
                },
                iconColor: Colors.indigo[600],
              ),
            ]),
          ),

          // ── Reading Goals ─────────────────────────────────────────────
          _buildSectionHeader('Reading Goals'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12)),
            child: Column(children: [
              _buildTile(
                icon: Icons.track_changes_outlined,
                title: 'Daily Reading Goal',
                subtitle:
                    '$_dailyGoal article${_dailyGoal == 1 ? '' : 's'} per day',
                iconColor: Colors.purple[600],
                onTap: _handleReadingGoal,
                trailing: Row(mainAxisSize: MainAxisSize.min, children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.purple[50],
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: Colors.purple[200]!),
                    ),
                    child: Text(
                      '$_dailyGoal',
                      style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.bold,
                          color: Colors.purple[700]),
                    ),
                  ),
                  const SizedBox(width: 4),
                  Icon(Icons.chevron_right,
                      color: Colors.grey[400], size: 20),
                ]),
              ),
              const Divider(height: 1, indent: 56),
              _buildGoalProgressTile(todayCount),
            ]),
          ),

          // ── Data & Privacy ────────────────────────────────────────────
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

          // ── About ─────────────────────────────────────────────────────
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

          // ── Glossary ──────────────────────────────────────────────────
          _buildSectionHeader('Glossary'),
          _buildGlossarySection(),

          // ── Account Actions ───────────────────────────────────────────
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
              style: TextStyle(fontSize: 11, color: Colors.grey[400]),
            ),
          ),
          const SizedBox(height: 24),
        ],
      ),
    );
  }
}

// ── Re-auth password dialog ────────────────────────────────────────────────────

class _ReauthPasswordDialog extends StatefulWidget {
  final String email;
  const _ReauthPasswordDialog({required this.email});
  @override
  State<_ReauthPasswordDialog> createState() => _ReauthPasswordDialogState();
}

class _ReauthPasswordDialogState extends State<_ReauthPasswordDialog> {
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
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      title: const Text('Confirm Your Password'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Enter your password for ${widget.email} to confirm account deletion.',
            style: TextStyle(
                fontSize: 13, color: Colors.grey[600], height: 1.4),
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
                icon: Icon(
                    _obscure ? Icons.visibility_off : Icons.visibility),
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
          style:
              TextButton.styleFrom(foregroundColor: Colors.red[700]),
          child: const Text('Confirm Delete',
              style: TextStyle(fontWeight: FontWeight.bold)),
        ),
      ],
    );
  }
}

// ── Edit display name dialog ───────────────────────────────────────────────────

class _EditDisplayNameDialog extends StatefulWidget {
  final String initialValue;
  const _EditDisplayNameDialog({required this.initialValue});
  @override
  State<_EditDisplayNameDialog> createState() =>
      _EditDisplayNameDialogState();
}

class _EditDisplayNameDialogState extends State<_EditDisplayNameDialog> {
  late final TextEditingController _controller;
  late final FocusNode _focusNode;
  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.initialValue);
    _focusNode  = FocusNode();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      _focusNode.requestFocus();
      _controller.selection = TextSelection(
          baseOffset: 0, extentOffset: _controller.text.length);
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
      shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
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
