import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  User? get user => FirebaseAuth.instance.currentUser;
  bool _notificationsEnabled = false;
  String? _displayName;

  @override
  void initState() {
    super.initState();
    _displayName = user?.displayName;
    _loadPreferences();
  }

  // ── Preferences persistence ───────────────────────────────────────────────

  Future<void> _loadPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      _notificationsEnabled = prefs.getBool('notifications_enabled') ?? false;
    });
  }

  Future<void> _setNotifications(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('notifications_enabled', value);
    if (!mounted) return;
    setState(() => _notificationsEnabled = value);
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(value ? 'Notifications enabled' : 'Notifications disabled'),
      duration: const Duration(seconds: 1),
      behavior: SnackBarBehavior.floating,
    ));
  }

  // ── Edit display name ─────────────────────────────────────────────────────

  Future<void> _handleEditDisplayName() async {
    final controller = TextEditingController(text: _displayName ?? '');
    final result = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: const Text('Edit Display Name'),
        content: TextField(
          controller: controller,
          autofocus: true,
          maxLength: 40,
          decoration: const InputDecoration(
            labelText: 'Display name',
            border: OutlineInputBorder(),
          ),
          textCapitalization: TextCapitalization.words,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, controller.text.trim()),
            child: const Text('Save'),
          ),
        ],
      ),
    );

    if (result == null || result.isEmpty || !mounted) return;

    try {
      await user?.updateDisplayName(result);
      await user?.reload();
      if (!mounted) return;
      setState(() => _displayName = result);
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: const Text('Display name updated'),
        backgroundColor: Colors.green[700],
        behavior: SnackBarBehavior.floating,
      ));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Failed to update name: $e'),
        backgroundColor: Colors.red,
        behavior: SnackBarBehavior.floating,
      ));
    }
  }

  // ── Change password ───────────────────────────────────────────────────────

  Future<void> _handleChangePassword() async {
    final email = user?.email;
    if (email == null) return;
    try {
      await FirebaseAuth.instance.sendPasswordResetEmail(email: email);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Password reset email sent to $email'),
        backgroundColor: Colors.green[700],
        behavior: SnackBarBehavior.floating,
      ));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Failed to send reset email: $e'),
        backgroundColor: Colors.red,
        behavior: SnackBarBehavior.floating,
      ));
    }
  }

  // ── Privacy policy modal ──────────────────────────────────────────────────

  void _showPrivacyPolicy() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => DraggableScrollableSheet(
        expand: false,
        initialChildSize: 0.6,
        maxChildSize: 0.9,
        builder: (_, scrollController) => Column(
          children: [
            const SizedBox(height: 12),
            Container(
              width: 40, height: 4,
              decoration: BoxDecoration(
                color: Colors.grey[300],
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
              child: Row(children: [
                Icon(Icons.privacy_tip_outlined, color: Colors.blue[700]),
                const SizedBox(width: 10),
                Text('Privacy Policy',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold,
                        color: Colors.grey[800])),
              ]),
            ),
            const Divider(),
            Expanded(
              child: ListView(
                controller: scrollController,
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
                children: [
                  _privacySection('Data We Collect',
                      'NewsScope collects your email address for authentication and your article reading history to generate your bias profile. No personal data is sold to third parties.'),
                  _privacySection('How We Use Your Data',
                      'Your reading history is used solely to calculate your Bias Profile and reading statistics displayed within the app. This data is stored securely and is only accessible to you.'),
                  _privacySection('Data Retention',
                      'Your account data is retained for as long as your account is active. You may delete your account at any time from the Account Actions section of Settings, which permanently removes all associated data.'),
                  _privacySection('Third-Party Services',
                      'NewsScope uses Firebase for authentication and Supabase for data storage. These services have their own privacy policies. Article content is sourced from third-party news providers.'),
                  _privacySection('Contact',
                      'For any privacy-related queries, please contact the developer via Technological University Dublin.'),
                  const SizedBox(height: 8),
                  Text('Last updated: March 2026 · NewsScope v1.0.0',
                      style: TextStyle(fontSize: 11, color: Colors.grey[400]),
                      textAlign: TextAlign.center),
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
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold,
                color: Colors.blue[800])),
        const SizedBox(height: 6),
        Text(body,
            style: TextStyle(fontSize: 13, color: Colors.grey[700], height: 1.5)),
      ]),
    );
  }

  // ── Logout ────────────────────────────────────────────────────────────────

  Future<void> _handleLogout() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Sign Out'),
        content: const Text('Are you sure you want to sign out?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
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

  // ── Delete account ────────────────────────────────────────────────────────

  Future<void> _handleDeleteAccount() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
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
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete',
                style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );

    if (confirmed != true || !mounted) return;

    try {
      await user?.delete();
      if (!mounted) return;
      Navigator.of(context).popUntil((route) => route.isFirst);
    } on FirebaseAuthException catch (e) {
      if (!mounted) return;
      if (e.code == 'requires-recent-login') {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text(
              'Please sign out and sign back in before deleting your account.'),
          behavior: SnackBarBehavior.floating,
        ));
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Failed to delete account: ${e.message}'),
          backgroundColor: Colors.red,
          behavior: SnackBarBehavior.floating,
        ));
      }
    }
  }

  // ── Shared widgets ────────────────────────────────────────────────────────

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 8),
      child: Text(
        title.toUpperCase(),
        style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold,
            color: Colors.blue[700], letterSpacing: 1.2),
      ),
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
          style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500,
              color: titleColor ?? Colors.grey[800])),
      subtitle: subtitle != null
          ? Text(subtitle,
              style: TextStyle(fontSize: 12, color: Colors.grey[500]))
          : null,
      trailing: trailing ??
          (onTap != null
              ? Icon(Icons.chevron_right, color: Colors.grey[400], size: 20)
              : null),
      onTap: onTap,
    );
  }

  // ── Glossary ──────────────────────────────────────────────────────────────

  static const _glossaryTerms = [
    (
      term: 'Political Leaning',
      icon: Icons.balance,
      color: Color(0xFF1565C0),
      definition:
          'Whether an article\'s language aligns with left-wing, centre, or right-wing political perspectives, determined by a machine learning model.'
    ),
    (
      term: 'Left Wing',
      icon: Icons.arrow_back,
      color: Color(0xFF1565C0),
      definition:
          'Content favouring progressive values such as social equality, expanded public services, and government intervention in the economy.'
    ),
    (
      term: 'Centre',
      icon: Icons.remove,
      color: Color(0xFF00796B),
      definition:
          'Content that does not lean strongly in either direction, typically presenting multiple perspectives without strongly favouring one side.'
    ),
    (
      term: 'Right Wing',
      icon: Icons.arrow_forward,
      color: Color(0xFFC62828),
      definition:
          'Content favouring conservative values such as free markets, traditional institutions, and limited government intervention.'
    ),
    (
      term: 'Sentiment',
      icon: Icons.sentiment_satisfied,
      color: Color(0xFF388E3C),
      definition:
          'The emotional tone of an article. Positive means optimistic or favourable language, negative means critical or alarming, neutral sits between.'
    ),
    (
      term: 'Credibility',
      icon: Icons.fact_check_outlined,
      color: Color(0xFF388E3C),
      definition:
          'A 0–100 score estimating reliability based on source reputation and fact-checks. Above 70 = Reliable, 40–70 = Mixed, below 40 = Low.'
    ),
    (
      term: 'Biased / Unbiased',
      icon: Icons.warning_amber,
      color: Color(0xFFE64A19),
      definition:
          'A flag indicating one-sided framing beyond normal leaning. An article can be Left Wing but Unbiased if it presents facts without manipulation.'
    ),
    (
      term: 'Bias Profile',
      icon: Icons.pie_chart,
      color: Color(0xFF1565C0),
      definition:
          'A personal summary of your reading history showing the distribution of political leanings and sentiment across articles you\'ve read.'
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
          child: Icon(Icons.menu_book_outlined, size: 18, color: Colors.blue[700]),
        ),
        title: Text('Glossary',
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500,
                color: Colors.grey[800])),
        subtitle: Text('Tap to learn key terms',
            style: TextStyle(fontSize: 12, color: Colors.grey[500])),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        children: _glossaryTerms.map((t) {
          return Padding(
            padding: const EdgeInsets.only(top: 12),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 36, height: 36,
                  decoration: BoxDecoration(
                    color: t.color.withAlpha(25),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(t.icon, size: 18, color: t.color),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(t.term,
                          style: TextStyle(fontSize: 13,
                              fontWeight: FontWeight.bold, color: t.color)),
                      const SizedBox(height: 3),
                      Text(t.definition,
                          style: TextStyle(fontSize: 12,
                              color: Colors.grey[600], height: 1.4)),
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

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: Text('Settings',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold,
                color: Colors.blue[800])),
      ),
      body: ListView(
        children: [
          // ── Account ───────────────────────────────────────────────────────
          _buildSectionHeader('Account'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
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
              const Divider(height: 1, indent: 56),
              _buildTile(
                icon: Icons.lock_reset_outlined,
                title: 'Change Password',
                subtitle: 'Send a reset link to your email',
                onTap: _handleChangePassword,
              ),
            ]),
          ),

          // ── Preferences ───────────────────────────────────────────────────
          _buildSectionHeader('Preferences'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: SwitchListTile(
              secondary: CircleAvatar(
                radius: 18,
                backgroundColor: Colors.blue[700]!.withAlpha(25),
                child: Icon(Icons.notifications_outlined,
                    size: 18, color: Colors.blue[700]),
              ),
              title: Text('Notifications',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500,
                      color: Colors.grey[800])),
              subtitle: Text('Breaking news alerts',
                  style: TextStyle(fontSize: 12, color: Colors.grey[500])),
              activeThumbColor: Colors.blue[700],
              activeTrackColor: Colors.blue[300],
              value: _notificationsEnabled,
              onChanged: _setNotifications,
            ),
          ),

          // ── About ─────────────────────────────────────────────────────────
          _buildSectionHeader('About'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: Column(children: [
              _buildTile(
                icon: Icons.info_outline,
                title: 'App Version',
                subtitle: '1.0.0 — Final Year Demo',
                trailing: const SizedBox.shrink(),
              ),
              const Divider(height: 1, indent: 56),
              _buildTile(
                icon: Icons.newspaper,
                title: 'NewsScope',
                subtitle: 'Bias-aware news aggregation for everyone.',
                trailing: const SizedBox.shrink(),
              ),
              const Divider(height: 1, indent: 56),
              _buildTile(
                icon: Icons.privacy_tip_outlined,
                title: 'Privacy Policy',
                trailing: Icon(Icons.open_in_new, size: 16,
                    color: Colors.grey[400]),
                onTap: _showPrivacyPolicy,
              ),
            ]),
          ),

          // ── Glossary ──────────────────────────────────────────────────────
          _buildSectionHeader('Glossary'),
          _buildGlossarySection(),

          // ── Account Actions ───────────────────────────────────────────────
          _buildSectionHeader('Account Actions'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
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
                subtitle: 'Permanently remove your account and data',
                iconColor: Colors.red[800],
                titleColor: Colors.red[800],
                trailing: const SizedBox.shrink(),
                onTap: _handleDeleteAccount,
              ),
            ]),
          ),

          const SizedBox(height: 32),
        ],
      ),
    );
  }
}
