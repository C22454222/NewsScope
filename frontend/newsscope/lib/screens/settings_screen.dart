import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final user = FirebaseAuth.instance.currentUser;
  bool _notificationsEnabled = false;

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
            child: const Text(
              'Sign Out',
              style: TextStyle(color: Colors.red),
            ),
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

  Future<void> _handleChangePassword() async {
    final email = user?.email;
    if (email == null) return;
    try {
      await FirebaseAuth.instance.sendPasswordResetEmail(email: email);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Password reset email sent to $email'),
          backgroundColor: Colors.green[700],
          behavior: SnackBarBehavior.floating,
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to send reset email: $e'),
          backgroundColor: Colors.red,
          behavior: SnackBarBehavior.floating,
        ),
      );
    }
  }

  Widget _buildSettingsTitle() {
    return RichText(
      text: TextSpan(
        style: const TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.bold,
          letterSpacing: 0.3,
        ),
        children: [
          const TextSpan(
            text: 'App ',
            style: TextStyle(color: Colors.white),
          ),
          TextSpan(
            text: 'Settings',
            style: TextStyle(color: Colors.blue[200]),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 8),
      child: Text(
        title.toUpperCase(),
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.bold,
          color: Colors.blue[700],
          letterSpacing: 1.2,
        ),
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
      title: Text(
        title,
        style: TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w500,
          color: titleColor ?? Colors.grey[800],
        ),
      ),
      subtitle: subtitle != null
          ? Text(
              subtitle,
              style: TextStyle(fontSize: 12, color: Colors.grey[500]),
            )
          : null,
      trailing: trailing ??
          (onTap != null
              ? Icon(Icons.chevron_right, color: Colors.grey[400], size: 20)
              : null),
      onTap: onTap,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: _buildSettingsTitle(),
      ),
      body: ListView(
        children: [
          // ── Account ───────────────────────────────────────────────────────
          _buildSectionHeader('Account'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            child: Column(
              children: [
                _buildTile(
                  icon: Icons.person_outline,
                  title: 'Display Name',
                  subtitle: user?.displayName?.isNotEmpty == true
                      ? user!.displayName!
                      : 'Not set',
                ),
                const Divider(height: 1, indent: 56),
                _buildTile(
                  icon: Icons.email_outlined,
                  title: 'Email',
                  subtitle: user?.email ?? 'Not signed in',
                ),
                const Divider(height: 1, indent: 56),
                _buildTile(
                  icon: Icons.lock_reset_outlined,
                  title: 'Change Password',
                  subtitle: 'Send a reset link to your email',
                  onTap: _handleChangePassword,
                ),
              ],
            ),
          ),

          // ── Preferences ───────────────────────────────────────────────────
          _buildSectionHeader('Preferences'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            child: Column(
              children: [
                SwitchListTile(
                  secondary: CircleAvatar(
                    radius: 18,
                    backgroundColor: Colors.blue[700]!.withAlpha(25),
                    child: Icon(
                      Icons.notifications_outlined,
                      size: 18,
                      color: Colors.blue[700],
                    ),
                  ),
                  title: Text(
                    'Notifications',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: Colors.grey[800],
                    ),
                  ),
                  subtitle: Text(
                    'Breaking news alerts',
                    style: TextStyle(fontSize: 12, color: Colors.grey[500]),
                  ),
                  activeThumbColor: Colors.blue[700],
                  activeTrackColor: Colors.blue[300],
                  value: _notificationsEnabled,
                  onChanged: (value) {
                    setState(() => _notificationsEnabled = value);
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text(
                          value
                              ? 'Notifications enabled'
                              : 'Notifications disabled',
                        ),
                        duration: const Duration(seconds: 1),
                        behavior: SnackBarBehavior.floating,
                      ),
                    );
                  },
                ),
                const Divider(height: 1, indent: 56),
                _buildTile(
                  icon: Icons.source_outlined,
                  title: 'Preferred Sources',
                  subtitle: 'BBC, CNN, RTÉ +3 more',
                  onTap: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Source preferences coming soon!'),
                        behavior: SnackBarBehavior.floating,
                      ),
                    );
                  },
                ),
              ],
            ),
          ),

          // ── About ─────────────────────────────────────────────────────────
          _buildSectionHeader('About'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            child: Column(
              children: [
                _buildTile(
                  icon: Icons.info_outline,
                  title: 'App Version',
                  subtitle: '1.0.0 — Final Year Demo',
                  trailing: const SizedBox.shrink(),
                ),
                const Divider(height: 1, indent: 56),
                _buildTile(
                  icon: Icons.privacy_tip_outlined,
                  title: 'Privacy Policy',
                  trailing: Icon(
                    Icons.open_in_new,
                    size: 16,
                    color: Colors.grey[400],
                  ),
                  onTap: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Privacy policy coming soon!'),
                        behavior: SnackBarBehavior.floating,
                      ),
                    );
                  },
                ),
              ],
            ),
          ),

          // ── Account Actions ───────────────────────────────────────────────
          _buildSectionHeader('Account Actions'),
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            elevation: 1,
            child: _buildTile(
              icon: Icons.logout,
              title: 'Sign Out',
              iconColor: Colors.red[600],
              titleColor: Colors.red[600],
              trailing: const SizedBox.shrink(),
              onTap: _handleLogout,
            ),
          ),

          const SizedBox(height: 32),
        ],
      ),
    );
  }
}
