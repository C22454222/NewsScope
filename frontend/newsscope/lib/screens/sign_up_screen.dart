import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:firebase_auth/firebase_auth.dart';

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final _formKey = GlobalKey<FormState>();
  final _usernameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLoading = false;
  bool _obscurePassword = true;
  bool _verificationSent = false;

  static final _emailRegex = RegExp(
    r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$',
  );

  @override
  void dispose() {
    _usernameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  String? _validateUsername(String? value) {
    final v = value?.trim() ?? '';
    if (v.isEmpty) return 'Display name is required';
    if (v.length < 2) return 'Display name must be at least 2 characters';
    if (v.length > 30) return 'Display name must be 30 characters or fewer';
    if (!RegExp(r"^[a-zA-Z0-9 '_\-]+$").hasMatch(v)) {
      return "Only letters, numbers, spaces, ' _ - allowed";
    }
    return null;
  }

  String? _validateEmail(String? value) {
    final v = value?.trim() ?? '';
    if (v.isEmpty) return 'Email is required';
    if (!_emailRegex.hasMatch(v)) return 'Enter a valid email address';
    return null;
  }

  String? _validatePassword(String? value) {
    final v = value ?? '';
    if (v.isEmpty) return 'Password is required';
    if (v.length < 6) return 'Password must be at least 6 characters';
    if (v.length > 128) return 'Password must be 128 characters or fewer';
    if (v.contains(' ')) return 'Password cannot contain spaces';
    return null;
  }

  Future<void> _signUp() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _isLoading = true);
    try {
      final userCred =
          await FirebaseAuth.instance.createUserWithEmailAndPassword(
        email: _emailController.text.trim(),
        password: _passwordController.text,
      );
      await userCred.user?.updateDisplayName(_usernameController.text.trim());
      await userCred.user?.sendEmailVerification();
      await FirebaseAuth.instance.signOut();
      if (!mounted) return;
      setState(() => _verificationSent = true);
    } on FirebaseAuthException catch (e) {
      _showError(_mapFirebaseError(e.code));
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  String _mapFirebaseError(String code) {
    switch (code) {
      case 'email-already-in-use':
        return 'An account with this email already exists';
      case 'invalid-email':
        return 'Invalid email address';
      case 'weak-password':
        return 'Password is too weak — use at least 6 characters';
      case 'network-request-failed':
        return 'No internet connection';
      default:
        return 'Registration failed — please try again';
    }
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(message),
      backgroundColor: Colors.red[700],
      behavior: SnackBarBehavior.floating,
    ));
  }

  // ── Verification sent screen ──────────────────────────────────────────────

  Widget _buildVerificationSentView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.mark_email_unread_outlined,
                size: 72, color: Colors.blue[700]),
            const SizedBox(height: 24),
            const Text(
              'Check your inbox',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            Text(
              'A verification email has been sent to:\n'
              '${_emailController.text.trim()}\n\n'
              'Click the link in the email to activate your account, '
              'then sign in.',
              textAlign: TextAlign.center,
              style:
                  TextStyle(fontSize: 14, color: Colors.grey[600], height: 1.5),
            ),
            const SizedBox(height: 32),
            ElevatedButton(
              onPressed: () => Navigator.pop(context),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue[700],
                foregroundColor: Colors.white,
                padding:
                    const EdgeInsets.symmetric(vertical: 14, horizontal: 32),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10)),
                textStyle:
                    const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
              child: const Text('Go to Sign In'),
            ),
            const SizedBox(height: 12),
            TextButton(
              onPressed: _isLoading
                  ? null
                  : () async {
                      setState(() => _isLoading = true);
                      try {
                        final userCred = await FirebaseAuth.instance
                            .signInWithEmailAndPassword(
                          email: _emailController.text.trim(),
                          password: _passwordController.text,
                        );
                        await userCred.user?.sendEmailVerification();
                        await FirebaseAuth.instance.signOut();
                        if (!mounted) return;
                        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                          content: const Text('Verification email resent'),
                          backgroundColor: Colors.green[700],
                          behavior: SnackBarBehavior.floating,
                        ));
                      } catch (_) {
                        if (!mounted) return;
                        _showError('Could not resend — please try again');
                      } finally {
                        if (mounted) setState(() => _isLoading = false);
                      }
                    },
              style: TextButton.styleFrom(foregroundColor: Colors.blue[700]),
              child: _isLoading
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text("Didn't receive it? Resend"),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: Text(
          _verificationSent ? 'Verify Your Email' : 'Create Account',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: Colors.blue[800],
          ),
        ),
      ),
      body: SafeArea(
        child: _verificationSent
            ? _buildVerificationSentView()
            : Center(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(28),
                  child: Form(
                    key: _formKey,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        // ── Header banner ─────────────────────────────────
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.fromLTRB(20, 18, 20, 18),
                          decoration: BoxDecoration(
                            color: const Color(0xFF0D1B3E),
                            borderRadius: BorderRadius.circular(16),
                            boxShadow: [
                              BoxShadow(
                                  color: Colors.black.withAlpha(50),
                                  blurRadius: 12,
                                  offset: const Offset(0, 4))
                            ],
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text('Join NewsScope',
                                  style: TextStyle(
                                      fontSize: 22,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.white)),
                              const SizedBox(height: 4),
                              Text(
                                'Track media bias and analyse news\n'
                                'across the political spectrum.',
                                style: TextStyle(
                                    color: Colors.white.withAlpha(180),
                                    fontSize: 13,
                                    height: 1.4),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 32),

                        // ── Display name ───────────────────────────────────
                        TextFormField(
                          controller: _usernameController,
                          validator: _validateUsername,
                          textCapitalization: TextCapitalization.words,
                          inputFormatters: [
                            FilteringTextInputFormatter.allow(
                                RegExp(r"[a-zA-Z0-9 '_\-]")),
                            LengthLimitingTextInputFormatter(30),
                          ],
                          decoration: InputDecoration(
                            labelText: 'Display Name',
                            hintText: 'e.g. John Smith',
                            prefixIcon: Icon(Icons.person_outline,
                                color: Colors.blue[700]),
                            border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(10)),
                            focusedBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(10),
                              borderSide:
                                  BorderSide(color: Colors.blue[700]!, width: 2),
                            ),
                          ),
                        ),
                        const SizedBox(height: 16),

                        // ── Email ──────────────────────────────────────────
                        TextFormField(
                          controller: _emailController,
                          validator: _validateEmail,
                          keyboardType: TextInputType.emailAddress,
                          autocorrect: false,
                          inputFormatters: [
                            FilteringTextInputFormatter.deny(RegExp(r'\s')),
                            LengthLimitingTextInputFormatter(254),
                          ],
                          decoration: InputDecoration(
                            labelText: 'Email',
                            hintText: 'you@example.com',
                            prefixIcon: Icon(Icons.email_outlined,
                                color: Colors.blue[700]),
                            border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(10)),
                            focusedBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(10),
                              borderSide:
                                  BorderSide(color: Colors.blue[700]!, width: 2),
                            ),
                          ),
                        ),
                        const SizedBox(height: 16),

                        // ── Password ───────────────────────────────────────
                        TextFormField(
                          controller: _passwordController,
                          validator: _validatePassword,
                          obscureText: _obscurePassword,
                          inputFormatters: [
                            FilteringTextInputFormatter.deny(RegExp(r'\s')),
                            LengthLimitingTextInputFormatter(128),
                          ],
                          decoration: InputDecoration(
                            labelText: 'Password',
                            helperText: 'At least 6 characters, no spaces',
                            prefixIcon: Icon(Icons.lock_outline,
                                color: Colors.blue[700]),
                            suffixIcon: IconButton(
                              icon: Icon(
                                _obscurePassword
                                    ? Icons.visibility_off_outlined
                                    : Icons.visibility_outlined,
                                color: Colors.grey[500],
                                size: 20,
                              ),
                              onPressed: () => setState(
                                  () => _obscurePassword = !_obscurePassword),
                            ),
                            border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(10)),
                            focusedBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(10),
                              borderSide:
                                  BorderSide(color: Colors.blue[700]!, width: 2),
                            ),
                          ),
                          onFieldSubmitted: (_) => _signUp(),
                        ),
                        const SizedBox(height: 28),

                        // ── Button ─────────────────────────────────────────
                        _isLoading
                            ? const Center(child: CircularProgressIndicator())
                            : ElevatedButton(
                                onPressed: _signUp,
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: Colors.blue[700],
                                  foregroundColor: Colors.white,
                                  padding:
                                      const EdgeInsets.symmetric(vertical: 15),
                                  shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(10)),
                                  textStyle: const TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.w600),
                                ),
                                child: const Text('Create Account'),
                              ),
                        const SizedBox(height: 20),

                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text('Already have an account?',
                                style: TextStyle(
                                    color: Colors.grey[600], fontSize: 14)),
                            TextButton(
                              onPressed: () => Navigator.pop(context),
                              style: TextButton.styleFrom(
                                  foregroundColor: Colors.blue[700],
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 6)),
                              child: const Text('Sign In',
                                  style:
                                      TextStyle(fontWeight: FontWeight.w600)),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ),
      ),
    );
  }
}
