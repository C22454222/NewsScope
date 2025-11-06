import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';

class SignInScreen extends StatelessWidget {
  const SignInScreen({super.key});

  Future<UserCredential> _signInWithGoogle() async {
    // Create a GoogleSignIn instance
    final GoogleSignIn googleSignIn = GoogleSignIn(
      scopes: ['email'],
    );

    // Trigger the sign-in flow
    final GoogleSignInAccount? googleUser = await googleSignIn.signIn();

    if (googleUser == null) {
      throw Exception("Google sign-in aborted");
    }

    // Get the authentication details
    final GoogleSignInAuthentication googleAuth = await googleUser.authentication;

    // Build a Firebase credential
    final credential = GoogleAuthProvider.credential(
      idToken: googleAuth.idToken,
      accessToken: googleAuth.accessToken,
    );

    // Sign in to Firebase
    return FirebaseAuth.instance.signInWithCredential(credential);
  }

  Future<UserCredential> _signInWithEmail() async {
    return FirebaseAuth.instance.signInWithEmailAndPassword(
      email: "test@example.com", // replace with form input
      password: "password123",   // replace with form input
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          ElevatedButton(
            onPressed: _signInWithGoogle,
            child: const Text("Sign in with Google"),
          ),
          ElevatedButton(
            onPressed: _signInWithEmail,
            child: const Text("Sign in with Email"),
          ),
        ]),
      ),
    );
  }
}