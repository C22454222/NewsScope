import 'package:flutter/material.dart';
import '../services/api_service.dart';

class FeedScreen extends StatefulWidget {
  const FeedScreen({super.key});

  @override
  State<FeedScreen> createState() => _FeedScreenState();
}

class _FeedScreenState extends State<FeedScreen> {
  final api = ApiService();
  String data = "Loading...";

  @override
  void initState() {
    super.initState();
    api.getStories().then((value) {
      setState(() {
        data = value; // show backend response
      });
    }).catchError((err) {
      setState(() {
        data = "Error: $err";
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("NewsScope Feed")),
      body: Center(child: Text(data)),
    );
  }
}