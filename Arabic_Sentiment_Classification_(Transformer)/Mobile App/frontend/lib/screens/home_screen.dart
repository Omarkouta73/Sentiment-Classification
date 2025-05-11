import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../services/app_api_services.dart';
import '../widgets/sentiment_result_widget.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController _controller = TextEditingController();
  Map<String, dynamic>? _result;
  bool _loading = false;

  Future<void> _predict(String text) async {
    setState(() {
      _loading = true;
      _result = null;
    });

    try {
      final result = await ApiService.predict(text);
      setState(() {
        _result = result;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _result = {'error': e.toString()};
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Sentiment Analyzer')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _controller,
              maxLines: 3,
              decoration: InputDecoration(labelText: 'Enter text'),
            ),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: () => _predict(_controller.text),
              child: Text('Analyze'),
            ),
            SizedBox(height: 24),
            if (_loading) CircularProgressIndicator(),
            if (_result != null) SentimentResultWidget(result: _result!),
          ],
        ),
      ),
    );
  }
}
