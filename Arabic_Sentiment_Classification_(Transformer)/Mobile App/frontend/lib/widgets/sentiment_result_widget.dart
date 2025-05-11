import 'package:flutter/material.dart';

class SentimentResultWidget extends StatelessWidget {
  final Map<String, dynamic> result;

  const SentimentResultWidget({Key? key, required this.result})
    : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (result.containsKey('error')) {
      return Text(
        'Error: ${result['error']}',
        style: TextStyle(color: Colors.red),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Text: ${result['text']}',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        SizedBox(height: 8),
        Text('Sentiment: ${result['sentiment']['label']}'),
        Text('Score: ${result['sentiment']['score']}'),
        // Text('Scores:'),
        // Text(' - Negative: ${result['scores']['negative']}'),
        // Text(' - Neutral: ${result['scores']['neutral']}'),
        // Text(' - Positive: ${result['scores']['positive']}'),
      ],
    );
  }
}
