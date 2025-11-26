import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';

class HttpRequestScreen extends StatefulWidget {
  const HttpRequestScreen({super.key});

  @override
  State<HttpRequestScreen> createState() => _HttpRequestScreenState();
}

class _HttpRequestScreenState extends State<HttpRequestScreen> {
  String result = "Нажми кнопку";

  Future<void> sendRequest() async {
    try {
        final url = Uri.parse("http://lk.stu.lipetsk.ru/");
        final response = await http.get(
                url,
                headers: {
                'User-Agent': 'FlutterApp/1.0',
                'Accept': 'text/html',
                },
                );

      if (response.statusCode == 200) {
        final data = response.body;
        setState(() => result = data.toString());
      } else {
        setState(() => result = "Ошибка: ${response.statusCode}");
      }
    } catch (e) {
      setState(() => result = "Ошибка: $e");
    }
  }

  Future<void> goToLink() async {
    final uri = Uri.parse("https://youtu.be/taBKBFmTfYU?si=JScPyyCAaPl2lqW5");

    if (await canLaunchUrl(uri)) {
      await launchUrl(uri);
    } else {
      setState(() => result = "Не удалось открыть ссылку");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("HTTP + Link")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            ElevatedButton(
              onPressed: sendRequest,
              child: const Text("Отправить HTTP-запрос"),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: goToLink,
              child: const Text("Перейти по ссылке"),
            ),
            const SizedBox(height: 20),
            Text(result),
          ],
        ),
      ),
    );
  }
}
