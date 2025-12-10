import 'package:flutter/material.dart';
import 'package:sensors_plus/sensors_plus.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  double ax = 0, ay = 0, az = 0;
  double gx = 0, gy = 0, gz = 0;

  @override
  void initState() {
    super.initState();

    accelerometerEvents.listen((AccelerometerEvent event) {
      setState(() {
        ax = event.x;
        ay = event.y;
        az = event.z;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Home")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("Акселерометр (m/s²):", style: TextStyle(fontWeight: FontWeight.bold)),
            Text("X: ${ax.toStringAsFixed(2)}  Y: ${ay.toStringAsFixed(2)}  Z: ${az.toStringAsFixed(2)}"),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}
