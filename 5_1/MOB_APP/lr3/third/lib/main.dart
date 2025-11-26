import 'package:flutter/material.dart';
import 'http.dart';
import 'sensors.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
    const MyApp({super.key});

    @override
        Widget build(BuildContext context) {
            return const MaterialApp(
                    home: MainScreen(),
                    );
        }
}

class MainScreen extends StatefulWidget {
    const MainScreen({super.key});

    @override
        State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
    int index = 0;

    final screens = const [
        Center(child: Text("Home")),

        CStandardsScreen(),
        CppStandardsScreen(),

        HttpRequestScreen(),
        HomeScreen(),
    ];

    @override
        Widget build(BuildContext context) {
            return Scaffold(
                    body: screens[index],
                    bottomNavigationBar: BottomNavigationBar(
                        currentIndex: index,
                        type: BottomNavigationBarType.fixed,
                        onTap: (int i) => setState(() => index = i),
                        items: const [
                        BottomNavigationBarItem(
                            icon: Icon(Icons.home),
                            label: "Home",
                            ),
                        BottomNavigationBarItem(
                            icon: Icon(Icons.code),
                            label: "C",
                            ),
                        BottomNavigationBarItem(
                                icon: Icon(Icons.build),
                                label: "C++",
                                ),
                        BottomNavigationBarItem(
                                icon: Icon(Icons.wifi),
                                label: "HTTP",
                                ),
                        BottomNavigationBarItem(
                                icon: Icon(Icons.phone),
                                label: "Sensors",
                                ),
                        ],
                        ),
                        );
        }
}

class CStandardsScreen extends StatelessWidget {
    const CStandardsScreen({super.key});

    final List<String> standards = const [
        "C89 / ANSI C",
        "C90 (ISO/IEC 9899:1990)",
        "C95 (Amendment 1)",
        "C99 (ISO/IEC 9899:1999)",
        "C11 (ISO/IEC 9899:2011)",
        "C17 (ISO/IEC 9899:2018)",
        "C23 (ISO/IEC 9899:2023)",
    ];

    @override
        Widget build(BuildContext context) {
            return ListView.builder(
                    itemCount: standards.length,
                    itemBuilder: (context, i) {
                    return ListTile(
                            title: Text(standards[i]),
                            );
                    },
                    );
        }
}

class CppStandardsScreen extends StatelessWidget {
    const CppStandardsScreen({super.key});

    final List<String> standards = const [
        "C++98",
        "C++03",
        "C++11",
        "C++14",
        "C++17",
        "C++20",
        "C++23",
        "C++26",
    ];

    @override
        Widget build(BuildContext context) {
            return ListView.builder(
                    itemCount: standards.length,
                    itemBuilder: (context, i) {
                    return ListTile(
                            title: Text(standards[i]),
                            );
                    },
                    );
        }
}
