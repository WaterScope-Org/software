import 'package:flutter/material.dart';

class Instructions extends StatefulWidget {
  @override
  _InstructionsState createState() => _InstructionsState();
}

class _InstructionsState extends State<Instructions> {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        primaryColor: Colors.indigo[900],
        colorScheme: ColorScheme.fromSwatch().copyWith(
          secondary: Colors.indigo[900],
        ),
        textSelectionTheme: TextSelectionThemeData(
          selectionColor: Colors.indigo[900],
        ),
      ),
      home: Scaffold(
        appBar: AppBar(
          backgroundColor: Colors.indigo[900],
          title: const Text('WaterScope Instructions Manual'),
        ),
        body: Center(
          child: Text('PDF Viewer is unavailable.'),
        ),
      ),
    );
  }
}
