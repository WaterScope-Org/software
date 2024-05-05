import 'dart:async';
import 'dart:convert';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:location/location.dart';
import 'package:waterscope/bluetooth.dart';
import 'package:waterscope/sample_result_page/result_view.dart';

import 'results_explanations.dart';
class SampleResultPage extends StatelessWidget {
  const SampleResultPage({
    Key? key,
    required this.sampleId,
    required this.incubationTime,
    required this.sampleVolume,
    this.comment,
    required this.sampleLocation,
    this.instruction,
    this.overrideSample = false,
    this.position,
  }) : super(key: key);

  final bool overrideSample;
  final int sampleId;
  final int sampleVolume;
  final String? comment;
  final String sampleLocation;
  final Duration incubationTime;
  final Future<BluetoothInstruction>? instruction;
  final LocationData? position;

  Future<Map<String, dynamic>> get resultConfigs async {
    return jsonDecode(
      await rootBundle.loadString('config/result_configs.json'),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.indigo[900],
        title: Text('Results Display'),
      ),
      // Don't use a ListView as the result view needs to be alive at all time
      body: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ResultView(
              overrideSample: overrideSample,
              instruction: instruction,
              sampleId: sampleId,
              configs: resultConfigs,
              incubationTime: incubationTime,
              sampleVolume: sampleVolume,
              comment: comment,
              sampleLocation: sampleLocation,
              position: position,
            ),
            ResultsExplanations(configs: resultConfigs),
          ],
        ),
      ),
    );
  }
}
