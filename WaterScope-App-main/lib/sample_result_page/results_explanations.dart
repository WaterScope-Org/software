import 'package:flutter/material.dart';

import 'result_config.dart';

class ResultsExplanations extends StatelessWidget {

  ResultsExplanations({required this.configs});

  final Future<Map<String, dynamic>> configs;

  Widget createInfoWidget(Map<String, dynamic> json) => Builder(
        builder: (BuildContext context) {
          final ResultConfig config = ResultConfig.fromJson(json);

          final baseStyle = Theme.of(context).textTheme.headline6!;

          return Padding(
            padding: const EdgeInsets.only(top: 32.0, bottom: 16.0),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  config.title,
                  style: TextStyle(
                    color: config.color,
                    fontSize: baseStyle.fontSize,
                    fontWeight: baseStyle.fontWeight,
                    fontStyle: baseStyle.fontStyle,
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(top: 8.0),
                  child: Text(config.explanation, textAlign: TextAlign.center),
                )
              ],
            ),
          );
        },
      );

  Widget createResultExplanations(Map<String, dynamic> config) {
    List<Widget> resultExplanations = [];

    config.values.forEach((entry) {
      if (entry is List) {
        resultExplanations.addAll(entry.map((json) => createInfoWidget(json)));
      } else {
        resultExplanations.add(createInfoWidget(entry));
      }
    });

    return Column(children: resultExplanations);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(top: 16),
      padding: const EdgeInsets.only(
        top: 32,
        right: 16,
        left: 16,
        bottom: 64,
      ),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border(
          top: BorderSide(color: Colors.black38, width: 0.5),
        ),
      ),
      child: Column(
        children: [
          Text(
            'What results can be shown?',
            style: Theme.of(context).textTheme.headline5,
          ),
          FutureBuilder(
            future: configs,
            builder: (_, AsyncSnapshot<Map<String, dynamic>> config) {
              if (config.hasData) {
                return createResultExplanations(config.data!);
              }

              return Padding(
                padding: const EdgeInsets.all(32),
                child: const CircularProgressIndicator(),
              );
            },
          ),
        ],
      ),
    );
  }
}
