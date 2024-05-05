import 'package:flutter/material.dart';

import 'result_list.dart';

class ResultHistory extends StatelessWidget {
  ResultHistory({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Result History')),
      body: SafeArea(child: ResultList()),
    );
  }
}
