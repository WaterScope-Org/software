import 'dart:async';
import 'dart:convert';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:pull_to_refresh/pull_to_refresh.dart';
import 'package:waterscope/bluetooth.dart';

import 'result_tile.dart';

class ResultList extends StatefulWidget {
  ResultList({Key? key}) : super(key: key);

  @override
  _ResultListState createState() => _ResultListState();
}

class _ResultListState extends State<ResultList> {
  RefreshController? _controller;
  StreamSubscription? _firestore;

  BluetoothInstruction? _bluetooth;

  Map<int, Map<String, dynamic>> _results = {};

  @override
  void initState() {
    super.initState();

    SchedulerBinding.instance?.addPostFrameCallback((_) {
      Bluetooth bluetooth = Bluetooth.of(context)!;

      setState(() {
        _controller = RefreshController(
          initialRefresh: bluetooth.connected.value,
        );
      });

      _firestore = FirebaseFirestore.instance
          .collection('devices')
          .doc(bluetooth.device.address)
          .collection('samples')
          .snapshots()
          .listen(
        (QuerySnapshot entries) {
          entries.docs.forEach((QueryDocumentSnapshot entry) {
            updateResult(
              int.parse(entry.id),
              location: entry['location'],
              eColi: getDocumentEntry(entry, 'eColiform'),
              otherColi: getDocumentEntry(entry, 'otherColiform'),
              comment: getDocumentEntry(entry, 'comment'),
              time: Duration(minutes: entry['incubationTime']),
              submissionTime: DateTime.fromMillisecondsSinceEpoch(
                (entry['submissionTime'] as Timestamp).seconds * 1000,
              ),
            );
          });
        },
      );
    });
  }

  @override
  void dispose() {
    _firestore?.cancel();
    _bluetooth?.cancel();
    super.dispose();
  }

  T? getDocumentEntry<T>(QueryDocumentSnapshot document, String field) {
    try {
      return document.get(field);
    } catch (error) {
      if (!(error is StateError)) {
        throw error;
      }
    }

    return null;
  }

  void updateResult(
    int id, {
    String? location,
    int? eColi,
    int? otherColi,
    Duration? time,
    String? comment,
    DateTime? submissionTime,
  }) {
    final result = {
      'location': location,
      'eColi': eColi,
      'otherColi': otherColi,
      'time': time,
      'comment': comment,
      'submissionTime': submissionTime,
    };

    final update = () {
      _results.update(id, (_) => result, ifAbsent: () => result);
    };

    if (mounted) {
      setState(update);
    } else {
      update();
    }
  }

  void onRefreshError({String? msg, BuildContext? c}) {
    msg = msg ?? 'Failed to fetch result history.';

    ScaffoldMessenger.of(c ?? context)
        .showSnackBar(SnackBar(content: Text(msg)));
    _controller?.refreshFailed();
  }

  Future<void> refresh() async {
    final BuildContext c = context;
    final bluetooth = Bluetooth.of(c)!;

    if (bluetooth.connected.value) {
      _bluetooth = BluetoothInstruction.send(
        bluetooth.connection!,
        bluetooth.stream!,
        'history',
      );

      _bluetooth!.result.then((results) {
        (json.decode(utf8.decode(results ?? [])) as List).forEach((result) {
          updateResult(
            result['id'],
            eColi: result['eColiform'],
            otherColi: result['otherColiform'],
            location: result['location'],
            time: Duration(minutes: result['time']),
            comment: result['comment'],
          );
        });
        _controller?.refreshCompleted();
      }).catchError((error, stack) {
        onRefreshError(msg: 'Refreshing timed out', c: c);
        FirebaseCrashlytics.instance.recordError(error, stack);
      }, test: (e) {
        return e is TimeoutException;
      }).catchError((error, stack) {
        onRefreshError();
        FirebaseCrashlytics.instance.recordError(error, stack);
      }, test: (e) {
        return !(e is TimeoutException);
      });
    } else {
      onRefreshError(msg: 'Not connected to WaterScope device');
    }
  }

  List<int> get _sortedIDs => _results.keys.toList()..sort((a, b) => a - b);

  @override
  Widget build(BuildContext context) {
    if (_controller == null) {
      return Center(child: const CircularProgressIndicator());
    } else {
      return SmartRefresher(
        controller: _controller!,
        onRefresh: refresh,
        child: ListView.builder(
          padding: const EdgeInsets.only(bottom: 64, top: 32),
          itemCount: _results.length,
          itemBuilder: (BuildContext context, int index) {
            final id = _sortedIDs.elementAt(index);
            final entry = _results[id];

            if (entry != null) {
              return ResultTile(id: id, entry: entry);
            } else {
              return Container(height: 0, width: 0);
            }
          },
        ),
      );
    }
  }
}
