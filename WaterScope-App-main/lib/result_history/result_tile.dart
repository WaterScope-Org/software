import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:timer_builder/timer_builder.dart';
import 'package:waterscope/bluetooth.dart';
import 'package:waterscope/sample_result_page/sample_result_page.dart';

class ResultTile extends StatelessWidget {
  const ResultTile({
    Key? key,
    required this.entry,
    required this.id,
  }) : super(key: key);

  final Map<String, dynamic> entry;
  final int id;

  Future<BluetoothInstruction> querySample(bluetooth, id) async {
    final payload = Uint8List.fromList(utf8.encode(json.encode({
      'action': 'get',
      'id': id,
    })));

    BluetoothInstruction instruction = BluetoothInstruction.send(
      bluetooth.connection,
      bluetooth.stream,
      'sample',
      payload: payload,
    );

    try {
      await instruction.result;
    } catch (e) {
      if (e is TimeoutException) {
        instruction = BluetoothInstruction.send(
          bluetooth.connection,
          bluetooth.stream,
          'sample',
          payload: payload,
        );
      } else {
        throw e;
      }
    }

    return instruction;
  }

  void showDetails(BuildContext context, int id) {
    Bluetooth bluetooth = Bluetooth.of(context)!;

    Future<BluetoothInstruction>? instruction;

    if (bluetooth.connected.value) {
      instruction = querySample(bluetooth, id);
    }

    Navigator.push(
      context,
      MaterialPageRoute(
        settings: RouteSettings(name: 'sample-results'),
        builder: (_) {
          Bluetooth bluetooth = Bluetooth.of(context)!;

          return Bluetooth(
            device: bluetooth.device,
            stream: bluetooth.stream,
            connection: bluetooth.connection,
            connected: bluetooth.connected,
            child: SampleResultPage(
              sampleId: id,
              instruction: instruction,
              comment: entry['comment'],
              incubationTime: entry['time'],
              sampleVolume: entry['sample_volume'],
              sampleLocation: entry['location'],
            ),
          );
        },
      ),
    );
  }

  String getDurationString(Duration duration) {
    int hours = duration.inHours;

    if (hours < 0) {
      hours = 0;
    }

    dynamic minutes = duration.inMinutes - hours * 60;

    if (minutes < 0) {
      minutes = 0;
    }

    minutes = minutes.toString();
    minutes = minutes.length > 1 ? minutes : '0$minutes';

    return '$hours:$minutes';
  }

  Widget createDisplay(
    BuildContext context,
    CrossAxisAlignment alignment,
    String name,
    String info,
  ) {
    final TextTheme textTheme = Theme.of(context).textTheme;

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      crossAxisAlignment: alignment,
      children: [
        Padding(
          padding: const EdgeInsets.only(bottom: 2),
          child: Text(
            name,
            style: textTheme.caption!.copyWith(
              color: Theme.of(context).colorScheme.primary,
            ),
          ),
        ),
        Text(
          info,
          style: textTheme.overline!.copyWith(
            fontSize: (textTheme.bodyText1!.fontSize ?? 1) * 1.2,
          ),
        ),
      ],
    );
  }

  Widget createCountDisplay(BuildContext context, int eColi, int otherColi) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          padding: const EdgeInsets.only(right: 8),
          margin: const EdgeInsets.only(right: 8),
          decoration: BoxDecoration(
            border: Border(
              right: BorderSide(
                color: Theme.of(context).colorScheme.secondary,
                width: 1,
              ),
            ),
          ),
          child: createDisplay(
            context,
            CrossAxisAlignment.end,
            'E. Coli',
            eColi.toString(),
          ),
        ),
        createDisplay(
          context,
          CrossAxisAlignment.start,
          'Other',
          otherColi.toString(),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final location = entry['location'];
    final Duration time = entry['time'];
    final String? comment = entry['comment'];
    final eColi = entry['eColi'];
    final otherColi = entry['otherColi'];
    final DateTime? submissionTime = entry['submissionTime'];
    final textTheme = Theme.of(context).textTheme;

    String idText = id.toString();

    while (idText.length < 4) {
      idText = '0' + idText;
    }

    Widget trailing;

    if (submissionTime?.add(time).isAfter(DateTime.now()) ?? false) {
      trailing = TimerBuilder.periodic(
        Duration(minutes: 1),
        builder: (_) {
          final leftTime = submissionTime!.add(time).difference(DateTime.now());

          return Padding(
            padding: const EdgeInsets.only(left: 4.0),
            child: createDisplay(
              context,
              CrossAxisAlignment.center,
              'Time left',
              getDurationString(leftTime),
            ),
          );
        },
      );
    } else if (eColi != null && otherColi != null) {
      trailing = createCountDisplay(context, eColi, otherColi);
    } else {
      trailing = ValueListenableBuilder(
        valueListenable: Bluetooth.of(context)!.connected,
        builder: (_, bool connected, __) => Text(
          connected ? 'Click to check details' : 'Not available',
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.caption,
        ),
      );
    }

    return ListTile(
      key: Key(idText),
      onTap: () => showDetails(context, id),
      contentPadding: const EdgeInsets.only(
        right: 16,
        left: 16,
        top: 8,
        bottom: 8,
      ),
      leading: Container(
        width: 60,
        alignment: Alignment.centerRight,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Padding(
              padding: const EdgeInsets.only(bottom: 1.0),
              child: Text('Sample ID', style: textTheme.overline),
            ),
            FittedBox(
              fit: BoxFit.scaleDown,
              child: Text(
                idText,
                style: textTheme.overline!.copyWith(
                  fontSize: (textTheme.bodyText2!.fontSize ?? 1) * 1.5,
                  height: 1,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ],
        ),
      ),
      title: Text('$location - ${getDurationString(time)}'),
      subtitle: comment?.isNotEmpty ?? false ? Text(comment!) : null,
      trailing: Container(
        width: 100,
        alignment: Alignment.center,
        child: Container(
          width: 100,
          height: 40,
          alignment: Alignment.center,
          child: trailing,
        ),
      ),
    );
  }
}
