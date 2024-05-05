import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';

import 'device_state_builder.dart';

class DeviceStatusBar extends StatelessWidget {
  const DeviceStatusBar({
    Key? key,
    required this.connected,
    @required this.deviceName,
    required this.connectHandler,
    required this.disconnectHandler,
    required this.connection,
    this.updateState,
    @required this.updateFile,
    required this.retryUpdate,
    @required this.onRetryChange,
    this.progress = 0,
  }) : super(key: key);

  final AsyncSnapshot connection;
  final ValueNotifier<bool> connected;
  final String? deviceName;

  final void Function() connectHandler;
  final void Function() disconnectHandler;

  final Future<void>? updateState;
  final File? updateFile;
  final bool retryUpdate;
  final void Function(bool value)? onRetryChange;
  final int progress;

  String get _device => deviceName != null ? '"$deviceName"' : "unknown device";

  Widget get _processIndicator => Container(
        width: 25,
        alignment: Alignment.centerRight,
        child: SizedBox(
          child: updateFile != null
              ? FutureBuilder(
                  future: updateFile!.length(),
                  builder: (_, AsyncSnapshot<int> length) {
                    double? percentage;

                    if (length.hasData) {
                      if (progress != 0 && length.data != progress) {
                        percentage = progress / length.data!;
                      }
                    }

                    return CircularProgressIndicator(
                      strokeWidth: 2,
                      value: percentage,
                    );
                  },
                )
              : const CircularProgressIndicator(strokeWidth: 2),
          height: 25,
        ),
      );

  Widget get _connectionIndicator => DeviceStatusBuilder(
        updateState: updateState,
        connected: connected,
        builder: (_, connected, updating) {
          if (updating) {
            return _processIndicator;
          }

          return Container(
            width: 25,
            alignment: Alignment.centerRight,
            child: connected
                ? Icon(Icons.check, color: Colors.lightGreen)
                : Icon(Icons.error_outline, color: Colors.red),
          );
        },
      );

  Widget get _connectionAction => DeviceStatusBuilder(
        connected: connected,
        updateState: updateState,
        builder: (context, bool connected, bool updating) {
          if (updating) {
            return Builder(
              builder: (BuildContext context) => Column(
                children: [
                  Expanded(
                    child: Switch(
                      value: retryUpdate,
                      onChanged: onRetryChange,
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.only(top: 4.0),
                    child: Text(
                      'Retry on\nfailure',
                      style: Theme.of(context).textTheme.overline,
                      textAlign: TextAlign.center,
                    ),
                  ),
                ],
              ),
            );
          }

          return ElevatedButton(
            style: ElevatedButton.styleFrom(
              primary: Colors.white,
            ),
            onPressed: connected ? disconnectHandler : connectHandler,
            child: Text(connected ? 'Disconnect' : 'Retry'),
          );
        },
      );

  Widget get _connectionText => DeviceStatusBuilder(
        connected: connected,
        updateState: updateState,
        builder: (_, bool connected, bool updating) {
          if (updating) {
            return Text('Updating $_device');
          } else if (connected) {
            return Text('Connected');
          }

          return Text('Disconnected');
        },
      );

  Widget get _deviceText => DeviceStatusBuilder(
        connected: connected,
        updateState: updateState,
        builder: (_, bool connected, bool updating) {
          if (updating) {
            return FutureBuilder(
              future: updateFile!.length(),
              builder: (_, AsyncSnapshot<int> length) {
                if (length.hasData) {
                  final estimation = length.data! * 0.8 / 1024 / 1024;

                  return Text(
                    'This should take around ${estimation.ceil()} min',
                  );
                }

                return Text('This will take a few minutes');
              },
            );
          } else if (connected) {
            return Text(
              '${_device[0].toUpperCase() + _device.substring(1)} is good to go',
            );
          }

          return Text('Connection lost to $_device');
        },
      );

  @override
  Widget build(BuildContext context) {
    final bool done = connection.connectionState == ConnectionState.done;
    final bool error = connection.hasError;

    Widget title, subtitle;
    if (done) {
      if (!error) {
        title = _connectionText;
        subtitle = _deviceText;
      } else {
        title = Text(
          'Could not connect',
          style: TextStyle(color: Colors.red),
        );

        subtitle = Text('Tried to reach $_device');
      }
    } else {
      title = Text('Connecting');
      subtitle = Text('Trying to reach $_device');
    }

    return BottomAppBar(
      color: Colors.white,
      child: ConstrainedBox(
        constraints: BoxConstraints(minHeight: 75),
        child: Center(
          heightFactor: 0,
          child: ListTile(
            visualDensity: VisualDensity.comfortable,
            contentPadding: const EdgeInsets.only(
              left: 16,
              right: 16,
              bottom: 4,
              top: 4,
            ),
            leading: done ? _connectionIndicator : _processIndicator,
            trailing: done ? _connectionAction : null,
            title: title,
            subtitle: subtitle,
          ),
        ),
      ),
    );
  }
}
