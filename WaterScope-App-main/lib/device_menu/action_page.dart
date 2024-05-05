import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:crypto/crypto.dart';
import 'package:file_picker/file_picker.dart';
import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import 'package:waterscope/bluetooth.dart';
import 'package:waterscope/result_history/result_history.dart';
import 'package:waterscope/sample_form.dart';
import 'package:waterscope/wifi_settings.dart';

import 'device_action_button.dart';
import 'device_status_bar.dart';

class ActionPage extends StatefulWidget {
  final BluetoothDevice server;

  const ActionPage({required this.server});

  @override
  _ActionPageState createState() => new _ActionPageState();
}

class _ActionPageState extends State<ActionPage> {
  Future<BluetoothConnection>? _connection;

  Stream<Uint8List>? _bluetoothStream;
  ValueNotifier<bool> _connected = ValueNotifier(false);

  StreamSubscription? _bluetoothSubscription;
  BluetoothInstruction? _diagnostics;

  File? _updateFile;
  Completer<void>? _update;
  bool _retryUpdate = true;

  int? _updateProgress;

  @override
  void initState() {
    super.initState();
    _connect();
  }

  @override
  void dispose() {
    _diagnostics?.cancel();
    _bluetoothSubscription?.cancel();
    _connection?.then((BluetoothConnection connection) {
      if (connection.isConnected) {
        connection.close();
      }
    });
    super.dispose();
  }

  void _saveDiagnostics(List<int>? data) async {
    Map<String, dynamic> entry = json.decode(utf8.decode(data ?? []));

    final FirebaseAnalytics analytics = FirebaseAnalytics.instance;
    analytics.setUserProperty(
      name: 'device',
      value: widget.server.address,
    );
    analytics.logEvent(name: 'diagnostics', parameters: entry);

    entry.putIfAbsent('diagnosticTime', () => DateTime.now());

    final doc = FirebaseFirestore.instance
        .collection('devices')
        .doc(widget.server.address);

    try {
      await doc.update(entry);
    } catch (e) {
      if (e is FirebaseException && e.code == 'not-found') {
        doc.set(entry);
      } else {
        throw e;
      }
    }
  }

  void _sendUpdate() async {
    setState(() => _updateProgress = 0);

    final Uint8List update = await _updateFile!.readAsBytes();

    BluetoothConnection connection = await _connection!;
    BluetoothInstruction? instruction;

    try {
      try {
        await _diagnostics?.result;
      } catch (_) {}

      instruction = BluetoothInstruction.sendFile(
        connection,
        _bluetoothStream!,
        'update',
        update,
      );

      instruction.processedPayload.addListener(
        () => setState(() {
          _updateProgress = instruction?.processedPayload.value;
        }),
      );

      await instruction.result;

      instruction = BluetoothInstruction.receive(
        connection,
        _bluetoothStream!,
        'update',
        timeout: null,
      );

      final status =
          json.decode(utf8.decode(await instruction.result ?? []))['status'];

      if (status == 'success') {
        _update?.complete();
      } else {
        _update?.completeError(Exception(status));
      }
    } catch (e) {
      // trying old update method
      if (_connected.value && instruction?.processedPayload.value == null) {
        // ignore: close_sinks
        final output = (await _connection)!.output;

        output.add(Uint8List.fromList(utf8.encode(
          '\r\nupdate=${sha1.convert(update)},length=${update.lengthInBytes}\r\n',
        )));

        output.add(update);
      } else {
        _update?.completeError(e);
      }
    }
  }

  void _oldUpdateListener(Uint8List data) {
    Bluetooth.streamListener(data, (String message, _) async {
      if (message.startsWith('received=')) {
        setState(
          () {
            try {
              _updateProgress = int.parse(
                Bluetooth.extractFromResult(message, 'received') ?? '',
              );
            } catch (e) {
              if (!(e is FormatException)) {
                throw e;
              }
            }
          },
        );
      } else if (message.startsWith('update=')) {
        switch (Bluetooth.extractFromResult(message, 'update')) {
          case 'ok':
            _update?.complete();
            break;
          case 'failure':
            _update?.completeError(Exception('Could not apply update'));
            break;
          case 'transmission_error':
            if (_retryUpdate) {
              _sendUpdate();
            } else {
              _update?.completeError('File transmission failed');
            }
            break;
        }
      }
    });
  }

  Future<void> _startUpdate(BuildContext context) async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['zip'],
    );

    if (result != null) {
      setState(() {
        _update = Completer();

        _update!.future.then(
          (_) => ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Update successful')),
          ),
        );

        _update!.future.catchError((exception, stack) {
          final text = exception is TimeoutException
              ? 'Update file transmission timed out.'
              : exception.toString();

          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(text)),
          );

          FirebaseCrashlytics.instance.recordError(exception, stack);
        });

        _updateFile = File(result.files.single.path!);

        _sendUpdate();
      });
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('No file selected')),
      );
    }
  }

  void _sendSample(BluetoothConnection connection) {
    Navigator.of(context).push(
      MaterialPageRoute(
        settings: RouteSettings(name: 'sample-results'),
        builder: (context) {
          return Bluetooth(
            device: widget.server,
            stream: _bluetoothStream!,
            connection: connection,
            connected: _connected,
            child: SampleForm(),
          );
        },
      ),
    );
  }

  void _lookupResults(BluetoothConnection? connection) {
    Navigator.of(context).push(
      MaterialPageRoute(
        settings: RouteSettings(name: 'result-history'),
        builder: (context) {
          return Bluetooth(
            device: widget.server,
            stream: _bluetoothStream,
            connection: connection,
            connected: _connected,
            child: ResultHistory(),
          );
        },
      ),
    );
  }

  void _wifiSettings(BluetoothConnection connection, scaffold) async {
    BluetoothInstruction? instruction = await showDialog(
      context: context,
      builder: (_) => Bluetooth(
        device: widget.server,
        stream: _bluetoothStream,
        connection: connection,
        connected: _connected,
        child: WifiSettings(),
      ),
    );

    if (instruction != null) {
      instruction.result.then((_) {
        scaffold.showSnackBar(SnackBar(
          content: Text('Added Wi-Fi successfully'),
        ));
      });

      instruction.result.catchError((exception, stack) {
        scaffold.showSnackBar(SnackBar(
          content: Text('Failed to add Wi-Fi.'),
        ));

        FirebaseCrashlytics.instance.recordError(exception, stack);
      });
    } else {
      scaffold.showSnackBar(SnackBar(
        content: Text('Connection to WaterScope device lost.'),
      ));
    }
  }

  void _disconnect() async {
    BluetoothConnection? connection = await this._connection;
    await connection?.close();

    _connected.value = false;
    connection?.dispose();
  }

  void _connect() {
    _connection = BluetoothConnection.toAddress(widget.server.address);

    _connection!.then((BluetoothConnection connection) async {
      _connected.value = true;

      if (mounted) {
        setState(
            () => _bluetoothStream = connection.input!.asBroadcastStream());
      } else {
        _bluetoothStream = connection.input!.asBroadcastStream();
      }

      if (_bluetoothSubscription == null) {
        _bluetoothSubscription = _bluetoothStream!.listen(_oldUpdateListener)
          ..onDone(() {
            if (!(_update?.isCompleted ?? true)) {
              _update!.completeError(Exception('Connection lost'));
            }
            _connected.value = false;

            _bluetoothSubscription = null;
          });

        await Future.delayed(const Duration(seconds: 2));

        if (_connected.value) {
          _diagnostics = BluetoothInstruction.send(
            connection,
            _bluetoothStream!,
            'diagnostics',
          );

          _diagnostics!.result.then(_saveDiagnostics);
        }
      }
    });
  }

  @override
  Widget build(BuildContext context) => FutureBuilder(
        future: _connection,
        builder: (_, AsyncSnapshot<BluetoothConnection> connection) {
          return Scaffold(
            appBar: AppBar(title: Text('Device Menu')),
            bottomNavigationBar: DeviceStatusBar(
              connectHandler: () => setState(() => _connect()),
              disconnectHandler: _disconnect,
              connected: _connected,
              connection: connection,
              deviceName: widget.server.name,
              updateState: _update?.future,
              updateFile: _updateFile,
              retryUpdate: _retryUpdate,
              progress: _updateProgress ?? 0,
              onRetryChange: (bool value) => setState(
                () => _retryUpdate = value,
              ),
            ),
            body: SafeArea(
              child: Stack(
                children: [
                  ListView(
                    padding: const EdgeInsets.only(top: 16, bottom: 32),
                    children: [
                      DeviceActionButton(
                        onTap: () => _sendSample(connection.data!),
                        text: 'Sample submission',
                        infoText: 'Analyse a new sample',
                        icon: Icons.send,
                        connected: _connected,
                        updateState: _update?.future,
                      ),
                      DeviceActionButton(
                        connectionRequired: false,
                        onTap: () => _lookupResults(connection.data),
                        text: 'Sample History',
                        infoText: 'Look up results',
                        icon: Icons.history,
                        connected: _connected,
                        updateState: _update?.future,
                      ),
                      Builder(
                        builder: (context) => DeviceActionButton(
                          onTap: () => _wifiSettings(
                            connection.data!,
                            Scaffold.of(context),
                          ),
                          text: 'Add Wi-Fi Network',
                          infoText: 'Add Water-Scope device to a Wi-Fi',
                          icon: Icons.wifi,
                          connected: _connected,
                          updateState: _update?.future,
                        ),
                      ),
                      Builder(
                        builder: (context) => DeviceActionButton(
                          onTap: () => _startUpdate(context),
                          text: 'Update Software',
                          infoText: 'Send and apply an update '
                              'to the Water-Scope device',
                          icon: Icons.upgrade,
                          connected: _connected,
                          updateState: _update?.future,
                        ),
                      )
                    ],
                  )
                ],
              ),
            ),
          );
        },
      );
}
