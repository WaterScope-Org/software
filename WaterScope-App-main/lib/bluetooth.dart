import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';

class BluetoothInstruction {
  StreamSubscription? _subscription;
  Timer? _timeout;
  BluetoothConnection _connection;
  BluetoothInstruction? _initializer;

  Map<String, dynamic>? _header;
  List<int> _payload = [];

  late Future<List<int>?> result;
  final ValueNotifier<int?> payloadSize = ValueNotifier(null);
  final ValueNotifier<int?> processedPayload = ValueNotifier(null);

  BluetoothInstruction.send(
    BluetoothConnection connection,
    Stream<Uint8List> stream,
    String name, {
    Uint8List? payload,
    Duration timeout = const Duration(minutes: 1),
  }) : _connection = connection {
    final int id = ++_id;

    final Uint8List instruction = _encode({
      'id': id,
      'instruction': name,
      'payload': _calculatePayload(payload),
    });

    final Completer<List<int>> completer = Completer();

    _subscription = stream.listen((data) {
      if (_header == null) {
        Bluetooth.streamListener(data, (message, raw) {
          _readHeader(
            message,
            completer,
            (response) => response['id'] == id,
            () {
              _timeout?.cancel();
              _timeout = _createTimeout(completer, name, timeout: timeout);

              _connection.output.add(payload!);
            },
          );
        });
      } else {
        _readPayload(
          data,
          completer,
          () {
            _timeout = _createTimeout(completer, name, timeout: timeout);

            try {
              _send(completer, instruction, payload);
            } catch (e, stack) {
              completer.completeError(e, stack);
            }
          },
          () => _createTimeout(
            completer,
            name,
            timeout: const Duration(seconds: 5),
          ),
        );
      }
    });

    _timeout = _createTimeout(completer, name, timeout: timeout);

    try {
      _send(completer, instruction, payload);
    } catch (e, stack) {
      completer.completeError(e, stack);
    }

    result = completer.future;
    result.then((_) => _timeout?.cancel());
  }

  BluetoothInstruction.receive(
    BluetoothConnection connection,
    Stream<Uint8List> stream,
    String name, {
    Function? check,
    Duration? timeout = const Duration(minutes: 1),
  }) : _connection = connection {
    final Completer<List<int>> completer = Completer();

    int retryCount = 0;
    const maxRetries = 5;

    _timeout = _createTimeout(completer, name, timeout: timeout);

    _subscription = stream.listen((data) {
      if (_header == null) {
        Bluetooth.streamListener(data, (message, raw) {
          _readHeader(
            message,
            completer,
            (response) => response['instruction'] == name,
          );
        });
      } else {
        final id = _header!['id'];

        _readPayload(
          data,
          completer,
          () {
            String status;

            if (retryCount >= maxRetries) {
              status = 'failed';
              cancel();
              completer.completeError(Exception(""));
            } else {
              _timeout = _createTimeout(
                completer,
                name,
                timeout: timeout,
              );
              status = 'invalid';
            }

            try {
              _send(
                completer,
                _encode({
                  'id': id,
                  'instruction': name,
                  'status': status,
                  'payload': {
                    'size': 0,
                    'checksum': null,
                  }
                }),
              );
            } catch (e, stack) {
              completer.completeError(e, stack);
            }
          },
          (init) {
            if (timeout == null) {
              return null;
            }

            return _createTimeout(
              completer,
              name,
              timeout: (init ?? false) ? const Duration(seconds: 5) : timeout,
            );
          },
          check,
        );
      }
    });

    result = completer.future;

    result.then((_) {
      _timeout?.cancel();

      try {
        _send(
          completer,
          _encode({
            'id': _header!['id'],
            'instruction': name,
            'status': 'ok',
            'payload': {
              'size': 0,
              'checksum': null,
            }
          }),
        );
      } catch (e, stack) {
        completer.completeError(e, stack);
      }
    });
  }

  BluetoothInstruction.sendFile(
    BluetoothConnection connection,
    Stream<Uint8List> stream,
    String filename,
    Uint8List file, {
    Duration timeout = const Duration(minutes: 1),
  }) : _connection = connection {
    final Completer<List<int>?> completer = Completer();

    const packageSize = const int.fromEnvironment(
          'PACKAGE_SIZE',
          defaultValue: 50,
        ) *
        1024; // in kilobytes

    List<String> hashes = [];

    _forEachFileChunk(
      file,
      (chunk, _) => hashes.add(sha1.convert(chunk).toString()),
    ).then((_) {
      _initializer = BluetoothInstruction.send(
        connection,
        stream,
        filename,
        payload: _encode({
          'size': file.length,
          'packageSize': packageSize,
          'hashes': hashes,
          'type': filename,
        }),
      );

      _initializer?.result.catchError((error, stack) {
        completer.completeError(error, stack);
      });

      _initializer?.result.then((_) async {
        processedPayload.value = 0;

        try {
          await _forEachFileChunk(file, (Uint8List chunk, part) async {
            Completer<void> chunkCompletion = Completer();

            _subscription = stream.listen(
              (data) async {
                if (_header == null) {
                  Bluetooth.streamListener(data, (message, raw) {
                    _header = json.decode(message);
                  });
                } else {
                  _payload.addAll(data);
                  _timeout?.cancel();

                  if (_payload.length == _header!['payload']['size']) {
                    Map file = json.decode(utf8.decode(_payload));

                    if (file['part'] == part && file['type'] == filename) {
                      final status = _header!['status'];

                      if (status == 'ok') {
                        processedPayload.value =
                            packageSize + (processedPayload.value ?? 0);

                        _subscription?.cancel();

                        chunkCompletion.complete();
                      } else if (status == 'invalid') {
                        _connection.output.add(chunk);
                        await _connection.output.allSent;

                        _timeout = _createTimeout(
                          chunkCompletion,
                          filename,
                          timeout: timeout,
                        );
                      } else if (status == 'failed') {
                        final exception = Exception(
                          'Transmitting file $filename failed',
                        );

                        chunkCompletion.completeError(exception);
                        throw exception;
                      }
                    }

                    _header = null;
                    _payload.clear();
                  } else {
                    _timeout = _createTimeout(
                      completer,
                      filename,
                      timeout: const Duration(
                        seconds: 5,
                      ),
                    );
                  }
                }
              },
            );

            _connection.output.add(chunk);
            await _connection.output.allSent;

            _timeout = _createTimeout(
              chunkCompletion,
              filename,
              timeout: timeout,
            );

            return chunkCompletion.future;
          });

          completer.complete();
        } catch (e, stack) {
          completer.completeError(e, stack);
        }
      });
    });

    result = completer.future;
    result.then((_) => _timeout?.cancel());
  }

  BluetoothInstruction.receiveFile(
    BluetoothConnection connection,
    Stream stream,
    Map<String, dynamic> payload, {
    Duration timeout = const Duration(minutes: 1),
  }) : _connection = connection {
    payloadSize.value = payload['size'];

    final int packageSize = payload['packageSize'];
    final List hashes = payload['hashes'];
    final id = payload['id'];
    final type = payload['type'];

    final Completer<List<int>> completer = Completer();

    _timeout = _createTimeout(completer, 'receive file', timeout: timeout);

    _subscription = stream.listen((data) {
      _timeout?.cancel();

      _payload.addAll(data);
      processedPayload.value = _payload.length;

      try {
        if (_payload.length == payloadSize.value) {
          final int offset = (hashes.length - 1) * packageSize;
          final checksum = sha1.convert(_payload.sublist(offset));

          bool valid = _validateFilePart(
            completer,
            checksum,
            hashes,
            offset,
            packageSize,
            id,
            type,
          );

          if (valid) {
            _subscription?.cancel();
            completer.complete(_payload);
            return;
          }
        } else if (_payload.length % packageSize == 0) {
          final int offset =
              ((_payload.length / packageSize).floor() - 1) * packageSize;

          final checksum = sha1.convert(
            _payload.sublist(offset, offset + packageSize),
          );

          _validateFilePart(
            completer,
            checksum,
            hashes,
            offset,
            packageSize,
            id,
            type,
          );
        }
      } catch (e, stack) {
        completer.completeError(e, stack);
      }

      _timeout = _createTimeout(
        completer,
        'receive file',
        timeout: const Duration(seconds: 5),
      );
    });

    result = completer.future;
  }

  void cancel() {
    _initializer?.cancel();
    _subscription?.cancel();
    _timeout?.cancel();
  }

  void _readHeader(message, completer, check, [Function? redo]) {
    try {
      Map<String, dynamic> response = json.decode(message);

      if (check(response)) {
        final status = response['status'];

        if ((status ?? 'ok') == 'ok') {
          _header = response;

          payloadSize.value = (_header?['payload']['size'] ?? 0);
          processedPayload.value = 0;

          if (payloadSize.value == 0) {
            cancel();

            completer.complete(_payload);
          }
        } else if (status == 'invalid' && redo != null) {
          redo();
        } else {
          cancel();

          completer.completeError(Exception(response['status']));
        }
      }
    } catch (e) {}
  }

  void _readPayload(data, Completer completer, onMismatch, createTimeout,
      [check]) {
    _payload.addAll(data.toList());
    _timeout?.cancel();

    processedPayload.value = _payload.length;

    // region checking if all required fields are present
    bool valid = _header?['payload']?['size'] != null;
    valid &= _header?['payload']?['checksum'] != null;

    if (!valid) {
      completer.completeError(
        Exception(
          'Information about payload send by the WaterScope device missing',
        ),
        StackTrace.current,
      );

      return;
    }
    // endregion

    if (_payload.length >= _header!['payload']['size']) {
      if (_payload.length > _header!['payload']['size']) {
        _payload.removeRange(_header!['payload']['size'], _payload.length);
      }

      if (sha1.convert(_payload).toString() ==
          _header!['payload']['checksum']) {
        _subscription?.cancel();

        if (check == null || check(_payload)) {
          completer.complete(_payload);
        } else {
          _header = null;
          processedPayload.value = null;
          _payload.clear();

          _timeout = createTimeout(true);
        }
      } else {
        _header = null;
        _payload.clear();

        onMismatch();
      }
    } else {
      _timeout = createTimeout();
    }
  }

  Future<void> _forEachFileChunk(file, callback) async {
    const packageSize = const int.fromEnvironment(
          'PACKAGE_SIZE',
          defaultValue: 50,
        ) *
        1024; // in kilobytes

    for (var offset = 0; offset < file.length; offset += packageSize) {
      int end = offset + packageSize;
      end = end > file.length ? file.length : end;

      final result = callback(
        file.sublist(offset, end),
        (offset / packageSize).floor(),
      );

      if (result is Future) {
        await result;
      }
    }
  }

  bool _validateFilePart(
    completer,
    checksum,
    hashes,
    offset,
    packageSize,
    id,
    type,
  ) {
    final int part = (offset / packageSize).floor();

    String status = 'ok';

    if (checksum.toString() != hashes[part]) {
      status = 'invalid';
      _payload.removeRange(offset, _payload.length);
      processedPayload.value = _payload.length;
    }

    Uint8List payload = _encode({
      'part': part,
      'id': id,
      'type': type,
    });

    _send(
      completer,
      _encode({'status': status, 'payload': _calculatePayload(payload)}),
      payload,
    );

    return status == 'ok';
  }

  static int _id = 0;

  Timer? _createTimeout(
    Completer completer,
    String instruction, {
    Duration? timeout,
  }) {
    return timeout == null
        ? null
        : Timer(
            timeout,
            () {
              _subscription?.cancel();

              if (!completer.isCompleted) {
                completer.completeError(
                  TimeoutException(
                    'WaterScope instruction "$instruction" took too long',
                    timeout,
                  ),
                  StackTrace.current,
                );
              }
            },
          );
  }

  Map _calculatePayload(Uint8List? payload) {
    final _size = payload?.length ?? 0;
    final _checksum = _size > 0 ? sha1.convert(payload!).toString() : '';

    return {
      'size': _size,
      'checksum': _checksum,
    };
  }

  void _send(
    Completer completer,
    Uint8List instruction, [
    Uint8List? payload,
  ]) {
    try {
      _connection.output.add(instruction);
    } catch (e, stack) {
      cancel();

      if (!completer.isCompleted) {
        completer.completeError(e, stack);
      }
    }

    if (payload?.isNotEmpty ?? false) {
      _connection.output.add(payload!);
    }
  }

  static _encode(Map map) => utf8.encode(json.encode(map));
}

class Bluetooth extends InheritedWidget {
  Bluetooth({
    Key? key,
    required this.device,
    @required this.connection,
    @required this.stream,
    required this.connected,
    required Widget child,
  }) : super(child: child, key: key);

  final BluetoothDevice device;
  final BluetoothConnection? connection;
  final Stream<Uint8List>? stream;
  final ValueNotifier<bool> connected;

  static Bluetooth? of(BuildContext context) {
    return context.dependOnInheritedWidgetOfExactType<Bluetooth>();
  }

  static String? extractFromResult(String resultString, String identifier) {
    final RegExpMatch? match = RegExp(
      r'(,|^| )' + identifier + r'=[^,]+(,|$)',
    ).firstMatch(resultString);

    if (match == null) {
      return null;
    }

    return resultString.substring(
      (match.start + (match.start == 0 ? 0 : 1)) + identifier.length + 1,
      match.end == resultString.length ? match.end : match.end - 1,
    );
  }

  static void streamListener(
    Uint8List data,
    void Function(String message, Uint8List raw) onString, {
    void Function(Uint8List data)? onRaw,
  }) {
    try {
      onString(utf8.decode(data).trim(), data);
    } catch (error, stacktrace) {
      if (error is FormatException) {
        if (onRaw != null) {
          onRaw(data);
        }
      } else {
        print(error);
        FirebaseCrashlytics.instance.recordError(error, stacktrace);
      }
    }
  }

  @override
  bool updateShouldNotify(covariant Bluetooth old) {
    return connection != old.connection || connected != old.connected;
  }
}
