import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:share/share.dart';
import 'package:waterscope/bluetooth.dart';
import 'package:waterscope/main.dart';
import 'package:workmanager/workmanager.dart';

import 'error_info.dart';

void _uploadCallbackDispatcher() {
  Workmanager().executeTask((taskName, inputData) async {
    try {
      WidgetsFlutterBinding.ensureInitialized();
      await Firebase.initializeApp();
      await initCrashlytics();

      final file = FirebaseStorage.instance.ref(inputData!['path']);

      bool moreRecent = true;

      while (moreRecent) {
        try {
          FullMetadata metadata = await file.getMetadata();

          moreRecent = DateTime.parse(inputData['sampleTime']).isAfter(
            metadata.updated ?? DateTime.fromMillisecondsSinceEpoch(0),
          );

          if (!moreRecent) {
            break;
          }
        } catch (error, stacktrace) {
          if (!(error is FirebaseException) ||
              error.code != 'object-not-found') {
            print(error);
            FirebaseCrashlytics.instance.recordError(error, stacktrace);
            return false;
          }
        }

        try {
          await file.putFile(File(inputData['localPath']));
        } catch (e) {
          if (e is FirebaseException &&
              e.code == "storage/retry-limit-exceeded") {
            Future.delayed(const Duration(minutes: 5));
            continue;
          } else {
            throw e;
          }
        }
        break;
      }
    } catch (error, stacktrace) {
      print(error);
      FirebaseCrashlytics.instance.recordError(error, stacktrace);
      return false;
    }

    return true;
  });
}

class ResultImage extends StatefulWidget {
  ResultImage({
    Key? key,
    required this.tag,
    required this.sampleId,
    @required this.fetcher,
    this.overrideSample = false,
  }) : super(key: key);

  final String tag;
  final int sampleId;
  final Future<BluetoothInstruction>? fetcher;
  final bool overrideSample;

  @override
  _ResultImageState createState() => _ResultImageState();
}

enum _PictureState {
  waiting,
  receiving,
  done,
  error,
  timeout,
  retry,
}

class _ResultImageState extends State<ResultImage> {
  BluetoothInstruction? _bluetooth;
  _PictureState _state = _PictureState.waiting;

  List<int> _imageData = [];

  bool? _saveAvailable;

  @override
  void initState() {
    super.initState();

    SchedulerBinding.instance?.addPostFrameCallback((_) {
      widget.fetcher?.then(
        setupBluetoothListener,
        onError: (e, stack) {
          FirebaseCrashlytics.instance.recordError(e, stack);

          _state = e is TimeoutException
              ? _PictureState.timeout
              : _PictureState.error;

          if (mounted) {
            setState(() => _state = _state);
          }
        },
      );

      syncSavedImage();
    });

    Workmanager().initialize(
      _uploadCallbackDispatcher,
      isInDebugMode: kDebugMode,
    );
  }

  @override
  void dispose() {
    widget.fetcher?.then((instruction) => instruction.cancel());
    _bluetooth?.cancel();
    super.dispose();
  }

  void setupBluetoothListener(BluetoothInstruction bluetooth) {
    _bluetooth = bluetooth;
    setState(() => _state = _PictureState.receiving);

    _bluetooth!.result.then((image) {
      if (image == null) {
        if (mounted) {
          setState(() => _state = _PictureState.error);
        }

        return;
      }

      setState(() {
        _imageData = image;
        _state = _PictureState.done;
      });

      saveImage(image);
    });

    _bluetooth!.result.catchError((exception, stack) {
      setState(() => _state = _PictureState.timeout);
      FirebaseCrashlytics.instance.recordError(exception, stack);
    }, test: (e) => e is TimeoutException);

    _bluetooth!.result.catchError((exception, stack) {
      setState(() => _state = _PictureState.error);
      FirebaseCrashlytics.instance.recordError(exception, stack);
    }, test: (e) => !(e is TimeoutException));
  }

  void syncSavedImage() async {
    final _saveFile = await saveFile;
    bool localSaved = await _saveFile.exists();

    if (localSaved) {
      if (mounted) {
        setState(() => _saveAvailable = !widget.overrideSample);
      }

      if (widget.overrideSample) {
        _saveFile.deleteSync();
      }
    }

    try {
      final remote = FirebaseStorage.instance.ref(remotePath);

      if (!widget.overrideSample) {
        DateTime remoteUpdate = (await remote.getMetadata()).updated ??
            DateTime.fromMillisecondsSinceEpoch(
              0,
            );

        if (!localSaved ||
            remoteUpdate.isAfter(await _saveFile.lastModified())) {
          Uint8List? data = await remote.getData();

          if (data != null) {
            await _saveFile.writeAsBytes(data);
          }

          if (mounted) {
            setState(() => _saveAvailable = data != null);
          }
        }
      } else {
        await remote.delete();
      }
    } catch (error, stacktrace) {
      if (_saveAvailable == null) {
        if (mounted) {
          setState(() => _saveAvailable = false);
        }
      }

      if (!(error is FirebaseException) || error.code != 'object-not-found') {
        FirebaseCrashlytics.instance.recordError(error, stacktrace);
      }
    }
  }

  Future<File> get saveFile async {
    final imageDirectory = Directory(path.join(
      (await getTemporaryDirectory()).path,
      Bluetooth.of(context)!.device.address,
      widget.sampleId.toString(),
    ));

    if (!(await imageDirectory.exists())) {
      await imageDirectory.create(recursive: true);
    }

    final localPath = path.join(imageDirectory.path, widget.tag + '.jpg');
    return File(localPath);
  }

  void updateState(_PictureState state) {
    if (mounted) {
      setState(() => _state = state);
    }
  }

  String get remotePath => '${Bluetooth.of(context)!.device.address}/'
      '${widget.sampleId}/${widget.tag + '.jpg'}';

  void saveImage(List<int> data) async {
    try {
      String device = Bluetooth.of(context)!.device.address;

      DateTime sampleTime = DateTime.now();

      final _saveFile = await saveFile;

      if (_saveFile.existsSync()) {
        if (sha1.convert(data) == sha1.convert(_saveFile.readAsBytesSync())) {
          return;
        }
      }

      await _saveFile.writeAsBytes(data);
      setState(() => _saveAvailable = true);
      Workmanager().registerOneOffTask(
          '$device - ${widget.sampleId} - ${widget.tag}', 'imageUpload',
          existingWorkPolicy: ExistingWorkPolicy.replace,
          constraints: Constraints(networkType: NetworkType.connected),
          inputData: <String, dynamic>{
            'path': remotePath,
            'localPath': _saveFile.path,
            'sampleTime': sampleTime.toIso8601String(),
          });
    } catch (error, stacktrace) {
      print(error);
      FirebaseCrashlytics.instance.recordError(error, stacktrace);
    }
  }

  Widget get loadingIndicator {
    String infoText;

    switch (_state) {
      case _PictureState.retry:
        infoText = 'Received ${widget.tag} image is corrupted. '
            'Retrying to fetch image.';
        break;

      case _PictureState.receiving:
        infoText = 'Receiving ${widget.tag} image. '
            'This might causes the app to become unresponsive.';
        break;

      default:
        infoText = 'Waiting on ${widget.tag} image.';
        break;
    }

    if (_saveAvailable == null) {
      infoText = 'Loading saved ${widget.tag} image';
    }

    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(bottom: 24, top: 24),
              child: Center(
                child: _state != _PictureState.receiving
                    ? const CircularProgressIndicator()
                    : ValueListenableBuilder(
                        valueListenable: _bluetooth!.payloadSize,
                        builder: (_, int? size, __) {
                          if ((size ?? 0) > 0) {
                            return ValueListenableBuilder(
                              valueListenable: _bluetooth!.processedPayload,
                              builder: (_, int? received, __) {
                                return CircularProgressIndicator(
                                  value: (received ?? 0) / size!,
                                );
                              },
                            );
                          }

                          return CircularProgressIndicator();
                        },
                      ),
              ),
            ),
          ),
          Text(
            infoText,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontStyle: FontStyle.italic,
              color: Colors.white,
            ),
          )
        ],
      ),
    );
  }

  Widget get placeholder {
    return ValueListenableBuilder(
      valueListenable: Bluetooth.of(context)!.connected,
      builder: (_, bool connected, __) {
        if (_saveAvailable == false && _state != _PictureState.done) {
          if (!connected) {
            return Center(
              child: ErrorInfo(
                color: Colors.white,
                errorMessage: 'Connection to WaterScope device lost '
                    'and no ${widget.tag} image saved',
              ),
            );
          }
        }

        if (_state == _PictureState.error) {
          return Center(
            child: ErrorInfo(
              color: Colors.white,
              errorMessage: 'Receiving ${widget.tag} image failed.',
            ),
          );
        } else if (_state == _PictureState.timeout) {
          return Center(
            child: ErrorInfo(
              color: Colors.white,
              errorMessage: 'Receiving ${widget.tag} image timed out.',
            ),
          );
        } else {
          return loadingIndicator;
        }
      },
    );
  }

  String get shareText =>
      'Have a look at this ${widget.tag} image of sample ${widget.sampleId}';

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 265,
      color: Colors.black,
      child: FutureBuilder(
        future: saveFile,
        builder: (_, AsyncSnapshot<File> saveFile) {
          ImageProvider? image;

          if (_state == _PictureState.done) {
            image = MemoryImage(Uint8List.fromList(_imageData));
          } else if (!widget.overrideSample) {
            if (saveFile.data?.existsSync() ?? false) {
              image = FileImage(saveFile.data!);
            }
          }

          return image != null
              ? LayoutBuilder(
                  builder: (_, BoxConstraints constraints) => Stack(
                    alignment: Alignment.center,
                    children: [
                      Image(
                        image: ResizeImage(
                          image!,
                          height: constraints.maxHeight.toInt(),
                          allowUpscaling: true,
                        ),
                      ),
                      if (_saveAvailable == true)
                        Positioned(
                          top: 0,
                          right: 0,
                          child: IconButton(
                            onPressed: () => Share.shareFiles(
                              [saveFile.data!.path],
                              mimeTypes: ['image/jpeg'],
                              text: shareText,
                            ),
                            icon: Icon(Icons.share, color: Colors.white),
                          ),
                        )
                    ],
                  ),
                )
              : placeholder;
        },
      ),
    );
  }
}
