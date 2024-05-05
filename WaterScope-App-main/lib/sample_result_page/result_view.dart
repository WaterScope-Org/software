import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:location/location.dart';
import 'package:preload_page_view/preload_page_view.dart';
import 'package:waterscope/bluetooth.dart';

import 'error_info.dart';
import 'result_config.dart';
import 'result_image.dart';

class ResultView extends StatefulWidget {
  const ResultView({
    Key? key,
    required this.configs,
    required this.sampleId,
    required this.incubationTime,
    required this.sampleVolume,
    required this.sampleLocation,
    required this.overrideSample,
    this.comment,
    this.instruction,
    this.position,
  }) : super(key: key);

  final Future<BluetoothInstruction>? instruction;
  final Future<Map<String, dynamic>> configs;
  final int sampleId;
  final int sampleVolume;
  final String? comment;
  final String sampleLocation;
  final Duration incubationTime;
  final bool overrideSample;
  final LocationData? position;

  @override
  _ResultViewState createState() => _ResultViewState();
}

enum _SampleState {
  waiting,
  submitted,
  analysing,
  defogging,
  autofocusing,
  imageCapturing,
  counting,
  result,
  timeout,
  error,
}

class _SampleStage {
  const _SampleStage(this.state, this.timeout, {this.terminating = false});

  final _SampleState state;
  final Duration timeout;
  final bool terminating;
}

class _ResultViewState extends State<ResultView> {
  _SampleState _sampleState = _SampleState.waiting;
  Map<String, dynamic>? _result;

  BluetoothInstruction? _bluetooth;
  Future<BluetoothInstruction>? _preview, _raw;

  bool? saveAvailable;

  final DateTime _submissionTime = DateTime.now();

  static const Map<String, _SampleStage> _stages = {
    'analysing': const _SampleStage(
      _SampleState.analysing,
      const Duration(minutes: 3),
    ),
    'defogging': const _SampleStage(
      _SampleState.defogging,
      const Duration(minutes: 4),
    ),
    'autofocusing': const _SampleStage(
      _SampleState.autofocusing,
      const Duration(minutes: 3),
    ),
    'image capturing': const _SampleStage(
      _SampleState.imageCapturing,
      const Duration(minutes: 3),
    ),
    'counting': const _SampleStage(
      _SampleState.counting,
      const Duration(minutes: 3),
    ),
    'result': const _SampleStage(
      _SampleState.result,
      const Duration(minutes: 3),
      terminating: true,
    )
  };

  @override
  void initState() {
    super.initState();

    SchedulerBinding.instance!.addPostFrameCallback((_) {
      Bluetooth bluetooth = Bluetooth.of(context)!;

      if (bluetooth.connected.value) {
        widget.instruction?.then((fetcher) {
          fetcher.result.then((data) {
            if (data?.isNotEmpty ?? false) {
              _result = json.decode(utf8.decode(data!));
              setState(() => _sampleState = _SampleState.result);
            } else {
              setState(() => _sampleState = _SampleState.submitted);

              listenOnUpdate(bluetooth, const Duration(minutes: 3));
            }
          }).catchError(
            (exception, stack) {
              FirebaseCrashlytics.instance.recordError(exception, stack);

              setState(
                () => _sampleState = exception is TimeoutException
                    ? _SampleState.timeout
                    : _SampleState.error,
              );
            },
          );
        });
      }

      savedResult.then((doc) => setState(() => saveAvailable = doc != null));
    });
  }

  @override
  void dispose() {
    widget.instruction?.then((fetcher) => fetcher.cancel());
    _bluetooth?.cancel();
    super.dispose();
  }

  void listenOnUpdate(Bluetooth bluetooth, Duration timeout) {
    _bluetooth = BluetoothInstruction.receive(
        bluetooth.connection!, bluetooth.stream!, 'sample', check: (payload) {
      try {
        payload = json.decode(utf8.decode(payload));

        return payload['id'] == widget.sampleId;
      } catch (e) {
        return false;
      }
    }, timeout: timeout);

    _bluetooth!.result.catchError((exception, stack) {
      void Function() update = () {
        if (exception is TimeoutException) {
          _sampleState = _SampleState.timeout;
        } else {
          _sampleState = _SampleState.error;
        }
      };

      FirebaseCrashlytics.instance.recordError(exception, stack);

      if (mounted) {
        setState(update);
      } else {
        update();
      }
    });

    _bluetooth!.result.then((payload) {
      // only valid json will get to this point
      Map<String, dynamic> data = json.decode(utf8.decode(payload!));

      _SampleStage? stage = _stages[data['status']];

      if (stage == null) {
        return;
      }

      void Function() update = () => _sampleState = stage.state;

      if (stage.terminating) {
        _result = data['result'];
        saveResult();
      } else {
        listenOnUpdate(bluetooth, stage.timeout);
      }

      if (mounted) {
        setState(update);
      } else {
        update();
      }
    });
  }

  Future<DocumentSnapshot?> get savedResult async {
    if (widget.overrideSample) {
      return null;
    }

    try {
      final doc = await FirebaseFirestore.instance
          .collection('devices')
          .doc(Bluetooth.of(context)!.device.address)
          .collection('samples')
          .doc(widget.sampleId.toString())
          .get();

      return doc.exists ? doc : null;
    } catch (e) {
      return null;
    }
  }

  Future<String?> fromResultSource(
    String? Function() fromJson,
    String? Function(DocumentSnapshot) fromSave,
  ) async {
    if (_result != null) {
      return fromJson();
    }

    DocumentSnapshot? doc = await savedResult;

    if (doc?.exists ?? false) {
      return fromSave(doc!);
    }

    return null;
  }

  dynamic retrieveOptionalField(DocumentSnapshot doc, String field) {
    try {
      return doc[field].toString();
    } catch (e) {
      if (!(e is StateError)) {
        throw e;
      }
    }

    return null;
  }

  Future<String?> get resultFlag => fromResultSource(
        () => _result?['flag'] ?? null,
        (doc) => retrieveOptionalField(doc, 'resultFlag'),
      );

  Future<int?> get otherColiCount async {
    String? count = await fromResultSource(
      () => _result?['otherColiform'].toString(),
      (doc) => retrieveOptionalField(doc, 'otherColiform'),
    );

    return count != null ? int.parse(count) : null;
  }

  Future<double?> get chlorineMeasurement async {
    String? measurement = await fromResultSource(
          () => _result?['chlorine_level'].toString(),
          (doc) => retrieveOptionalField(doc, 'chlorine_level'),
    );
    return measurement != null ? double.tryParse(measurement) : null;
  }

  Future<int?> get eColiCount async {
    String? count = await fromResultSource(
      () => _result?['eColiform'].toString(),
      (doc) => retrieveOptionalField(doc, 'eColiform'),
    );

    return count != null ? int.parse(count) : null;
  }

  Future<Map<String, dynamic>> get resultEntryData async {
    Map<String, dynamic> entryData = {};

    int? eColi = await eColiCount;
    if (eColi != null) {
      entryData.putIfAbsent('eColiform', () => eColi);
    }

    int? otherColi = await otherColiCount;
    if (otherColi != null) {
      entryData.putIfAbsent('otherColiform', () => otherColi);
    }



    String? flag = await resultFlag;
    if (flag != null) {
      entryData.putIfAbsent('resultFlag', () => flag);
    }

    if (widget.comment != null) {
      entryData.putIfAbsent('comment', () => widget.comment);
    }

    return entryData;
  }

  void saveResult() async {
    DocumentReference entryRef = FirebaseFirestore.instance
        .collection('devices')
        .doc(Bluetooth.of(context)!.device.address)
        .collection('samples')
        .doc(widget.sampleId.toString());

    Map<String, dynamic> entryData = await resultEntryData;

    entryData.addAll({
      'submissionTime': _submissionTime,
      'location': widget.sampleLocation,
      'incubationTime': widget.incubationTime.inMinutes,
      'sampleVolume': widget.sampleVolume,
      'coordinates': widget.position != null
          ? GeoPoint(
              widget.position!.latitude!,
              widget.position!.longitude!,
            )
          : null,
    });

    entryRef.set(entryData);
  }

  Future<ResultConfig> get resultConfig async {
    Map<String, dynamic> configs = await this.widget.configs;
    String? _resultFlag = await resultFlag;

    if (_resultFlag != null && configs.containsKey(_resultFlag)) {
      return ResultConfig.fromJson(
        Map<String, dynamic>.from(configs[_resultFlag]),
      );
    }

    List levels = configs['levels'];
    levels = levels
        .map((json) => ResultConfig.fromJson(json as Map<String, dynamic>))
        .toList();

    for (ResultConfig level in levels) {
      final int maxColi = level.maxColi ?? double.maxFinite.toInt();
      final int maxEColi = level.maxEColi ?? double.maxFinite.toInt();

      if ((await otherColiCount)! <= maxColi &&
          (await eColiCount)! <= maxEColi) {
        return level;
      }
    }

    return levels.last;
  }

  Widget createLoadingIndicator(String infoText) => Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Padding(
              padding: const EdgeInsets.only(bottom: 24, top: 24),
              child: const CircularProgressIndicator(),
            ),
            Text(
              infoText,
              textAlign: TextAlign.center,
              style: TextStyle(fontStyle: FontStyle.italic),
            )
          ],
        ),
      );

  Future<BluetoothInstruction> fetchImage(bluetooth, String name) async {
    final payload = Uint8List.fromList(utf8.encode(
      json.encode({
        'action': 'get $name image',
        'id': widget.sampleId,
      }),
    ));

    BluetoothInstruction request = BluetoothInstruction.send(
      bluetooth.connection,
      bluetooth.stream,
      'sample',
      payload: payload,
    );

    Map<String, dynamic> imageInfo;

    try {
      if (await request.result == null) {
        throw Exception('WaterScope device did not send image info');
      }
    } catch (e) {
      if (e is TimeoutException) {
        request = BluetoothInstruction.send(
          bluetooth.connection,
          bluetooth.stream,
          'sample',
          payload: payload,
        );
      } else {
        throw e;
      }
    }

    imageInfo = json.decode(utf8.decode((await request.result)!));

    return BluetoothInstruction.receiveFile(
      bluetooth.connection,
      bluetooth.stream,
      imageInfo,
    );
  }

  Widget get images {
    Bluetooth bluetooth = Bluetooth.of(context)!;

    if (bluetooth.connected.value && _preview == null) {
      _preview = fetchImage(bluetooth, 'preview');

      _raw = Future<BluetoothInstruction>(() async {
        await (await _preview)!.result;

        return await fetchImage(bluetooth, 'raw');
      });
    }

    return Container(
      height: 265,
      color: Colors.black,
      child: PreloadPageView(
        preloadPagesCount: 2,
        children: [
          ResultImage(
            sampleId: widget.sampleId,
            tag: 'preview',
            key: Key('preview_image'),
            fetcher: _preview,
            overrideSample: widget.overrideSample,
          ),
          ResultImage(
            sampleId: widget.sampleId,
            tag: 'raw',
            key: Key('raw_image'),
            fetcher: _raw,
            overrideSample: widget.overrideSample,
          ),
        ],
      ),
    );
  }

  Widget createBody(ResultConfig config, int? eColi, int? otherColiform, double? chlorineLevel) {
    late String resultText;
    if(chlorineLevel == null){
    if (config.resultOverride != null)  {
      resultText = config.resultOverride!;
    } else {
      resultText =
           '                  E. coli: ${eColi ?? 'unknown'} CFUs/100ml \n Other coliforms: ${otherColiform ?? 'unknown'} CFUs/100ml';

    }}
    else {
      resultText = 'Free chlorine: \n ${chlorineLevel.toStringAsFixed(2)} mg/L';
    }

    List<Widget> bodyWidgets = [
        images,
        Container(
          padding: const EdgeInsets.all(16),
          color: config.color,
          alignment: Alignment.center,
          child: Text(
            resultText,
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.only(
            left: 16.0,
            right: 16,
            top: 16,
            bottom: 0,
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.info,
                color: config.color,
                size: 24,
              ),
              Flexible(
                fit: FlexFit.loose,
                child: Padding(
                  padding: const EdgeInsets.only(
                    left: 8.0,
                    right: 32.0,
                  ),
                  child: Text(
                    config.resultText,
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.bodyText1,
                  ),
                ),
              )
            ],
          ),
        ),
      ];

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: bodyWidgets,
    );
  }



  Widget get body => FutureBuilder(
        future: eColiCount,
    builder: (_, AsyncSnapshot<int?> eColi) {
          if (eColi.hasError) {
            if (!(eColi.error is ArgumentError)) {
              print(eColi.error);
              FirebaseCrashlytics.instance.recordError(
                eColi.error,
                StackTrace.current,
              );
            }

            return ErrorInfo(errorMessage: 'Failed to parse E. coliform count');
          } else if (!eColi.hasData) {
            return createLoadingIndicator('Parsing E. Coli count');
          }

          return FutureBuilder(
            future: otherColiCount,
            builder: (_, AsyncSnapshot<int?> coliform) {
              if (eColi.hasError) {
                return ErrorInfo(
                  errorMessage: 'Failed to parse other coliform count',
                );
              } else if (!eColi.hasData) {
                return createLoadingIndicator('Parsing other coliform count');
              }

          return FutureBuilder(
            future: chlorineMeasurement,
            builder: (_, AsyncSnapshot<double?> chlorine) {
              if (eColi.hasError) {
              return ErrorInfo(
              errorMessage: 'Failed to parse other Chlorine measurement',
              );
              } else if (!eColi.hasData) {
              return createLoadingIndicator('Parsing Chlorine measurement');
              }

              return FutureBuilder(
                future: resultConfig,
                builder: (_, AsyncSnapshot<ResultConfig> config) {
                  if (config.hasError) {
                    print(
                      'Failed to load the corresponding result '
                      'config for result "$_result": ${config.error}',
                    );
                    return ErrorInfo(errorMessage: config.error.toString());
                  } else if (!config.hasData) {
                    return createLoadingIndicator('Comparing result');
                  }

                  return createBody(config.data!, eColi.data, coliform.data, chlorine.data);
                },
              );
            },
          );
            },
          );
    },
  );

  Widget createErrorInfo(String errorMessage) {
    return Padding(
      padding: const EdgeInsets.only(top: 16.0),
      child: ErrorInfo(errorMessage: errorMessage),
    );
  }

  bool get resultsAvailable =>
      _sampleState == _SampleState.result || saveAvailable == true;

  @override
  Widget build(BuildContext context) => ValueListenableBuilder(
    valueListenable: Bluetooth.of(context)!.connected,
        builder: (_, bool connected, __) {
          if (saveAvailable == true) {
            return body;
          }

          if (!resultsAvailable && !connected) {
            return createErrorInfo(
              'Connection to WaterScope device lost and results not saved',
            );
          }

          if (saveAvailable == null && !connected) {
            return createLoadingIndicator('Loading saved results');
          }

          switch (_sampleState) {
            case _SampleState.submitted:
              return createLoadingIndicator('Sample submitted successfully');
            case _SampleState.analysing:
              return createLoadingIndicator('Analysing sample');
            case _SampleState.defogging:
              return createLoadingIndicator('Defogging');
            case _SampleState.autofocusing:
              return createLoadingIndicator('Autofocusing');
            case _SampleState.imageCapturing:
              return createLoadingIndicator('Capturing image');
            case _SampleState.counting:
              return createLoadingIndicator('Counting bacteria colonies');
            case _SampleState.result:
              return body;
            case _SampleState.timeout:
              return createErrorInfo(
                'WaterScope device took too long to respond.',
              );
            case _SampleState.error:
              return createErrorInfo(
                'An unexpected error occurred.',
              );
            default:
              return createLoadingIndicator(
                'Waiting for WaterScope device to respond',
              );
          }
        },
      );
}
