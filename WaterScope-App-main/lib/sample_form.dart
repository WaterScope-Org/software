import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:geocoding/geocoding.dart' as geocoding;
import 'package:location/location.dart';
import 'package:timezone/timezone.dart' as tz;

import 'bluetooth.dart';
import 'dialog.dart' as DefaultDialog;
import 'sample_result_page/sample_result_page.dart';

class SampleForm extends StatefulWidget {
  SampleForm({Key? key}) : super(key: key);

  @override
  _SampleFormState createState() => _SampleFormState();
}

class _SampleFormState extends State<SampleForm> {
  static final notifications = FlutterLocalNotificationsPlugin();

  final Location _locator = Location();
  Future<LocationData>? _position;
  Future<List<geocoding.Placemark>>? _places;

  bool _reminder = false;
  int? _sampleId;
  String? _location, _comment;
  Key _locationInputKey = UniqueKey();
  Duration _incubationTime = const Duration(hours: 0);
  static const int _defaultIncubationTime = 21; // in hours
  Duration _reminderTime = const Duration(hours: 9);
  String _remindBefore = 'in 1 hour';
  double? _ph;
  double? _tds;
  double? _turbidity;
  double? _salinity;
  double? _conductivity;
  double? _orp;
  double? _specificGravity;
  double? _waterTemperature;

  String _analysisType = 'E.coli/Coliform'; // default value
  String _sampleVol = '100 ml';
  int _sampleVol_integer = 100;
  BluetoothInstruction? _bluetooth;
  Future? _idCheck;
  // Duration dropdownValue = const Duration(hours:7);

  @override
  void initState() {
    super.initState();

    _locator.serviceEnabled().then((enabled) {
      if (enabled) {
        _locator.hasPermission().then((permission) {
          if (permission == PermissionStatus.granted) {
            _position = _locator.getLocation();
          }
        });
      }
    });
  }

  @override
  void dispose() {
    _bluetooth?.cancel();
    super.dispose();
  }

  String getTimeDigitString(int digits) {
    String result = digits.toString();
    return result.length == 1 ? '0$result' : result;
  }

  Future<BluetoothInstruction> submitSample(bluetooth, position) async {
    final payload = Uint8List.fromList(utf8.encode(json.encode({
      'action': 'submit',
      'id': _sampleId,
      'location': _location,
      'analysis_type': _analysisType,
      'time': _defaultIncubationTime * 60, // Default incubation time in minutes
      'sample_volume': _sampleVol_integer,
      'comment': _comment?.trim(),
      'pH': _ph,
      'TDS': _tds,
      'Turbidity': _turbidity,
      'Salinity': _salinity,
      'Conductivity': _conductivity,
      'ORP': _orp,
      'Specific Gravity': _specificGravity,
      'Water Temp': _waterTemperature,
      'coordinates': position != null
          ? {
        'latitude': position.latitude,
        'longitude': position.longitude,
      }

          : null,
    })));

    _bluetooth = BluetoothInstruction.send(
      bluetooth.connection,
      bluetooth.stream,
      'sample',
      payload: payload,
    );

    try {
      await _bluetooth!.result;
    } catch (e) {
      if (e is TimeoutException) {
        _bluetooth = BluetoothInstruction.send(
          bluetooth.connection,
          bluetooth.stream,
          'sample',
          payload: payload,
        );
      } else {
        throw e;
      }
    }

    return _bluetooth!;
  }

  Future<LocationData?> requestPosition() async {
    if (await _locator.hasPermission() != PermissionStatus.granted) {
      if (await _locator.requestPermission() != PermissionStatus.granted) {
        return null;
      }
    }

    return _locator.getLocation();
  }

  Future<void> continueWithSample(BuildContext c) async {
    final bluetooth = Bluetooth.of(c)!;




    try {
      _bluetooth = BluetoothInstruction.send(
        bluetooth.connection!,
        bluetooth.stream!,
        'sample',
        payload: Uint8List.fromList(utf8.encode(json.encode({
          'action': 'get',
          'id': _sampleId,
        }))),
        timeout: const Duration(seconds: 10),
      );

      WidgetsBinding.instance?.addPostFrameCallback(
        (_) => setState(() {
          _idCheck = _bluetooth!.result;
        }),
      );

      await _bluetooth!.result;

      bool override = await showDialog(
        context: c,
        builder: (context) => DefaultDialog.Dialog(
          title: 'Duplicated sample ID',
          action: 'Override',
          content: Padding(
            padding: const EdgeInsets.only(top: 8.0),
            child: Text('The sample ID $_sampleId is already used.'),
          ),
          onSubmission: () => Navigator.of(context).pop(true),
        ),
      );

      if (!override) {
        return;
      }
    } catch (e, stack) {
      if (e is TimeoutException) {
        ScaffoldMessenger.of(c).showSnackBar(SnackBar(
          content: (Text('Sample ID check timed out.')),
        ));

        return;
      } else if (e.toString() != 'Exception: Sample #$_sampleId not found') {
        ScaffoldMessenger.of(c).showSnackBar(SnackBar(
          content: (Text('Sample ID check timed out.')),
        ));

        FirebaseCrashlytics.instance.recordError(e, stack);
        return;
      }
    }

    LocationData? position;

    try {
      if (!(await _locator.serviceEnabled())) {
        if (await _locator.requestService()) {
          position = await requestPosition();
        }
      } else {
        position = await requestPosition();
      }
    } catch (e, stack) {
      FirebaseCrashlytics.instance.recordError(e, stack);
    }

    final title = 'Sample $_sampleId';
    final body = 'Make sure to check on your sample again';

    final notificationDetails = NotificationDetails(
      android: AndroidNotificationDetails(
        'reminder',
        'Sample reminders',
        channelDescription: 'Displaying sample reminders scheduled by you.',
        importance: Importance.defaultImportance,
        icon: 'drawable/ic_stat_reminder',
      ),
    );

    switch (_remindBefore) {
      case 'Now':
        _reminderTime = Duration(hours: 0);
        break;
      case 'in 1 hour':
        _reminderTime = Duration(hours: 1);
        break;
      case 'in 2 hours':
        _reminderTime = Duration(hours: 2);
        break;
      case 'in 3 hours':
        _reminderTime = Duration(hours: 3);
        break;
      case 'in 4 hours':
        _reminderTime = Duration(hours: 4);
        break;
      case 'in 5 hours':
        _reminderTime = Duration(hours: 1);
        break;
    }
    _reminderTime = _remindBefore.substring(0, 2)=='in' ? Duration(hours: int.parse(_remindBefore.replaceAll(new RegExp(r'[^0-9]'),''))) : Duration(seconds: 10);

    _sampleVol_integer = int.parse(_sampleVol.replaceAll(new RegExp(r'[^0-9]'),''));

    notifications.zonedSchedule(
      _sampleId!,
      title,
      body,
      tz.TZDateTime.now(tz.local)
          .add(const Duration(seconds: 5) + _reminderTime),
      notificationDetails,
      androidAllowWhileIdle: false,
      uiLocalNotificationDateInterpretation:
          UILocalNotificationDateInterpretation.absoluteTime,
    );

    if (mounted) {
      final Bluetooth bluetooth = Bluetooth.of(c)!;
      final submission = submitSample(bluetooth, position);

      Navigator.of(c).push(
        MaterialPageRoute(
          settings: RouteSettings(name: 'sample-results'),
          builder: (_) {
            return Bluetooth(
              device: bluetooth.device,
              connection: bluetooth.connection,
              connected: bluetooth.connected,
              stream: bluetooth.stream,
              child: SampleResultPage(
                overrideSample: true,
                instruction: submission,
                sampleId: _sampleId!,
                sampleVolume: _sampleVol_integer,
                incubationTime: _incubationTime,
                comment: _comment,
                sampleLocation: _location!,
                position: position,
              ),
            );
          },
        ),
      );
    }
  }

  void parseSampleId(String? source) {
    int? input;

    if (idValidator(source) == null) {
      input = int.parse(source!);
    }

    setState(() => _sampleId = input);
  }

  String? idValidator(String? source) {
    if (source?.isEmpty ?? true) {
      return 'An ID is required';
    }

    return null;
  }

  String? locationValidator(String? source) {
    if (source?.trim().isEmpty ?? true) {
      return 'A location is required.';
    }

    return null;
  }

  Widget createLocationTextInput({bool hasError = false, String? initValue}) {
    if (initValue != null && _location == null) {
      _location = initValue;
      _locationInputKey = Key(initValue);
    }

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        TextFormField(
          textInputAction: TextInputAction.done,
          key: _locationInputKey,
          initialValue: initValue,
          autovalidateMode: AutovalidateMode.onUserInteraction,
          onChanged: (location) => setState(() => _location = location.trim()),
          validator: locationValidator,
          decoration: InputDecoration(
            hintText: 'City, Postal Code, Country',
            labelText: 'Location',
          ),
        ),
        if (hasError && _location == null)
          Padding(
            padding: const EdgeInsets.only(top: 8.0),
            child: Text(
              'Failed to fetch your location, please type it in manually.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontWeight: FontWeight.w300,
                color: Colors.red,
                fontSize: 13,
              ),
            ),
          )
      ],
    );
  }

  Widget _createMetricInput(String label, String hint, void Function(String) onChanged) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 8.0),
      child: TextFormField(
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
        ),
        keyboardType: TextInputType.number,
        onChanged: onChanged,
      ),
    );
  }

  Widget get sampleIdInput => TextFormField(
        autofocus: true,
        decoration: InputDecoration(hintText: '1234', labelText: 'Sample ID'),
        onChanged: parseSampleId,
        validator: idValidator,
        textInputAction: TextInputAction.next,
        autovalidateMode: AutovalidateMode.onUserInteraction,
        inputFormatters: [
          FilteringTextInputFormatter.allow(RegExp('[1234567890]')),
        ],
        keyboardType: TextInputType.numberWithOptions(
          decimal: false,
          signed: false,
        ),
      );

  Widget get locationInput => FutureBuilder(
        future: _position,
    builder: (_, AsyncSnapshot<LocationData> position) {
          if (position.hasData) {
            if (_places == null) {
              _places = geocoding.placemarkFromCoordinates(
                position.data!.latitude!,
                position.data!.longitude!,
              );
            }

            return FutureBuilder(
              future: _places,
              builder: (_, AsyncSnapshot<List<geocoding.Placemark>> places) {
                if (places.hasData) {
                  geocoding.Placemark place = places.data!.first;

                  return createLocationTextInput(
                    initValue: '${place.locality}, '
                        '${place.postalCode}, '
                        '${place.country}',
                  );
                }

                return createLocationTextInput(hasError: places.hasError);
              },
            );
          }

          return createLocationTextInput(hasError: position.hasData);
        },
      );

  Widget createTimeOption(Duration duration) {
    final hours = duration.inHours;
    String minutes = (duration.inMinutes - hours * 60).toString();
    minutes = minutes.length < 1 ? minutes : '0$minutes';

    return TextButton(
      style: TextButton.styleFrom(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.zero),
        padding: const EdgeInsets.only(left: 16, right: 16),
      ),
      key: ValueKey(duration),
      onPressed: () => setState(() => _incubationTime = duration),
      child: Row(
        children: [
          Radio(
            value: duration,
            groupValue: _incubationTime,
            onChanged: (Duration? value) =>
                setState(() => _incubationTime = value!),
            key: ValueKey(duration),
          ),
          Text(
            (hours==0) & (minutes=='00') ? 'New Sample' : '$hours h $minutes min',
            style: Theme.of(context).textTheme.bodyText2,
          ),
        ],
      ),
    );
  }
  Widget get analysisTypePicker => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Padding(
        padding: const EdgeInsets.only(
          right: 32,
          left: 32,
          bottom: 8,
          top: 16,
        ),
        child: Text(
          'Analysis Type',
          style: Theme.of(context).textTheme.subtitle1,
        ),
      ),
      ListTile(
        title: const Text('E.coli/Coliform'),
        leading: Radio<String>(
          value: 'E.coli/Coliform',
          groupValue: _analysisType,
          onChanged: (String? value) {
            setState(() {
              _analysisType = value!;
            });
          },
        ),
      ),
      ListTile(
        title: const Text('Chlorine/pH/TDS...'),
        leading: Radio<String>(
          value: 'Chlorine',
          groupValue: _analysisType,
          onChanged: (String? value) {
            setState(() {
              _analysisType = value!;
            });
          },
        ),
      ),
    ],
  );

/*
  Widget get timePicker => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(
              right: 32,
              left: 32,
              bottom: 8,
              top: 16,
            ),
            child: Text(
              'Incubation time',
              style: Theme.of(context).textTheme.subtitle1,
            ),
          ),
          createTimeOption(const Duration(hours: 0)),
          createTimeOption(const Duration(hours: 9)),
          createTimeOption(const Duration(hours: 12)),
          createTimeOption(const Duration(hours: 18)),
          createTimeOption(const Duration(hours: 21)),
          createTimeOption(const Duration(hours: 24)),
        ],
      );*/

  Widget get reminderInput {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.only(right: 18.0, left: 32),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  'Reminder',
                  style: Theme.of(context).textTheme.subtitle1,
                ),
              ),
              Switch(
                value: _reminder,
                onChanged: (state) {
                  FocusScope.of(context).unfocus();
                  setState(() => _reminder = state);
                },
              ),
            ],
          ),
        ),
      ],
    );
  }
// this is the dropdown implementation of reminderInput, where for incubation time > 0 hour, have options
// to be reminded between 1 to 5 hours prior to completion
  Widget get reminderInput2 {
    final hours = _incubationTime.inHours;
    String minutes = (_incubationTime.inMinutes - hours * 60).toString();
    minutes = minutes.length < 1 ? minutes : '0$minutes';
    List<String> dropdownlist = [
      for (var i = 1; i <= 24; i++) (i==1 ? 'in $i hour' : 'in $i hours'),
    ];
    dropdownlist.insert(0, 'Now');
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.only(right: 18.0, left: 32),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  'Reminder',
                  style: Theme.of(context).textTheme.subtitle1,
                ),
              ),
              // Switch(
              //   value: _reminder,
              //   onChanged: (state) {
              //     FocusScope.of(context).unfocus();
              //     setState(() => _reminder = state);
              //   },
              // ),
              Switch(
                value: _reminder,
                onChanged: (state) {
                  FocusScope.of(context).unfocus();
                  setState(() => _reminder = state);
                },
              ),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.only(
            right: 32,
            left: 32,
            top: 4,
            bottom: 8,
          ),
        child:
        ((hours==0) & (minutes=='00') ? (_reminder==true ? DropdownButton(
          value: _remindBefore,
          icon: const Icon(Icons.arrow_downward),
          elevation: 4,
          itemHeight: 48.0,
          menuMaxHeight: 200,
          isExpanded: true,
          style: const TextStyle(color: Colors.deepPurple),
          underline: Container(
            height: 1,
            color: Colors.deepPurpleAccent,
          ),
          onChanged: (String? newValue) {
            setState(() {
              _remindBefore = newValue!;
            });
          },
          items: dropdownlist
              .map<DropdownMenuItem<String>>((String value) {
            return DropdownMenuItem<String>(
              value: value,
              child: Text(value.toString()),
            );
          }).toList(),
        ) : Container(
          width:200,
          child: DropdownButton(
            onChanged: null,
            isExpanded: true,
            items: <String>['in 1 hour'].map<DropdownMenuItem<String>>((String value) {
              return DropdownMenuItem<String>(
                value: value,
                child: Text(value.toString()),
              );
            }).toList(),
          ),
        )) : (_reminder==true ? DropdownButton(
          value: _remindBefore,
          icon: const Icon(Icons.arrow_downward),
          elevation: 4,
          itemHeight: 48.0,
          menuMaxHeight: 200,
          isExpanded: true,
          style: const TextStyle(color: Colors.deepPurple),
          underline: Container(
            height: 1,
            color: Colors.deepPurpleAccent,
          ),
          onChanged: (String? newValue) {
            setState(() {
              _remindBefore = newValue!;
            });
          },
          items: dropdownlist
              .map<DropdownMenuItem<String>>((String value) {
            return DropdownMenuItem<String>(
              value: value,
              child: Text(value.toString()),
            );
          }).toList(),
        ) : Container(
          width:200,
          child: DropdownButton(
            onChanged: null,
            isExpanded: true,
            items: <String>['in 1 hour'].map<DropdownMenuItem<String>>((String value) {
              return DropdownMenuItem<String>(
                value: value,
                child: Text(value.toString()),
              );
            }).toList(),
          ),
        ))),
    ),
        Padding(
          padding: const EdgeInsets.only(
            right: 32,
            left: 32,
            top: 4,
            bottom: 8,
          ),
          child: Text(
            'Remind me in again when the incubation time has completed.',
            style: TextStyle(
              color: _reminder ? Colors.black : Colors.grey,
              height: 1.5,
            ),
          ),
        ),
      ],
    );
  }

  Widget get sampleVolumeInput {

    List<String> dropdownlist_samplevol = [
      for (var i = 1; i <= 100; i++) ('$i ml'),
    ];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.only(right: 18.0, left: 32),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  'Sample volume',
                  style: Theme.of(context).textTheme.subtitle1,
                ),
              )
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.only(
            right: 32,
            left: 32,
            top: 4,
            bottom: 8,
          ),
          child:
          DropdownButton(
            value: _sampleVol,
            icon: const Icon(Icons.arrow_downward),
            elevation: 4,
            itemHeight: 48.0,
            menuMaxHeight: 200,
            isExpanded: true,
            style: const TextStyle(color: Colors.deepPurple),
            underline: Container(
              height: 1,
              color: Colors.deepPurpleAccent,
            ),
            onChanged: (String? newValue) {
              setState(() {
                _sampleVol = newValue!;
              });
            },
            items: dropdownlist_samplevol
                .map<DropdownMenuItem<String>>((String value) {
              return DropdownMenuItem<String>(
                value: value,
                child: Text(value.toString()),
              );
            }).toList(),
          ),
        ),
      ],
    );
  }

  Widget get commentInput => TextFormField(
        onChanged: (input) => setState(() => _comment = input.trim()),
        decoration: InputDecoration(
          hintText: 'Some text',
          labelText: 'Sample comment (optional)',
        ),
      );

  Widget get sampleIDCheckIndicator {
    final style = Theme.of(context).textTheme.button!;

    return Row(
      children: [
        Container(
          margin: const EdgeInsets.only(right: 8, left: 8),
          height: style.fontSize,
          width: style.fontSize,
          child: CircularProgressIndicator(
            strokeWidth: 2,
            valueColor: AlwaysStoppedAnimation(style.color),
          ),
        ),
        Expanded(
          child: Padding(
            padding: EdgeInsets.only(right: 8 + style.fontSize!),
            child: Text(
              'Checking sample ID…',
              style: style,
              textAlign: TextAlign.center,
            ),
          ),
        )
      ],
    );
  }

  Widget get continueButton {
    bool validForm = _sampleId != null;
    validForm &= locationValidator(_location) == null;

    return FutureBuilder(
      future: _idCheck,
      builder: (_, AsyncSnapshot state) => ValueListenableBuilder(
        valueListenable: Bluetooth.of(context)!.connected,
        builder: (_, bool connected, __) {
          bool active = validForm &&
              connected &&
              state.connectionState != ConnectionState.waiting;

          return Builder(
            builder: (BuildContext c) => ElevatedButton(
              onPressed: active ? () => continueWithSample(c) : null,
              child: state.connectionState != ConnectionState.waiting
                  ? Text(
                      connected ? 'Submit sample' : 'Not connected',
                      style: Theme.of(context).textTheme.button,
                    )
                  : sampleIDCheckIndicator,
            ),
          );
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Submit Sample')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.only(top: 32, bottom: 32),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              sampleIdInput,
              locationInput,
              analysisTypePicker,
              if (_analysisType == 'Chlorine') ...[
                _createMetricInput('pH', 'Enter pH value', (value) => _ph = double.tryParse(value)),
                _createMetricInput('TDS', 'Enter TDS value', (value) => _tds = double.tryParse(value)),
                _createMetricInput('Turbidity (NTU)', 'Enter Turbidity value', (value) => _turbidity = double.tryParse(value)),
                _createMetricInput('Salinity', 'Enter Salinity value', (value) => _salinity = double.tryParse(value)),
                _createMetricInput('Conductivity', 'Enter Conductivity value', (value) => _conductivity = double.tryParse(value)),
                _createMetricInput('ORP (mV)', 'Enter ORP value', (value) => _orp = double.tryParse(value)),
                _createMetricInput('Specific Gravity', 'Enter Specific Gravity value', (value) => _specificGravity = double.tryParse(value)),
                _createMetricInput('Water Temperature (°C)', 'Enter Water Temperature (°C)', (value) => _waterTemperature = double.tryParse(value)),
              ],
              reminderInput2,
              commentInput,
              continueButton,
            ].map((Widget child) {
              double horizontalPadding = child is Column ? 0 : 32;

              return Padding(
                padding: EdgeInsets.only(
                  bottom: child is ListTile ? 16 : 32,
                  left: horizontalPadding,
                  right: horizontalPadding,
                ),
                child: child,
              );
            }).toList(),
          ),
        ),
      ),
    );
  }
  }
