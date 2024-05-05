import 'dart:async';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';

import 'SelectBondedDevicePage.dart';
import 'device_menu/action_page.dart';
import 'instructions_pdf.dart';

class MainPage extends StatefulWidget {
  @override
  _MainPage createState() => new _MainPage();
}

class _MainPage extends State<MainPage> {
  BluetoothState _bluetoothState = BluetoothState.UNKNOWN;

  Timer? _discoverableTimeoutTimer;

  @override
  void initState() {
    super.initState();

    // Get current state
    FlutterBluetoothSerial.instance.state.then((state) {
      setState(() {
        _bluetoothState = state;
      });
    });

    // Listen for futher state changes
    FlutterBluetoothSerial.instance
        .onStateChanged()
        .listen((BluetoothState state) {
      setState(() {
        _bluetoothState = state;

        // Discoverable mode is disabled when Bluetooth gets disabled
        _discoverableTimeoutTimer = null;
      });
    });
  }

  @override
  void dispose() {
    FlutterBluetoothSerial.instance.setPairingRequestHandler(null);
    _discoverableTimeoutTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final raisedButtonStyle = ElevatedButton.styleFrom(
      primary: Colors.indigo[900],
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(20)),
      ),
    );

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.indigo[900],
        title: const Text('WaterScope App'),
      ),
      body: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Column(
              children: [
                Padding(
                  padding: const EdgeInsets.only(top: 40.0),
                  child: Image(
                    image: AssetImage('images/logo.png'),
                    width: 250,
                    height: 250,
                  ),
                ),
              ],
            ),
            Padding(
              padding: const EdgeInsets.only(top: 0, bottom: 0),
              child: ElevatedButton(
                style: raisedButtonStyle,
                child: const Text(
                  'Connect to a WaterScope device',
                  style: TextStyle(color: Colors.white),
                ),
                onPressed: () async {
                  final BluetoothDevice? selectedDevice =
                      await Navigator.of(context).push(
                    MaterialPageRoute(
                      settings: RouteSettings(name: 'device-selector'),
                      builder: (context) {
                        return SelectBondedDevicePage(checkAvailability: false);
                      },
                    ),
                  );

                  if (selectedDevice != null) {
                    print('Connect -> selected ' + selectedDevice.address);
                    _startChat(context, selectedDevice);
                  } else {
                    print('Connect -> no device selected');
                  }
                },
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 0, bottom: 50.0),
              child: ElevatedButton(
                style: raisedButtonStyle,
                onPressed: () {
                  Navigator.of(context).push(
                    MaterialPageRoute(
                      settings: RouteSettings(name: 'instructions'),
                      builder: (context) {
                        return Instructions();
                      },
                    ),
                  );
                },
                child: const Text(
                  'Read Instructions Manual',
                  style: TextStyle(color: Colors.white),
                ),
              ),
            ),
            Container(
              padding: EdgeInsets.only(top: 0.0),
              //color: Colors.green,
              child: Column(
                children: [
                  Divider(),
                  ListTile(
                    title: const Text(
                      'Device bluetooth settings',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ),
                  SwitchListTile(
                    activeColor: Colors.indigo[900],
                    title: const Text('Enable Bluetooth'),
                    value: _bluetoothState.isEnabled,
                    onChanged: (bool value) {
                      // Do the request and update with the true value then
                      future() async {
                        // async lambda seems to not working
                        if (value)
                          await FlutterBluetoothSerial.instance.requestEnable();
                        else
                          await FlutterBluetoothSerial.instance
                              .requestDisable();
                      }

                      future().then((_) {
                        setState(() {});
                      });
                    },
                  ),
                  ListTile(
                    title: const Text('Device Bluetooth'),
                    trailing: ElevatedButton(
                      style: raisedButtonStyle,
                      child: const Text(
                        'Settings',
                        style: TextStyle(color: Colors.white),
                      ),
                      onPressed: () {
                        FlutterBluetoothSerial.instance.openSettings();
                      },
                    ),
                  ),
                ],
              ),
            )
          ],
        ),
      ),
    );
  }

  void _startChat(BuildContext context, BluetoothDevice server) {
    Navigator.of(context).push(
      MaterialPageRoute(
        settings: RouteSettings(name: 'device-actions'),
        builder: (context) {
          return ActionPage(server: server);
        },
      ),
    );
  }
}
