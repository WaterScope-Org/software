import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:waterscope/bluetooth.dart';

import 'dialog.dart' as DefaultDialog;

class WifiSettings extends StatefulWidget {
  @override
  _WifiSettingsState createState() => _WifiSettingsState();
}

class _WifiSettingsState extends State<WifiSettings> {
  String? _name, _password;

  bool _visible = false;

  String? nameValidator(String? name) {
    final int length = name?.trim().length ?? 0;

    if (length > 32) {
      return 'SSID too long';
    } else if (length == 0) {
      return 'SSID cannot be empty';
    }

    return null;
  }

  String? passwordValidator(String? password) {
    final int length =
        password != null ? utf8.encode(password.trim()).length : 0;

    if (length > 63) {
      return 'Password too long';
    } else if (length == 0) {
      return 'Password cannot be empty';
    } else if (length < 8) {
      return 'Password too short';
    }

    return null;
  }

  bool get validInput =>
      passwordValidator(_password) == null && nameValidator(_name) == null;

  void submitCredentials() {
    final Bluetooth bluetooth = Bluetooth.of(context)!;

    final payload = utf8.encode(json.encode({
      'action': 'add',
      'ssid': _name,
      'password': _password,
    }));

    if (bluetooth.connected.value) {
      Navigator.of(context).pop(BluetoothInstruction.send(
        bluetooth.connection!,
        bluetooth.stream!,
        'wifi',
        payload: Uint8List.fromList(payload),
      ));
    }
  }

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder(
      valueListenable: Bluetooth.of(context)!.connected,
      builder: (_, bool connected, Widget? content) => DefaultDialog.Dialog(
        title: 'Add Wi-Fi network',
        action: 'Submit',
        content: content,
        onSubmission: connected && validInput ? submitCredentials : null,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          TextFormField(
            autofocus: true,
            textInputAction: TextInputAction.next,
            autovalidateMode: AutovalidateMode.onUserInteraction,
            validator: nameValidator,
            onChanged: (name) => setState(() => _name = name.trim()),
            decoration: InputDecoration(
              hintText: 'Example SSID',
              labelText: 'Wi-Fi name',
            ),
          ),
          TextFormField(
            obscureText: !_visible,
            autovalidateMode: AutovalidateMode.onUserInteraction,
            validator: passwordValidator,
            onChanged: (password) => setState(
              () => _password = password.trim(),
            ),
            decoration: InputDecoration(
                hintText: 'Password1234',
                labelText: 'Wi-Fi password',
                suffixIcon: Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: IconButton(
                    onPressed: () => setState(() => _visible = !_visible),
                    icon: Icon(
                      _visible ? Icons.visibility_off : Icons.visibility,
                    ),
                  ),
                )),
          ),
        ]
            .map(
              (Widget child) => Padding(
                padding: EdgeInsets.only(bottom: 8),
                child: child,
              ),
            )
            .toList(),
      ),
    );
  }
}
