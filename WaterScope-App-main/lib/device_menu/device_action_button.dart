import 'package:animated_widgets/animated_widgets.dart';
import 'package:flutter/material.dart';

import 'device_state_builder.dart';

class DeviceActionButton extends StatelessWidget {
  const DeviceActionButton({
    Key? key,
    required ValueNotifier<bool> connected,
    @required this.onTap,
    required this.text,
    required this.infoText,
    required this.icon,
    @required this.updateState,
    this.connectionRequired = true,
  })  : _connected = connected,
        super(key: key);

  final ValueNotifier<bool> _connected;
  final VoidCallback? onTap;
  final String text;
  final String infoText;
  final IconData icon;
  final bool connectionRequired;
  final Future<void>? updateState;

  Widget createTile(bool enabled) => Padding(
        padding: const EdgeInsets.only(bottom: 4.0),
        child: Material(
          color: Colors.white,
          child: InkWell(
            onTap: enabled ? onTap : null,
            child: ListTile(
              contentPadding: const EdgeInsets.only(
                right: 32,
                left: 32,
                bottom: 8,
                top: 8,
              ),
              title: Text(
                text,
                style: !enabled ? TextStyle(color: Colors.grey.shade400) : null,
              ),
              subtitle: Text(
                infoText,
                style: !enabled ? TextStyle(color: Colors.grey.shade300) : null,
              ),
              leading: enabled
                  ? ShakeAnimatedWidget(
                      duration: Duration(milliseconds: 1500),
                      shakeAngle: Rotation.deg(z: 10),
                      curve: Curves.linear,
                      child: Icon(icon, color: Colors.blue, size: 35),
                    )
                  : Icon(icon, color: Colors.grey.shade300, size: 35),
            ),
          ),
        ),
      );

  @override
  Widget build(BuildContext context) {
    return DeviceStatusBuilder(
      connected: _connected,
      updateState: updateState,
      builder: (_, connected, updating) {
        return createTile((connected && !updating) || !connectionRequired);
      },
    );
  }
}
