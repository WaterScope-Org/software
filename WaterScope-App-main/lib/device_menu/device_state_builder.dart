import 'package:flutter/material.dart';

class DeviceStatusBuilder extends StatelessWidget {
  const DeviceStatusBuilder({
    Key? key,
    required this.connected,
    @required this.updateState,
    required this.builder,
  }) : super(key: key);

  final Future<void>? updateState;
  final ValueNotifier<bool> connected;
  final Widget Function(BuildContext c, bool connected, bool updating) builder;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder(
      valueListenable: connected,
      builder: (_, bool connected, __) {
        if (updateState != null) {
          return FutureBuilder(
            future: updateState,
            builder: (_, AsyncSnapshot state) {
              return builder(
                context,
                connected,
                state.connectionState != ConnectionState.done,
              );
            },
          );
        } else {
          return builder(context, connected, false);
        }
      },
    );
  }
}
