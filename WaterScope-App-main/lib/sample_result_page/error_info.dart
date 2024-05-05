import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

class ErrorInfo extends StatelessWidget {
  const ErrorInfo({
    Key? key,
    this.color,
    required this.errorMessage,
  }) : super(key: key);

  final Color? color;
  final String errorMessage;

  @override
  Widget build(BuildContext context) {
    final TextStyle theme = Theme.of(context).textTheme.bodyText1!;

    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.only(bottom: 16),
            child: Stack(
              alignment: Alignment.center,
              children: [
                Container(
                  height: 25,
                  width: 25,
                  decoration: BoxDecoration(
                    color: color,
                    borderRadius: BorderRadius.circular(32),
                  ),
                ),
                Icon(Icons.error, color: Colors.red, size: 32),
              ],
            ),
          ),
          Text(
            errorMessage,
            textAlign: TextAlign.center,
            style: color != null
                ? TextStyle(
                    color: color,
                    fontWeight: theme.fontWeight,
                    fontStyle: theme.fontStyle,
                    fontSize: theme.fontSize,
                  )
                : theme,
          ),
        ],
      ),
    );
  }
}
