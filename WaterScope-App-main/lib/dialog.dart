import 'package:flutter/material.dart';

class Dialog extends StatelessWidget {
  const Dialog({
    Key? key,
    this.content,
    required this.title,
    @required this.onSubmission,
    required this.action,
  }) : super(key: key);

  final Widget? content;
  final String title;
  final void Function()? onSubmission;
  final String action;

  @override
  Widget build(BuildContext context) {
    final secondaryColor = Theme.of(context).colorScheme.secondary;

    return AlertDialog(
      titlePadding: const EdgeInsets.only(
        left: 20,
        right: 20,
        top: 20,
        bottom: 8,
      ),
      actionsPadding: const EdgeInsets.only(left: 12, right: 12, top: 8),
      contentPadding: const EdgeInsets.only(left: 20, right: 20),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      title: Text(title),
      content: content != null
          ? SingleChildScrollView(
              child: content,
            )
          : null,
      actions: [
        OutlinedButton(
          style: OutlinedButton.styleFrom(
            side: BorderSide(color: secondaryColor),
          ),
          onPressed: () => Navigator.of(context).pop(),
          child: Text(
            'Cancel',
            style: TextStyle(
              color: secondaryColor,
            ),
          ),
        ),
        ElevatedButton(
          onPressed: onSubmission,
          child: Text(
            action,
            style: Theme.of(context).textTheme.button,
          ),
        ),
      ],
    );
  }
}
