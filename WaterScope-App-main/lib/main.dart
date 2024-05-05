import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_crashlytics/firebase_crashlytics.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
//import 'package:flutter_native_timezone/flutter_native_timezone.dart';
import 'package:timezone/data/latest.dart' as tz;
import 'package:timezone/timezone.dart' as tz;

import 'MainPage.dart';


void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  tz.initializeTimeZones();
  tz.setLocalLocation(tz.getLocation('Europe/London'));
  /*FlutterNativeTimezone.getLocalTimezone().then((String timeZoneName) {
    tz.setLocalLocation(tz.getLocation(timeZoneName));
  });*/
  await Firebase.initializeApp(options: FirebaseOptions(
    apiKey: "AIzaSyB32X2af9w9yyvqS8echd3Sq2RKYoxuKS0",
    appId: "1:352023970714:android:3465c564755f6151f3b023",
    messagingSenderId: "352023970714",
    projectId: "waterscope-4dd2f",
  ),);
  runApp(Application());
}

Future<void> initCrashlytics() async {
  String debugCrashlytics = const String.fromEnvironment(
    'CRASH_DEBUG',
  ).toLowerCase();

  await FirebaseCrashlytics.instance.setCrashlyticsCollectionEnabled(
    !kDebugMode || debugCrashlytics == 'true',
  );

  FlutterError.onError = FirebaseCrashlytics.instance.recordFlutterError;
}

class Application extends StatelessWidget {
  const Application({Key? key}) : super(key: key);


  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      future: Firebase.initializeApp(options: FirebaseOptions(
        apiKey: "AIzaSyB32X2af9w9yyvqS8echd3Sq2RKYoxuKS0",
        appId: "1:352023970714:android:3465c564755f6151f3b023",
        messagingSenderId: "352023970714",
        projectId: "waterscope-4dd2f",
      ),),
      builder: (_, AsyncSnapshot state) {
        Widget home;
        List<NavigatorObserver> navigationObservers = [];

        if (state.connectionState != ConnectionState.done) {
          home = Container(
            color: Colors.grey.shade200,
            alignment: Alignment.center,
            child: const CircularProgressIndicator(),
          );
        } else {
          initCrashlytics();

          FirebaseFirestore.instance.settings = Settings(
            cacheSizeBytes: Settings.CACHE_SIZE_UNLIMITED,
          );

          const String firestoreHost = String.fromEnvironment('FIREBASE_HOST');
          if (firestoreHost.isNotEmpty) {
            FirebaseFirestore.instance.settings = Settings(
              host: firestoreHost,
              sslEnabled: false,
              cacheSizeBytes:
                  FirebaseFirestore.instance.settings.cacheSizeBytes,
            );
          }

          home = MainPage();
          navigationObservers = [
            FirebaseAnalyticsObserver(analytics: FirebaseAnalytics.instance),
          ];
        }

        final theme = ThemeData();

        const minimumButtonSize = Size(88, 36);
        const buttonPadding = const EdgeInsets.symmetric(horizontal: 16.0);
        final buttonShape = RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16.0),
        );

        return MaterialApp(
          home: home,
          navigatorObservers: navigationObservers,
          theme: ThemeData(
            primaryColor: Colors.indigo[900],
            // still used by the pull to refresh indicator
            textTheme: TextTheme(
              labelLarge: TextStyle(color: Colors.white),
            ),
            toggleableActiveColor: Colors.indigo[900],
            colorScheme: theme.colorScheme.copyWith(
              primary: Colors.indigo[900],
              secondary: Colors.indigo[900],
            ),
            outlinedButtonTheme: OutlinedButtonThemeData(
              style: OutlinedButton.styleFrom(
                minimumSize: minimumButtonSize,
                shape: buttonShape,
              ),
            ),
            elevatedButtonTheme: ElevatedButtonThemeData(
              style: ElevatedButton.styleFrom(
                foregroundColor: Colors.black87, backgroundColor: Colors.indigo[900], minimumSize: minimumButtonSize,
                padding: buttonPadding,
                shape: buttonShape,
              ),
            ),
          ),
        );
      },
    );
  }
}
