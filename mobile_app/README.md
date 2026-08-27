# mobile_app - folder map

Flutter Android client. Kotlin captures camera frames and runs pose estimation; everything else is Dart.

## Top level

| Folder | Contents |
|---|---|
| `lib/` | All Dart code |
| `android/` | Android project, including the Kotlin native layer |
| `test/` | Dart tests - `cues`, `rep_detector`, `storage` |

## Inside `lib/`

| File / folder | Role |
|---|---|
| `main.dart` | Entry point and app shell |
| `config.dart` | Constants - server address, fallback frame rate, thresholds |
| `pose/` | Receives landmark frames from Kotlin over the event channel |
| `analysis/` | On-device movement analysis: joint angles (`live_angles`), repetition detection (`rep_detector`), spoken cues (`cues`) |
| `api/` | HTTP client for the Python scoring service |
| `state/` | App-wide state, profile and session history in memory |
| `storage/` | Local persistence - session records written to disk as JSON |
| `screens/` | One file per page |
| `widgets/` | Reusable UI, currently the skeleton overlay |

### Screens

| File | Page |
|---|---|
| `home_shell.dart` | Bottom navigation container |
| `exercise_screen.dart` | Pick an exercise and start a set |
| `record_screen.dart` | Live camera, skeleton overlay, real-time cues |
| `summary_screen.dart` | Results for a finished set |
| `history_screen.dart` | Past sessions, stored locally |
| `profile_screen.dart` | Name, age, and body measurements |
| `coach_screen.dart` | Adaptive progression advice |
| `video_screen.dart` | Scoring a pre-recorded clip, for testing |

## Native layer

`android/app/src/main/kotlin/ua/lpnu/intelligent_rehab/` holds two files:

| File | Role |
|---|---|
| `MainActivity.kt` | Registers the event and method channels |
| `PoseStreamHandler.kt` | CameraX capture plus MediaPipe pose estimation |

Kotlin does capture and pose estimation only. It emits raw landmarks with their
capture timestamps and knows nothing about repetitions, errors or scoring - every
algorithm lives in Dart or in the Python service.
