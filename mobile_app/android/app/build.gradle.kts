plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "ua.lpnu.intelligent_rehab"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = "27.0.12077973"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_11.toString()
    }

    defaultConfig {
        applicationId = "ua.lpnu.intelligent_rehab"
        // MediaPipe Tasks needs 24+; Flutter's default floor is lower.
        minSdk = maxOf(flutter.minSdkVersion, 24)

        // arm64 only, and this is what makes the emulator work.
        //
        // mediapipe tasks-vision 0.10.21 ships arm64-v8a, armeabi-v7a and x86 --
        // but NOT x86_64. Flutter's own libflutter.so does have x86_64, so on an
        // x86_64 emulator the APK installs as x86_64, finds no
        // libmediapipe_tasks_vision_jni.so for that ABI, and dies with
        // UnsatisfiedLinkError the moment a PoseLandmarker is created.
        //
        // Restricting to arm64-v8a forces the emulator down its arm64 translation
        // path (modern images list "x86_64,arm64-v8a"), where the MediaPipe library
        // exists. Every real Android phone is arm64 anyway, so nothing is lost --
        // only 32-bit ARM devices are excluded, and MediaPipe would be too slow on
        // those regardless.
        //
        // `flutter run` has no --target-platform flag, so this has to live here
        // rather than on the command line.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    androidResources {
        // The .task bundle is already compressed; letting aapt deflate it makes
        // MediaPipe unable to memory-map it from the APK.
        noCompress += "task"
    }

    buildTypes {
        release {
            // Prototype: signing with the debug keys so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
            // MediaPipe's bundled AutoValue processor classes reference javax.lang.model,
            // which does not exist on Android, so R8 fails the build without these rules.
            // It also resolves task graphs reflectively, hence the keep rules.
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }
}

dependencies {
    implementation("com.google.mediapipe:tasks-vision:0.10.21")

    // CameraX: ImageAnalysis gives frames without us managing Camera2 state.
    val cameraX = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraX")
    implementation("androidx.camera:camera-camera2:$cameraX")
    implementation("androidx.camera:camera-lifecycle:$cameraX")
}

flutter {
    source = "../.."
}
