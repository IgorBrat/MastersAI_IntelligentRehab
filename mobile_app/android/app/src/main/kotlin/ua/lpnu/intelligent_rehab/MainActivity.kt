package ua.lpnu.intelligent_rehab

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

    private var poseHandler: PoseStreamHandler? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        val handler = PoseStreamHandler(applicationContext, this)
        poseHandler = handler

        EventChannel(flutterEngine.dartExecutor.binaryMessenger, PoseStreamHandler.EVENT_CHANNEL)
            .setStreamHandler(handler)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, PoseStreamHandler.METHOD_CHANNEL)
            .setMethodCallHandler(handler)
    }

    override fun onDestroy() {
        // CameraX follows this activity's lifecycle but the landmarker does not:
        // leaving it open leaks a native inference graph across a rotation.
        poseHandler?.stop()
        poseHandler = null
        super.onDestroy()
    }
}
