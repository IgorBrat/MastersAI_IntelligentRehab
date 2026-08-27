package ua.lpnu.intelligent_rehab

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.LifecycleOwner
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** Streaming camera frames through MediaPipe PoseLandmarker to Flutter */
class PoseStreamHandler(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
) : EventChannel.StreamHandler, MethodChannel.MethodCallHandler {

    private var events: EventChannel.EventSink? = null
    private var landmarker: PoseLandmarker? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var analysisExecutor: ExecutorService? = null
    private var analysis: ImageAnalysis? = null

    /** Guarding the landmarker against being closed while a frame is being submitted */
    private val landmarkerLock = Any()

    /** False once teardown has begun, so queued frames are dropped rather than processed */
    @Volatile
    private var running = false

    /** Lens facing, chosen from Dart. Front by default: the patient watches the screen */
    private var useFrontCamera = true

    /** Frames analysed so far, used to decimate the preview stream */
    private var previewCounter = 0

    /** Size of the last upright frame, so Dart can scale the overlay to the image */
    private var lastFrameSize: Pair<Int, Int>? = null

    /** `lite` on a phone; `full` is in assets but roughly 3x the cost */
    private var modelAsset = "pose_landmarker_lite.task"

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "configure" -> {
                useFrontCamera = call.argument<Boolean>("frontCamera") ?: true
                modelAsset = call.argument<String>("model") ?: modelAsset
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    override fun onListen(arguments: Any?, sink: EventChannel.EventSink) {
        events = sink
        running = true
        try {
            startLandmarker()
            startCamera()
        } catch (t: Throwable) {
            Log.e(TAG, "failed to start pose stream", t)
            running = false
            sink.error("start_failed", t.message, null)
        }
    }

    override fun onCancel(arguments: Any?) {
        stop()
        events = null
    }

    /**
     * Tearing down camera and inference, safe to call twice or mid-frame
     *
     * The ORDER is the point: clear `running`, detach the analyser, drain the
     * executor, and only then close the native landmarker. `shutdown()` refuses new
     * tasks but does not wait for the running one, so closing before the drain
     * leaves `detectAsync` holding a freed graph.
     */
    fun stop() {
        running = false

        analysis?.clearAnalyzer()
        analysis = null
        cameraProvider?.unbindAll()
        cameraProvider = null

        val executor = analysisExecutor
        analysisExecutor = null
        if (executor != null) {
            executor.shutdown()
            try {
                // Bounded: a frame takes ~30 ms, so this returns almost immediately. The
                // timeout exists only so a wedged analyser cannot hang the main thread.
                executor.awaitTermination(500, TimeUnit.MILLISECONDS)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }

        synchronized(landmarkerLock) {
            landmarker?.close()
            landmarker = null
        }
    }

    private fun startLandmarker() {
        val base = BaseOptions.builder()
            .setModelAssetPath(modelAsset)
            // GPU is markedly faster but falls back inconsistently across vendors;
            // CPU keeps the prototype predictable. `lite` on CPU sustains 30 fps.
            .setDelegate(Delegate.CPU)
            .build()

        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(0.5f)
            .setMinPosePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener { result, _ -> emit(result) }
            .setErrorListener { error -> Log.e(TAG, "landmarker error", error) }
            .build()

        landmarker = PoseLandmarker.createFromOptions(context, options)
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun startCamera() {
        val executor = Executors.newSingleThreadExecutor()
        analysisExecutor = executor
        val future = ProcessCameraProvider.getInstance(context)

        future.addListener({
            val provider = future.get()

            // The provider future resolves asynchronously, so the stream may already have
            // been cancelled by the time this runs -- leaving a camera bound to a handler
            // whose landmarker is closed, feeding an executor that has been shut down.
            if (!running) {
                provider.unbindAll()
                return@addListener
            }
            cameraProvider = provider

            val imageAnalysis = ImageAnalysis.Builder()
                // Dropping frames is correct here. Queueing them would grow latency
                // without bound whenever inference briefly falls behind capture, and
                // stale feedback is worse than sparser feedback.
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            imageAnalysis.setAnalyzer(executor) { proxy -> analyse(proxy) }
            analysis = imageAnalysis

            val selector = CameraSelector.Builder()
                .requireLensFacing(
                    if (useFrontCamera) CameraSelector.LENS_FACING_FRONT
                    else CameraSelector.LENS_FACING_BACK
                )
                .build()

            provider.unbindAll()
            provider.bindToLifecycle(lifecycleOwner, selector, imageAnalysis)
        }, ContextCompat_getMainExecutor(context))
    }

    private fun analyse(proxy: ImageProxy) {
        if (!running) {
            proxy.close()
            return
        }
        try {
            // Read everything off the proxy BEFORE closing it. `imageInfo` is invalid
            // once the frame is released, so touching it afterwards yields stale or
            // zero values - and a zero timestamp makes LIVE_STREAM reject the frame.
            val rotation = proxy.imageInfo.rotationDegrees
            val timestampMs = proxy.imageInfo.timestamp / 1_000_000
            val raw = copyToBitmap(proxy)
            proxy.close()

            // Rotate to upright HERE rather than passing rotationDegrees to MediaPipe.
            //
            // Both work for the pose estimate, but only this gives ONE coordinate space.
            // The preview frame sent to Dart and the landmarks drawn over it must agree,
            // and if MediaPipe rotates internally while Dart receives the unrotated frame
            // they do not -- which is precisely the confusing mismatch to eliminate.
            //
            // Orientation matters beyond the overlay: the camera sensor is mounted
            // landscape, so a phone held upright delivers frames rotated 90 degrees. In
            // that frame gravity points sideways, and every angle measured against
            // vertical -- trunk lean, forward flexion, elevation -- is wrong.
            val upright = rotateUpright(raw, rotation)
            lastFrameSize = upright.width to upright.height

            // Submitting the frame and closing the landmarker are mutually exclusive. The
            // null check must be INSIDE the lock: reading the reference and using it as
            // two separate steps is exactly the window that crashed.
            synchronized(landmarkerLock) {
                val current = landmarker
                if (current == null || !running) return
                current.detectAsync(BitmapImageBuilder(upright).build(), timestampMs)
            }
            maybeSendPreview(upright)
        } catch (t: Throwable) {
            Log.e(TAG, "frame analysis failed", t)
            runCatching { proxy.close() }
        }
    }

    /** Rotating a frame to upright, returning the input untouched when already upright */
    private fun rotateUpright(source: Bitmap, degrees: Int): Bitmap {
        if (degrees % 360 == 0) return source
        val matrix = android.graphics.Matrix().apply { postRotate(degrees.toFloat()) }
        val rotated = Bitmap.createBitmap(
            source, 0, 0, source.width, source.height, matrix, true
        )
        if (rotated != source) source.recycle()
        return rotated
    }

    /** Sending a downscaled JPEG of the frame to Dart for the preview */
    private fun maybeSendPreview(frame: Bitmap) {
        previewCounter += 1
        if (previewCounter % PREVIEW_EVERY_N != 0) return
        val scale = PREVIEW_WIDTH.toFloat() / frame.width.coerceAtLeast(1)
        val small = if (scale >= 1f) frame else Bitmap.createScaledBitmap(
            frame, PREVIEW_WIDTH, (frame.height * scale).toInt().coerceAtLeast(1), true
        )
        val stream = java.io.ByteArrayOutputStream()
        small.compress(Bitmap.CompressFormat.JPEG, PREVIEW_QUALITY, stream)
        if (small != frame) small.recycle()
        postToMain(mapOf("preview" to stream.toByteArray()))
    }

    /**
     * Copying an RGBA_8888 ImageProxy into a Bitmap, honouring the plane row stride
     *
     * CameraX pads each row to a hardware-friendly stride, so copying the buffer
     * wholesale treats the padding as pixels and shears the image progressively down
     * the frame - which MediaPipe then reads as a bent body.
     */
    private fun copyToBitmap(proxy: ImageProxy): Bitmap {
        val plane = proxy.planes[0]
        val rowStride = plane.rowStride
        val tightStride = proxy.width * plane.pixelStride
        if (rowStride == tightStride) {
            return Bitmap.createBitmap(proxy.width, proxy.height, Bitmap.Config.ARGB_8888)
                .apply { copyPixelsFromBuffer(plane.buffer) }
        }
        // Padded rows: build a bitmap the full stride wide, then crop to the real width.
        val padded = Bitmap.createBitmap(
            rowStride / plane.pixelStride, proxy.height, Bitmap.Config.ARGB_8888
        ).apply { copyPixelsFromBuffer(plane.buffer) }
        val cropped = Bitmap.createBitmap(padded, 0, 0, proxy.width, proxy.height)
        if (cropped != padded) padded.recycle()
        return cropped
    }

    private fun emit(result: PoseLandmarkerResult) {
        val sink = events ?: return
        val world = result.worldLandmarks()
        if (world.isEmpty()) {
            // An explicit no-detection message, not silence: Dart needs to tell
            // "person left the frame" apart from "the stream stalled".
            postToMain(mapOf("detected" to false))
            return
        }

        val landmarks = world[0]
        val flatWorld = FloatArray(landmarks.size * 3)
        val visibility = FloatArray(landmarks.size)
        landmarks.forEachIndexed { i, lm ->
            flatWorld[i * 3] = lm.x()
            flatWorld[i * 3 + 1] = lm.y()
            flatWorld[i * 3 + 2] = lm.z()
            visibility[i] = lm.visibility().orElse(1.0f)
        }

        // Image-space landmarks, for the overlay only. Never for measurement.
        val screen = result.landmarks().firstOrNull()
        val flatScreen = FloatArray((screen?.size ?: 0) * 2)
        screen?.forEachIndexed { i, lm ->
            flatScreen[i * 2] = lm.x()
            flatScreen[i * 2 + 1] = lm.y()
        }

        val (frameWidth, frameHeight) = lastFrameSize ?: (0 to 0)
        postToMain(
            mapOf(
                "detected" to true,
                "timestampMs" to result.timestampMs(),
                "frameWidth" to frameWidth,
                "frameHeight" to frameHeight,
                "world" to flatWorld,
                "visibility" to visibility,
                "screen" to flatScreen,
            )
        )
    }

    /**
     * Delivering one event on the main thread, dropping it if the sink is gone
     *
     * The result listener runs off the main thread and EventSink is not thread-safe.
     * The sink is re-checked inside the posted block because a result can be produced,
     * posted, and only then have the stream cancelled.
     */
    private fun postToMain(payload: Map<String, Any>) {
        val sink = events ?: return
        android.os.Handler(android.os.Looper.getMainLooper()).post {
            if (events === sink) {
                runCatching { sink.success(payload) }
                    .onFailure { Log.w(TAG, "dropped event after teardown", it) }
            }
        }
    }

    private fun ContextCompat_getMainExecutor(context: Context) =
        androidx.core.content.ContextCompat.getMainExecutor(context)

    companion object {
        private const val TAG = "PoseStreamHandler"
        const val EVENT_CHANNEL = "ua.lpnu.intelligent_rehab/pose_stream"
        const val METHOD_CHANNEL = "ua.lpnu.intelligent_rehab/pose_control"

        /** Preview width in pixels. Enough to see framing; small enough to be cheap. */
        private const val PREVIEW_WIDTH = 360

        /** JPEG quality for the preview. It is a viewfinder, not a recording. */
        private const val PREVIEW_QUALITY = 70

        /** Send one preview frame per this many analysed frames. */
        private const val PREVIEW_EVERY_N = 2
    }
}
