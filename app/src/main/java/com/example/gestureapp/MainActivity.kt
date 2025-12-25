package com.example.gestureapp

import android.Manifest
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.gestureapp.databinding.ActivityMainBinding
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.*

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var handDetector: Interpreter
    private lateinit var handLandmark: Interpreter

    // Properties for Temporal Smoothing to prevent flicker
    private val gestureHistory = mutableListOf<String>()
    private val HISTORY_SIZE = 5
    private val STABILITY_THRESHOLD = 3
    private var stableGesture: String = "No hand detected"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        requestPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                loadModels()
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
                finish()
            }
        }

    private fun loadModels() {
        handDetector = Interpreter(loadModelFile("hand_detector.tflite"))
        handLandmark = Interpreter(loadModelFile("hand_landmarks_detector.tflite"))
        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun getInputShape(interpreter: Interpreter): Triple<Int, Int, DataType> {
        val tensor = interpreter.getInputTensor(0)
        val shape = tensor.shape()
        val dataType = tensor.dataType()
        return Triple(shape[1], shape[2], dataType)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }
            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        try {
                            val bitmap = imageProxyToBitmap(imageProxy)
                            val resultText = runGesturePipeline(bitmap)
                            runOnUiThread { binding.gestureOutputText.text = resultText }
                        } catch (e: Exception) {
                            // Suppress minor errors during rapid analysis
                        } finally {
                            imageProxy.close()
                        }
                    }
                }
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analyzer)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun runGesturePipeline(bitmap: Bitmap): String {
        try {
            // Stage 1: Hand Detection
            val (detectorW, detectorH, detectorType) = getInputShape(handDetector)
            val detectorInput = preprocessImage(bitmap, detectorW, detectorH, detectorType)
            val palmDetections = Array(1) { Array(2016) { FloatArray(18) } }
            val detectionScores = Array(1) { Array(2016) { FloatArray(1) } }
            val outputs = mapOf(0 to palmDetections, 1 to detectionScores)
            handDetector.runForMultipleInputsOutputs(arrayOf(detectorInput), outputs)

            val scores = detectionScores[0].map { it[0] }
            val maxIdx = scores.indices.maxByOrNull { scores[it] } ?: -1
            if (maxIdx == -1 || scores[maxIdx] < 0.75f) {
                return getStableGesture("No hand detected")
            }

            // Crop the hand
            val detection = palmDetections[0][maxIdx]
            val cx = detection[0]; val cy = detection[1]
            val w = detection[2]; val h = detection[3]
            val x = (cx - w / 2).coerceAtLeast(0f)
            val y = (cy - h / 2).coerceAtLeast(0f)
            val absX = max((x * bitmap.width).toInt(), 0)
            val absY = max((y * bitmap.height).toInt(), 0)
            val absW = (w * bitmap.width).toInt()
            val absH = (h * bitmap.height).toInt()
            val handCrop = Bitmap.createBitmap(bitmap, absX, absY, min(bitmap.width - absX, absW), min(bitmap.height - absY, absH))

            // Stage 2: Hand Landmark Detection
            val (landmarkW, landmarkH, _) = getInputShape(handLandmark)
            val landmarkInput = preprocessImageToFloatArray(handCrop, landmarkW, landmarkH)

            val imageLandmarksOutput = Array(1) { FloatArray(63) }
            val handednessOutput = Array(1) { FloatArray(1) }
            val extraScoreOutput = Array(1) { FloatArray(1) }
            val worldLandmarksOutput = Array(1) { FloatArray(63) }
            val landmarkOutputs = mapOf(0 to imageLandmarksOutput, 1 to handednessOutput, 2 to extraScoreOutput, 3 to worldLandmarksOutput)
            handLandmark.runForMultipleInputsOutputs(arrayOf(landmarkInput), landmarkOutputs)

            val landmarks = Array(1) { Array(21) { FloatArray(3) } }
            for (i in 0 until 21) {
                landmarks[0][i][0] = imageLandmarksOutput[0][i * 3]
                landmarks[0][i][1] = imageLandmarksOutput[0][i * 3 + 1]
                landmarks[0][i][2] = imageLandmarksOutput[0][i * 3 + 2]
            }

            // Stage 3 & 4: Classify and Stabilize
            val currentGesture = classifyGestureManually(landmarks[0])
            return getStableGesture(currentGesture)

        } catch (e: Exception) {
            return getStableGesture("Gesture Unclear")
        }
    }

    private fun getStableGesture(currentGesture: String): String {
        if (currentGesture == "No hand detected") {
            gestureHistory.clear()
            stableGesture = "No hand detected"
            return stableGesture
        }
        if (gestureHistory.size >= HISTORY_SIZE) {
            gestureHistory.removeAt(0)
        }
        gestureHistory.add(currentGesture)
        val vote = gestureHistory.groupingBy { it }.eachCount()
        val mostCommon = vote.maxByOrNull { it.value }
        if (mostCommon != null && mostCommon.value >= STABILITY_THRESHOLD) {
            if (mostCommon.key != "Gesture Unclear") {
                stableGesture = mostCommon.key
            }
        }
        return stableGesture
    }

    // Landmark constants
    private val THUMB_CMC = 1; private val THUMB_MCP = 2; private val THUMB_IP = 3
    private val INDEX_MCP = 5; private val INDEX_PIP = 6; private val INDEX_TIP = 8
    private val MIDDLE_MCP = 9; private val MIDDLE_PIP = 10; private val MIDDLE_TIP = 12
    private val RING_MCP = 13; private val RING_PIP = 14; private val RING_TIP = 16
    private val PINKY_MCP = 17; private val PINKY_PIP = 18; private val PINKY_TIP = 20

    private fun getAngle(p1: FloatArray, p2: FloatArray, p3: FloatArray): Double {
        val dx1 = (p1[0] - p2[0]).toDouble()
        val dy1 = (p1[1] - p2[1]).toDouble()
        val dx2 = (p3[0] - p2[0]).toDouble()
        val dy2 = (p3[1] - p2[1]).toDouble()
        val angle = atan2(dy1, dx1) - atan2(dy2, dx2)
        var degrees = Math.toDegrees(angle)
        if (degrees < 0) degrees += 360.0
        if (degrees > 180) degrees = 360.0 - degrees // Get the smaller interior angle
        return degrees
    }

    private fun classifyGestureManually(landmarks: Array<FloatArray>): String {
        if (landmarks.isEmpty()) return "Gesture Unclear"

        // ** THRESHOLDS - You can tune these values **
        val straightnessThreshold = 150.0 // Angle greater than this is "straight" for the 4 fingers
        val thumbStraightnessThreshold = 130.0 // Thumb has a different range of motion

        // Calculate angles for each finger
        val indexAngle = getAngle(landmarks[INDEX_MCP], landmarks[INDEX_PIP], landmarks[INDEX_TIP])
        val middleAngle = getAngle(landmarks[MIDDLE_MCP], landmarks[MIDDLE_PIP], landmarks[MIDDLE_TIP])
        val ringAngle = getAngle(landmarks[RING_MCP], landmarks[RING_PIP], landmarks[RING_TIP])
        val pinkyAngle = getAngle(landmarks[PINKY_MCP], landmarks[PINKY_PIP], landmarks[PINKY_TIP])
        val thumbAngle = getAngle(landmarks[THUMB_CMC], landmarks[THUMB_MCP], landmarks[THUMB_IP])

        // Determine if each finger is straight
        val indexIsStraight = indexAngle > straightnessThreshold
        val middleIsStraight = middleAngle > straightnessThreshold
        val ringIsStraight = ringAngle > straightnessThreshold
        val pinkyIsStraight = pinkyAngle > straightnessThreshold
        val thumbIsStraight = thumbAngle > thumbStraightnessThreshold

        // ** VITAL DEBUGGING LOG **
        // Look at this in Logcat to see the live data and tune thresholds.
        val fingerAngles = "Angles -> T:${thumbAngle.toInt()} I:${indexAngle.toInt()} M:${middleAngle.toInt()} R:${ringAngle.toInt()} P:${pinkyAngle.toInt()}"
        Log.d(TAG, fingerAngles)

        // Define gestures with strict, mutually exclusive rules.
        return when {
            // Rule for Open Palm: All 5 fingers must be straight.
            thumbIsStraight && indexIsStraight && middleIsStraight && ringIsStraight && pinkyIsStraight -> "Open_Palm"
            // Rule for Thumb Up: Thumb is straight, others are curled.
            thumbIsStraight && !indexIsStraight && !middleIsStraight && !ringIsStraight && !pinkyIsStraight -> "Thumb_Up"
            // Rule for Closed Fist: No fingers are straight.
            !thumbIsStraight && !indexIsStraight && !middleIsStraight && !ringIsStraight && !pinkyIsStraight -> "Closed_Fist"
            // Default case
            else -> "Gesture Unclear"
        }
    }

    // --- All other helper functions with full implementations ---

    private fun preprocessImage(bitmap: Bitmap, width: Int, height: Int, dataType: DataType): ByteBuffer {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val intValues = IntArray(width * height)
        scaledBitmap.getPixels(intValues, 0, width, 0, 0, width, height)
        val bytePerChannel = if (dataType == DataType.FLOAT32) 4 else 1
        val inputBuffer = ByteBuffer.allocateDirect(bytePerChannel * width * height * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        for (pixel in intValues) {
            val r = (pixel shr 16 and 0xFF); val g = (pixel shr 8 and 0xFF); val b = (pixel and 0xFF)
            if (dataType == DataType.FLOAT32) {
                inputBuffer.putFloat(r / 255f); inputBuffer.putFloat(g / 255f); inputBuffer.putFloat(b / 255f)
            } else {
                inputBuffer.put(r.toByte()); inputBuffer.put(g.toByte()); inputBuffer.put(b.toByte())
            }
        }
        inputBuffer.rewind(); return inputBuffer
    }

    private fun preprocessImageToFloatArray(bitmap: Bitmap, width: Int, height: Int): Array<Array<Array<FloatArray>>> {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val floatInput = Array(1) { Array(height) { Array(width) { FloatArray(3) } } }
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = scaledBitmap.getPixel(x, y)
                floatInput[0][y][x][0] = (Color.red(pixel) / 255.0f); floatInput[0][y][x][1] = (Color.green(pixel) / 255.0f); floatInput[0][y][x][2] = (Color.blue(pixel) / 255.0f)
            }
        }
        return floatInput
    }

    private fun loadModelFile(filename: String): ByteBuffer {
        val fileDescriptor = assets.openFd(filename)
        val inputStream = fileDescriptor.createInputStream()
        val fileSize = fileDescriptor.length.toInt()
        val buffer = ByteArray(fileSize)
        inputStream.read(buffer)
        val byteBuffer = ByteBuffer.allocateDirect(fileSize); byteBuffer.order(ByteOrder.nativeOrder()); byteBuffer.put(buffer); byteBuffer.rewind()
        return byteBuffer
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer; val vuBuffer = imageProxy.planes[2].buffer
        val ySize = yBuffer.remaining(); val vuSize = vuBuffer.remaining()
        val nv21 = ByteArray(ySize + vuSize); yBuffer.get(nv21, 0, ySize); vuBuffer.get(nv21, ySize, vuSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream(); yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        handDetector.close()
        handLandmark.close()
    }

    companion object {
        private const val TAG = "HandGestureRecognizer"
    }
}