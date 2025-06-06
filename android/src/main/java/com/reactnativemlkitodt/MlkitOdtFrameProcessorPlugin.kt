package com.reactnativemlkitodt

import android.content.Context
import android.graphics.*
import android.graphics.Rect
import android.graphics.RectF
import android.net.Uri
import com.facebook.react.bridge.*
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.*
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import com.google.mlkit.vision.objects.custom.CustomObjectDetectorOptions
import com.google.mlkit.common.model.LocalModel
import java.lang.Exception
import org.tensorflow.lite.task.vision.detector.Detection
import com.google.android.odml.image.MediaMlImageBuilder
import com.google.android.odml.image.BitmapMlImageBuilder
import com.google.android.odml.image.BitmapExtractor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import java.nio.ByteBuffer
import android.util.Log
import android.media.Image
import android.media.ImageReader
import androidx.camera.core.ImageProxy
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks

import androidx.annotation.NonNull
import androidx.annotation.Nullable

import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.VisionCameraProxy
import com.facebook.react.bridge.ReactApplicationContext
import java.lang.ref.WeakReference
import com.facebook.react.bridge.ReadableNativeArray
import com.facebook.react.bridge.ReadableNativeMap
import com.facebook.react.bridge.WritableNativeArray
import com.facebook.react.bridge.WritableNativeMap
import android.annotation.SuppressLint
import java.io.ByteArrayOutputStream

import java.io.IOException
import kotlin.text.format
import android.os.Environment

import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MlkitOdtFrameProcessorPlugin(reactContext: ReactApplicationContext, proxy: VisionCameraProxy, options: Map<String, Any>?): FrameProcessorPlugin() {
   
    private val _context:ReactApplicationContext = reactContext
    private var tfObjectDetector: TFObjectDetectorHelper? = null
    
    override fun callback(frame: Frame, params: Map<String, Any>?): Any? {
        val imageProxy = frame.imageProxy
        val bitmap = convertImageProxyToBitmap(frame.imageProxy)
        @SuppressLint("UnsafeOptInUsageError")
        val mediaImage: Image? = imageProxy.image
        Log.d("OB Detector....","${imageProxy.imageInfo.rotationDegrees}");
        try {

            if (params != null && mediaImage != null) {
                
                        val detectorModeRaw = params["detectorMode"]
                val isSingleImageMode = when (detectorModeRaw) {
                    is Number -> detectorModeRaw.toInt()
                    else -> {
                        Log.w("MlkitOdtPlugin", "detectorMode is missing or not a Number (e.g., Double or Int). Received: '$detectorModeRaw'. Defaulting to 0.")
                        0 // Default to STREAM_MODE or equivalent
                    }
                }

                val enableClassificationRaw = params["shouldEnableClassification"]
                val enableClassification = when (enableClassificationRaw) {
                    is Boolean -> enableClassificationRaw
                    else -> {
                        Log.w("MlkitOdtPlugin", "shouldEnableClassification is missing or not a Boolean. Received: '$enableClassificationRaw'. Defaulting to false.")
                        false
                    }
                }

                val enableMultiDetectRaw = params["shouldEnableMultipleObjects"]
                val enableMultiDetect = when (enableMultiDetectRaw) {
                    is Boolean -> enableMultiDetectRaw
                    else -> {
                        Log.w("MlkitOdtPlugin", "shouldEnableMultipleObjects is missing or not a Boolean. Received: '$enableMultiDetectRaw'. Defaulting to false.")
                        false
                    }
                }

                val customModel = params["customModel"] as? String // Expects a String, null if not present or wrong type
                val modelName = params["modelName"] as? String     // Expects a String, null if not present or wrong type

                if(customModel == "automl"){
                    val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
                    //val image = InputImage.fromMediaImage(mediaImage, 0)
                    var localModel = LocalModel.Builder()
                        .setAssetFilePath("custom_models/object_labeler.tflite")
                        // or .setAbsoluteFilePath(absolute file path to model file)
                        // or .setUri(URI to model file)
                        .build()
                    var customOptionsBuilder = CustomObjectDetectorOptions.Builder(localModel)
                    if (isSingleImageMode == 1) {
                    customOptionsBuilder.setDetectorMode(CustomObjectDetectorOptions.SINGLE_IMAGE_MODE)
                    }
                    if (enableClassification == true) {
                    customOptionsBuilder.enableClassification()
                    }
                    if (enableMultiDetect == true) {
                    customOptionsBuilder.enableMultipleObjects()
                    }
                    
                    val customObjectDetector = ObjectDetection.getClient(customOptionsBuilder.build())
                    val task: Task<List<DetectedObject>>  = customObjectDetector.process(image)
                    try {
                        val detectedObjects:List<DetectedObject> = Tasks.await(task);
                        return makeResultObject(detectedObjects)
                        
                    } catch (e: Exception) {
                        return null
                    } finally {
                        customObjectDetector.close()
                    }
                
                }else if(customModel == "tensorflow") {
                    if(tfObjectDetector == null)
                        tfObjectDetector = TFObjectDetectorHelper(0.5f, 2, 3, 0, 2, modelName, _context)
                    //var mlImage = MediaMlImageBuilder(mediaImage).setRotation(0).build()
                    //val bitmap = imageProxyToBitmap(imageProxy) ?: return null
                    
                    // Ensure newImage is not null before using it
                    if (bitmap != null) {
                    //var mlImage = MediaMlImageBuilder(mediaImage).setRotation(imageProxy.imageInfo.rotationDegrees).build()
                    //var mlImage = MediaMlImageBuilder(mediaImage).setRotation(imageProxy.imageInfo.rotationDegrees).build()
                        var mlImage = BitmapMlImageBuilder(bitmap).setRotation(0).build()
                        var results = tfObjectDetector?.detectFrameProcessor(mlImage)
                    return tfMakeResultObject(results)
                    }else{
                         // Handle the error case (e.g., log or return early)
                        Log.e("ImageProcessing", "Failed to rotate image: Image is null")
                        return null;
                    }
                    
                }else{

                    //val image = InputImage.fromMediaImage(mediaImage, frame.imageInfo.rotationDegrees)
                    val image = InputImage.fromMediaImage(mediaImage, 0)
                    var optionsBuilder = ObjectDetectorOptions.Builder()
                    if (isSingleImageMode == 1) {
                    optionsBuilder.setDetectorMode(ObjectDetectorOptions.SINGLE_IMAGE_MODE)
                    }
                    if (enableClassification == true) {
                    optionsBuilder.enableClassification()
                    }
                    if (enableMultiDetect == true) {
                    optionsBuilder.enableMultipleObjects()
                    }
                    
                    val objectDetector = ObjectDetection.getClient(optionsBuilder.build())
                    val task2: Task<List<DetectedObject>>  = objectDetector.process(image)
                    try {
                        val detectedObjects2:List<DetectedObject> = Tasks.await(task2);
                        return makeResultObject(detectedObjects2)
                        
                    } catch (e: Exception) {
                        return null
                    } finally {
                        objectDetector.close()
                    }

                }
            }
            return null;

        }catch (e: Exception) {
            e.printStackTrace()
            return null;
        }
        return null
    }

    private fun tfMakeResultObject(objects: List<Detection>?): List<Map<String, Any?>> {
        if (objects == null) return emptyList()

        val data = mutableListOf<Map<String, Any?>>()
        for ((i, detectedObject) in objects.withIndex()) {
            // Ensure there's at least one category to process
            if (detectedObject.categories.isEmpty()) {
                Log.w("MlkitOdtPlugin", "Detected object has no categories, skipping.")
                continue
            }
            val category = detectedObject.categories[0]

            val labelsData = mutableListOf<Map<String, String?>>()
            labelsData.add(mapOf(
                "text" to category.label,
                "index" to category.index.toString(), // Use actual index if available, otherwise "0" or similar
                "confidence" to String.format("%.2f", category.score)
            ))

            val outputObject = mutableMapOf<String, Any?>(
                "bounding" to tfGetBoundingResult(detectedObject.boundingBox),
                "trackingID" to (category.label + i.toString()),
                "labels" to labelsData
            )
            data.add(outputObject)
        }
    return data
  }

  private fun makeResultObject(objects: List<DetectedObject>): List<Map<String, Any?>> {
        val data = mutableListOf<Map<String, Any?>>()
        var i = 0
        for (detectedObject in objects) {
            val labelsData = mutableListOf<Map<String, String?>>()
            detectedObject.labels.forEach { label ->
                labelsData.add(mapOf(
                    "text" to label.text,
                    "index" to label.index.toString(),
                    "confidence" to label.confidence.toString()
                ))
            }

            val outputObject = mutableMapOf<String, Any?>(
                "bounding" to getBoundingResult(detectedObject.boundingBox),
                "labels" to labelsData
            )
            if (detectedObject.trackingId != null) {
                outputObject["trackingID"] = detectedObject.trackingId?.toString() + i.toString()
            }
            data.add(outputObject)
            i++
        }
        return data
    }

    private fun getBoundingResult(boundingBox: Rect): Map<String, Int> {
        return mapOf(
            "top" to boundingBox.top,
            "left" to boundingBox.left,
            "width" to boundingBox.width(),
            "height" to boundingBox.height()
        )
    }

    private fun tfGetBoundingResult(boundingBox: RectF): Map<String, Int> {
        return mapOf(
            "top" to boundingBox.top.toInt(),
            "left" to boundingBox.left.toInt(),
            "width" to boundingBox.width().toInt(),
            "height" to boundingBox.height().toInt()
        )
    }

    private fun convertImageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val image = imageProxy.image ?: return null
        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val byteArray = out.toByteArray()
        val originalBitmap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.size)

        // Rotate the bitmap based on the rotation degrees
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())

        val rotatedBitmap = Bitmap.createBitmap(
            originalBitmap,
            0,
            0,
            originalBitmap.width,
            originalBitmap.height,
            matrix,
            true
        )
        // Recycle the original bitmap if it's different from the rotated one and no longer needed
        if (originalBitmap != rotatedBitmap) {
            originalBitmap.recycle()
        }


        // --- Save the rotated bitmap to a temporary file ---

        //saveBitmapToFile(_context, rotatedBitmap, "rotated_image.jpg")

        // --- End saving ---

        return rotatedBitmap
    }
    fun rotateBitmap(mediaImage: Image, rotationDegrees: Int): Bitmap? {
        Log.d("ImageProcessing", "Received image for rotation.")
        // Convert mediaImage to Bitmap
        val bitmap = mediaImageToBitmap(mediaImage)
        if (bitmap == null) {
            Log.e("ImageProcessing", "Failed to convert mediaImage to Bitmap")
            return null
        }
        Log.d("ImageProcessing", "Converted Image to Bitmap successfully.")
        // Rotate the Bitmap
        val rotatedBitmap = rotateBitmap(bitmap, rotationDegrees)
        Log.d("ImageProcessing", "Rotated Bitmap successfully.")
        return bitmap
    }

    private fun mediaImageToBitmap(image: Image): Bitmap? {
        try {
            val yBuffer = image.planes[0].buffer // Y
            val uBuffer = image.planes[1].buffer // U
            val vBuffer = image.planes[2].buffer // V

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            // Copy Y plane
            yBuffer.get(nv21, 0, ySize)
            // Copy U and V planes
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
            val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, image.width, image.height, null)
            val outputStream = java.io.ByteArrayOutputStream()
            yuvImage.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 100, outputStream)
            val jpegBytes = outputStream.toByteArray()
            return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
        } catch (e: Exception) {
            Log.e("ImageProcessing", "Exception while converting YUV_420_888 to Bitmap: ${e.message}")
            return null
        }
    }

    // Rotate Bitmap using Matrix
    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    // Convert Bitmap to Image (JPEG format)
    private fun bitmapToImage(bitmap: Bitmap): Image? {
        try {
            val width = bitmap.width
            val height = bitmap.height
            val imageReader = ImageReader.newInstance(width, height, android.graphics.ImageFormat.JPEG, 1)
            val outputImage = imageReader.acquireLatestImage()
            imageReader.close()
            return outputImage
        } catch (e: Exception) {
            Log.e("ImageProcessing", "Exception while converting Bitmap to Image: ${e.message}")
            return null
        }
    }

     private fun saveBitmapToFile(context: Context, bitmap: Bitmap, baseFilename: String? = null) {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmssSSS", Locale.US).format(Date())
        val actualFilename = baseFilename

        // Choose storage location:
        // 1. App-specific cache directory (recommended for temporary files)
        val cacheDir = context.cacheDir
        val file = File(cacheDir, actualFilename)

        // 2. App-specific files directory (for files you want to keep longer but private to app)
        // val filesDir = context.getExternalFilesDir(null) // Or context.filesDir for internal
        // val file = File(filesDir, actualFilename)

        // 3. Public directory (requires more permissions and careful handling of Scoped Storage on Android 10+)
        //    For temporary images, app-specific storage is usually better.
        // val publicDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        // val file = File(publicDir, actualFilename)
        // if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q && !publicDir.exists()) {
        // publicDir.mkdirs()
        // }

        try {
            FileOutputStream(file).use { outStream ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, outStream) // Adjust quality as needed
                outStream.flush()
                Log.d("VisionCameraOCR", "Bitmap saved successfully to: ${file.absolutePath}")
            }
        } catch (e: IOException) {
            Log.e("VisionCameraOCR", "Error saving bitmap to file", e)
        }
    }
        
}
