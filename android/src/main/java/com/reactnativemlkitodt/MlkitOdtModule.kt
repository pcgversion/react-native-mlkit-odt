package com.reactnativemlkitodt

import android.graphics.*
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.graphics.RectF
import android.net.Uri
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import com.facebook.react.bridge.*
import com.google.firebase.ml.modeldownloader.*
import com.google.firebase.ml.modeldownloader.CustomModel
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader
import com.google.mlkit.common.model.LocalModel
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.*
import com.google.mlkit.vision.objects.custom.CustomObjectDetectorOptions
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import java.io.File
import java.io.IOException
import java.lang.Exception
import java.nio.file.*
import java.nio.file.Files
import java.util.concurrent.TimeUnit
import org.tensorflow.lite.task.vision.detector.Detection

class MlkitOdtModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val _context: ReactApplicationContext = reactContext

    override fun getName(): String {
        return "MlkitOdt"
    }

    @ReactMethod
    fun detectFromUri(
        uri: String,
        isSingleImageMode: Int,
        enableClassification: Int,
        enableMultiDetect: Int,
        modelName: String,
        customModel: String,
        promise: Promise
    ) {
        val image: InputImage
        try {

            if (customModel == "automl") {

                var currentModelName = _context.filesDir.getPath() + "/custom_models/" + modelName

                var localModel =
                    LocalModel.Builder()
                        // .setAssetFilePath("custom_models/object_labeler.tflite")
                        .setAbsoluteFilePath(currentModelName + ".tflite")
                        // or .setUri(URI to model file)
                        .build()
                var customOptionsBuilder = CustomObjectDetectorOptions.Builder(localModel)
                if (isSingleImageMode == 1) {
                    customOptionsBuilder.setDetectorMode(
                        CustomObjectDetectorOptions.SINGLE_IMAGE_MODE
                    )
                }
                if (enableClassification == 1) {
                    customOptionsBuilder.enableClassification()
                }
                if (enableMultiDetect == 1) {
                    customOptionsBuilder.enableMultipleObjects()
                }
                image = InputImage.fromFilePath(reactApplicationContext, Uri.parse(uri))
                val customObjectDetector = ObjectDetection.getClient(customOptionsBuilder.build())
                customObjectDetector
                    .process(image)
                    .addOnSuccessListener { detectedObjects ->
                        promise.resolve(makeResultObject(detectedObjects))
                    }
                    .addOnFailureListener { e ->
                        promise.reject(e)
                        e.printStackTrace()
                    }
            } else if (customModel == "tensorflow") {

                var objectDetector =
                    TFObjectDetectorHelper(0.5f, 2, 5, 0, 2, modelName, reactApplicationContext)
                image = InputImage.fromFilePath(reactApplicationContext, Uri.parse(uri))
                var bitmapImage = BitmapFactory.decodeFile(Uri.parse(uri).getPath())

                val exifInterface = ExifInterface(Uri.parse(uri).getPath().toString())
                val orientation =
                    exifInterface.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_UNDEFINED
                    )
                // Log.d("ORIENTATION....","${orientation}")

                /*if(orientation == 6) {
                    bitmapImage =   rotateImage(bitmapImage, 90f)
                }
                if(orientation == 3) {
                    bitmapImage =   rotateImage(bitmapImage, 180f)
                }*/
                var results = objectDetector?.detect(bitmapImage)
                promise.resolve(tfMakeResultObject(results))
            } else {
                var optionsBuilder = ObjectDetectorOptions.Builder()
                if (isSingleImageMode == 1) {
                    optionsBuilder.setDetectorMode(ObjectDetectorOptions.SINGLE_IMAGE_MODE)
                }
                if (enableClassification == 1) {
                    optionsBuilder.enableClassification()
                }
                if (enableMultiDetect == 1) {
                    optionsBuilder.enableMultipleObjects()
                }
                image = InputImage.fromFilePath(reactApplicationContext, Uri.parse(uri))
                val objectDetector = ObjectDetection.getClient(optionsBuilder.build())
                objectDetector
                    .process(image)
                    .addOnSuccessListener { detectedObjects ->
                        promise.resolve(makeResultObject(detectedObjects))
                    }
                    .addOnFailureListener { e ->
                        promise.reject(e)
                        e.printStackTrace()
                    }
            }
        } catch (e: Exception) {
            promise.reject(e)
            e.printStackTrace()
        }
    }

    @ReactMethod
    fun downloadCustomModel(modelName: String, promise: Promise) {
        try {
            Log.i("DOWNLOAD CUSTOM MODEL....", modelName)

            val conditions =
                CustomModelDownloadConditions.Builder()
                    .requireWifi() // Adjust this as needed
                    .build()

            val modelDownloader = FirebaseModelDownloader.getInstance()
            modelDownloader
                .getModel(modelName, DownloadType.LATEST_MODEL, conditions)
                .addOnSuccessListener { model: CustomModel? ->
                    // Download complete. Depending on your app, you could enable the ML
                    // feature, or switch from the local model to the remote model, etc.

                    // The CustomModel object contains the local path of the model file,
                    // which you can use to instantiate a TensorFlow Lite interpreter.
                    val modelFile = model?.file
                    if (modelFile != null) {
                        Log.d("ModelDownload", "Model downloaded to: ${modelFile.absolutePath}")
                        val newDirectory = File(reactApplicationContext.filesDir, "custom_models")
                        if (!newDirectory.exists()) {
                            Log.i("MODEL DOWNLOAD", "Creating Directory 'custom_models'")
                            newDirectory.mkdirs()
                        }
                    } else {
                        Log.d("ModelDownload", "Model file not available locally")
                        promise.resolve(makeResultObjectCustomModelDownload(false))
                    }
                }
                .addOnFailureListener { e: Exception ->
                    Log.e("ModelDownload", "Failed to download model", e)
                    promise.resolve(makeResultObjectCustomModelDownload(false))
                }
                .addOnCompleteListener { task ->
                    if (task.isSuccessful) {
                        val downloadedCustomModel = task.result
                        downloadedCustomModel?.let {
                            val modelFile = it.getFile()
                            if (modelFile != null) {

                                // This will be called irrespective of success or failure

                                val newDirectory =
                                    File(reactApplicationContext.filesDir, "custom_models")

                                var destinationPath =
                                    newDirectory.getPath() + "/" + modelName + ".tflite"
                                val sourceFile: File = modelFile
                                val destinationFile: File = File(destinationPath)

                                try {
                                    val timeDifference =
                                        destinationFile.lastModified() - sourceFile.lastModified()
                                    val timeDifferenceInMinutes =
                                        TimeUnit.MILLISECONDS.toMinutes(timeDifference)

                                    Log.i(
                                        "MODELDOWNLOAD",
                                        "Time difference:" + timeDifferenceInMinutes.toString()
                                    )
                                    // Get the current time in milliseconds
                                    val currentTime = System.currentTimeMillis()
                                    Log.i(
                                        "MODELDOWNLOAD",
                                        "Current Time difference:" +
                                            TimeUnit.MILLISECONDS.toMinutes(
                                                    currentTime - sourceFile.lastModified()
                                                )
                                                .toString()
                                    )
                                    Log.d(
                                        "MODELDONWLOAD FILE EXISTS",
                                        "${(!destinationFile.exists() || timeDifference <= 0)}"
                                    )
                                    // Copy the downloaded model to the destination file
                                    if (!destinationFile.exists() || timeDifference <= 0) {
                                        Files.copy(
                                            sourceFile.toPath(),
                                            destinationFile.toPath(),
                                            StandardCopyOption.REPLACE_EXISTING
                                        )
                                        // Or you can move the downloaded model to the destination
                                        // file
                                        // Files.move(sourceFile.toPath(), destinationFile.toPath(),
                                        // StandardCopyOption.REPLACE_EXISTING)

                                        // Now the model is saved at the destination path for later
                                        // usage
                                        Log.d(
                                            "ModelDownload",
                                            "Model File copied from temporary directory to custom_models folder"
                                        )
                                    }
                                    promise.resolve(makeResultObjectCustomModelDownload(true))
                                    Log.d("ModelDownload", "Model downloaded")
                                } catch (e: IOException) {
                                    // Handle the exception
                                    Log.e(
                                        "ModelDownload - Copy File",
                                        "Copying downloaded model failed",
                                        e
                                    )
                                    promise.resolve(makeResultObjectCustomModelDownload(false))
                                }
                            }
                        }
                    }
                    Log.i("ModelDownload", "End of addOnCompleteListener....")
                }
        } catch (e: Exception) {
            promise.reject(e)
            e.printStackTrace()
        }
    }

    public fun isFileModifiedWithinLastHour(file: File): Boolean {
        // Get the file's last modified time in milliseconds
        val lastModifiedTime = file.lastModified()

        // Get the current time in milliseconds
        val currentTime = System.currentTimeMillis()

        // Calculate the difference in time
        val timeDifference = currentTime - lastModifiedTime

        // Convert the time difference to minutes
        val timeDifferenceInMinutes = TimeUnit.MILLISECONDS.toMinutes(timeDifference)

        // Check if the difference is within an hour (60 minutes)
        return timeDifferenceInMinutes <= 60
    }

    private fun tfMakeResultObject(objects: List<Detection>?): WritableArray {
        val data: WritableArray = Arguments.createArray()
        for ((i, detectedObject) in objects!!.withIndex()) {
            val outputObject = Arguments.createMap()

            outputObject.putMap("bounding", tfGetBoundingResult(detectedObject!!.boundingBox))
            Log.i("detected object...", "${detectedObject!!.categories[0]}")
            outputObject.putString(
                "trackingID",
                detectedObject!!.categories[0].label + i.toString()
            )
            val labels = Arguments.createArray()
            val lbl = Arguments.createMap()
            lbl.putString("text", detectedObject!!.categories[0].label)
            lbl.putString("index", "0")
            lbl.putString("confidence", String.format("%.2f", detectedObject!!.categories[0].score))
            labels.pushMap(lbl)
            outputObject.putArray("labels", labels)

            data.pushMap(outputObject)
        }
        return data
    }

    private fun makeResultObjectCustomModelDownload(success: Boolean): WritableArray {
        val data: WritableArray = Arguments.createArray()
        val outputObject = Arguments.createMap()
        outputObject.putBoolean("result", success)
        data.pushMap(outputObject)
        return data
    }

    private fun makeResultObject(objects: List<DetectedObject>): WritableArray {
        val data: WritableArray = Arguments.createArray()
        for (detectedObject in objects) {
            val outputObject = Arguments.createMap()
            outputObject.putMap("bounding", getBoundingResult(detectedObject.boundingBox))
            if (detectedObject.trackingId != null)
                outputObject.putString("trackingID", detectedObject.trackingId?.toString())
            val labels = Arguments.createArray()
            detectedObject.labels.forEach { l ->
                val lbl = Arguments.createMap()
                lbl.putString("text", l.text)
                lbl.putString("index", l.index?.toString())
                lbl.putString("confidence", l.confidence?.toString())
                labels.pushMap(lbl)
            }
            outputObject.putArray("labels", labels)
            data.pushMap(outputObject)
        }
        return data
    }

    private fun getBoundingResult(boundingBox: Rect): WritableMap {
        val coordinates: WritableMap = Arguments.createMap()
        coordinates.putInt("top", boundingBox.top)
        coordinates.putInt("left", boundingBox.left)
        coordinates.putInt("width", boundingBox.width())
        coordinates.putInt("height", boundingBox.height())
        return coordinates
    }

    private fun tfGetBoundingResult(boundingBox: RectF): WritableMap {
        val coordinates: WritableMap = Arguments.createMap()
        coordinates.putInt("top", boundingBox.top.toInt())
        coordinates.putInt("left", boundingBox.left.toInt())
        coordinates.putInt("width", boundingBox.width().toInt())
        coordinates.putInt("height", boundingBox.height().toInt())
        return coordinates
    }
    /** rotateImage(): Decodes and crops the captured image from camera. */
    private fun rotateImage(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }
}
