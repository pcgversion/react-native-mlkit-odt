package com.reactnativemlkitodt

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
import androidx.camera.core.ImageProxy
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks

import androidx.annotation.NonNull
import androidx.annotation.Nullable

import com.mrousavy.camera.frameprocessor.FrameProcessorPlugin
import com.facebook.react.bridge.ReactApplicationContext
import java.lang.ref.WeakReference
import com.facebook.react.bridge.ReadableNativeArray
import com.facebook.react.bridge.ReadableNativeMap
import com.facebook.react.bridge.WritableNativeArray
import com.facebook.react.bridge.WritableNativeMap
import android.annotation.SuppressLint
import java.io.ByteArrayOutputStream


class MlkitOdtFrameProcessorPlugin(reactContext: ReactApplicationContext): FrameProcessorPlugin("detectObjects") {
   
    private val _context:ReactApplicationContext = reactContext

    override fun callback(frame: ImageProxy, params: Array<Any>): Any? {
        
        @SuppressLint("UnsafeOptInUsageError")
        val mediaImage: Image? = frame.getImage()
        Log.d("OB Detector....","${frame.imageInfo.rotationDegrees}");
        try {

            if (params != null && mediaImage != null) {
                var objectDetectionOptions: ReadableNativeMap =  params[0] as ReadableNativeMap
                var isSingleImageMode = objectDetectionOptions.getInt("detectorMode")
                var enableClassification = objectDetectionOptions.getBoolean("shouldEnableClassification") 
                var enableMultiDetect = objectDetectionOptions.getBoolean("shouldEnableMultipleObjects")
                var customModel = objectDetectionOptions.getString("customModel")
                var modelName = objectDetectionOptions.getString("modelName")
                if(customModel == "automl"){
                    val image = InputImage.fromMediaImage(mediaImage, frame.imageInfo.rotationDegrees)
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
                    var objectDetector = TFObjectDetectorHelper(0.5f, 2, 3, 0, 2, modelName, _context)
                    //var mlImage = MediaMlImageBuilder(mediaImage).setRotation(0).build()
                    var mlImage = MediaMlImageBuilder(mediaImage).setRotation(frame.imageInfo.rotationDegrees).build()
                    var results = objectDetector?.detectFrameProcessor(mlImage)
                    return tfMakeResultObject(results)
                    
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
        }
        return null
    }

    private fun tfMakeResultObject(objects: List<Detection>?): WritableArray {
        val data: WritableArray = Arguments.createArray()
        for ((i, detectedObject) in objects!!.withIndex()) {
        val outputObject = Arguments.createMap()
        
        outputObject.putMap("bounding", tfGetBoundingResult(detectedObject!!.boundingBox))
        Log.i("detected object...","${detectedObject!!.categories[0]}")
            outputObject.putString("trackingID", detectedObject!!.categories[0].label+""+i)
            val labels = Arguments.createArray()
            val lbl = Arguments.createMap()
            lbl.putString("text", detectedObject!!.categories[0].label)
            lbl.putString("index","0")
            lbl.putString("confidence", String.format("%.2f", detectedObject!!.categories[0].score))
            labels.pushMap(lbl)
            outputObject.putArray("labels", labels)
        
        data.pushMap(outputObject)
        }
    return data
  }

  private fun makeResultObject(objects: List<DetectedObject>): WritableArray {
    val data: WritableArray = Arguments.createArray()
    var i = 0;
    for (detectedObject in objects) {
      val outputObject = Arguments.createMap()
      outputObject.putMap("bounding", getBoundingResult(detectedObject.boundingBox))
      if (detectedObject.trackingId != null)
        outputObject.putString("trackingID", detectedObject.trackingId?.toString()+""+i.toString())
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
      i++;
    }
    return data
  }

  private fun getBoundingResult(boundingBox: Rect): WritableMap {
    val coordinates: WritableMap = Arguments.createMap()
    coordinates.putInt("top", boundingBox.top)
    coordinates.putInt("left", boundingBox.left)
    coordinates.putInt("width", boundingBox.width())
    coordinates.putInt("height", boundingBox.height())
    return coordinates;
  }
  private fun tfGetBoundingResult(boundingBox: RectF): WritableMap {
    val coordinates: WritableMap = Arguments.createMap()
    coordinates.putInt("top", boundingBox.top.toInt())
    coordinates.putInt("left", boundingBox.left.toInt())
    coordinates.putInt("width", boundingBox.width().toInt())
    coordinates.putInt("height", boundingBox.height().toInt())
    return coordinates;
  }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val planes = imageProxy.planes
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val width = imageProxy.width
        val height = imageProxy.height
        val yuvImage = YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        var bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
         Log.d("OB DETECTOR.....ROTATION", "${rotationDegrees}")
        if (rotationDegrees != 0) {
            val matrix = Matrix()
            matrix.postRotate(rotationDegrees.toFloat())
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        }
        return bitmap
    }


    /**
     * rotateImage():
     *     Decodes and crops the captured image from camera.
     */
    private fun rotateImage(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height,
            matrix, true
        )
    }
        
}
