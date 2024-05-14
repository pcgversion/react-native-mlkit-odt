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
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.nio.ByteBuffer
import android.util.Log
import androidx.exifinterface.media.ExifInterface

class MlkitOdtModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

  override fun getName(): String {
    return "MlkitOdt"
  }

  @ReactMethod
  fun detectFromUri(uri: String, isSingleImageMode: Int, enableClassification: Int, enableMultiDetect: Int, modelName:String, customModel: String, promise: Promise) {
    val image: InputImage;
    try {

      

      if(customModel == "automl"){

        var localModel = LocalModel.Builder()
              .setAssetFilePath("custom_models/object_labeler.tflite")
              // or .setAbsoluteFilePath(absolute file path to model file)
              // or .setUri(URI to model file)
              .build()
        var customOptionsBuilder = CustomObjectDetectorOptions.Builder(localModel)
        if (isSingleImageMode == 1) {
          customOptionsBuilder.setDetectorMode(CustomObjectDetectorOptions.SINGLE_IMAGE_MODE)
        }
        if (enableClassification == 1) {
          customOptionsBuilder.enableClassification()
        }
        if (enableMultiDetect == 1) {
          customOptionsBuilder.enableMultipleObjects()
        }
        image = InputImage.fromFilePath(reactApplicationContext, Uri.parse(uri));
        val customObjectDetector = ObjectDetection.getClient(customOptionsBuilder.build())
        customObjectDetector.process(image).addOnSuccessListener { detectedObjects ->
          promise.resolve(makeResultObject(detectedObjects))
        }.addOnFailureListener { e ->
          promise.reject(e)
          e.printStackTrace()
        }
        
       
      }else if(customModel == "tensorflow") {
        
        var objectDetector = TFObjectDetectorHelper(0.5f, 2, 5, 0, 2, modelName, reactApplicationContext)
        image = InputImage.fromFilePath(reactApplicationContext, Uri.parse(uri));
        var bitmapImage = BitmapFactory.decodeFile(Uri.parse(uri).getPath())
        
        val exifInterface = ExifInterface(Uri.parse(uri).getPath().toString())
        val orientation = exifInterface.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_UNDEFINED
        )
        //Log.d("ORIENTATION....","${orientation}")
                
        /*if(orientation == 6) {
            bitmapImage =   rotateImage(bitmapImage, 90f)
        }
        if(orientation == 3) {
            bitmapImage =   rotateImage(bitmapImage, 180f)
        }*/
        var results = objectDetector?.detect(bitmapImage)
        promise.resolve(tfMakeResultObject(results))
        
      }else{
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
        image = InputImage.fromFilePath(reactApplicationContext, Uri.parse(uri));
        val objectDetector = ObjectDetection.getClient(optionsBuilder.build())
        objectDetector.process(image).addOnSuccessListener { detectedObjects ->
          promise.resolve(makeResultObject(detectedObjects))
        }.addOnFailureListener { e ->
          promise.reject(e)
          e.printStackTrace()
        }
      }
      
      
    } catch (e: Exception) {
      promise.reject(e)
      e.printStackTrace()
    }
  }
  private fun tfMakeResultObject(objects: List<Detection>?): WritableArray {
    val data: WritableArray = Arguments.createArray()
    for ((i, detectedObject) in objects!!.withIndex()) {
      val outputObject = Arguments.createMap()
      
      outputObject.putMap("bounding", tfGetBoundingResult(detectedObject!!.boundingBox))
      Log.i("detected object...","${detectedObject!!.categories[0]}")
        outputObject.putString("trackingID", detectedObject!!.categories[0].label+i.toString())
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
