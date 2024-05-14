/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.reactnativemlkitodt

import com.facebook.react.bridge.*
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import java.util.LinkedList
import kotlin.math.max
import org.tensorflow.lite.task.vision.detector.Detection
import android.media.Image
import com.google.android.odml.image.MlImage

class TFObjectDetectorHelper(
  var threshold: Float = 0.5f,
  var numThreads: Int = 2,
  var maxResults: Int = 3,
  var currentDelegate: Int = 0,
  var currentModel: Int = 2,
  var modelName: String?,
  val context: ReactApplicationContext,
  
) {

    // For this example this needs to be a var so it can be reset on changes. If the ObjectDetector
    // will not change, a lazy val would be preferable.
    private var objectDetector: ObjectDetector? = null
    
    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        objectDetector = null
    }

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    fun setupObjectDetector() {
        // Create the base options for the detector using specifies max results and score threshold
        val optionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)

        // Set general detection options, including number of used threads
        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    //objectDetectorListener?.onError("GPU is not supported on this device")
                    Log.e("TFObjectDetectorHelper", "GPU is not supported on this device")
                }
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        var currentModelName =
            when (currentModel) {
                MODEL_EFFICIENTDETV0 -> "custom_models/efficientdet-lite0.tflite"
                MODEL_EFFICIENTDETV1 -> "custom_models/efficientdet-lite1.tflite"
                MODEL_EFFICIENTDETV2 -> "custom_models/efficientdet-lite2.tflite"
                MODEL_EFFICIENTDETV3 -> "custom_models/efficientdet-lite3.tflite"
                MODEL_EFFICIENTDETV4 -> "custom_models/efficientdet-lite4.tflite"
                else -> "mobilenetv1.tflite"
            }
        if(modelName != "")
            currentModelName = "custom_models/"+modelName

        try {
            objectDetector =
                ObjectDetector.createFromFileAndOptions(context, currentModelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
            Log.e("TFObjectDetectorHelper", "Object detector failed to initialize. See error logs for details:" + e.message)
            /*objectDetectorListener?.onError(
                "Object detector failed to initialize. See error logs for details"
            )*/
            //Log.e("Test", "TFLite failed to load model with error: " + e.message)
        }
    }

    fun detect(image: Bitmap) : List <Detection>? {
        if (objectDetector == null) {
            setupObjectDetector()
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
       /* val imageProcessor =
            ImageProcessor.Builder()
                .add(Rot90Op(-imageRotation / 90))
                .build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))*/
        var results = objectDetector?.detect(TensorImage.fromBitmap(image))
        
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
       return results
    }
   fun detectFrameProcessor(image: MlImage) : List <Detection>? {
        if (objectDetector == null) {
            setupObjectDetector()
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
       /* val imageProcessor =
            ImageProcessor.Builder()
                .add(Rot90Op(-imageRotation / 90))
                .build()

        // Preprocess the image and convert it into a TensorImage for detection.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))*/
        var results = objectDetector?.detect(image)
        
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
       return results
    }
  

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_EFFICIENTDETV0 = 0
        const val MODEL_EFFICIENTDETV1 = 1
        const val MODEL_EFFICIENTDETV2 = 2
        const val MODEL_EFFICIENTDETV3 = 3
        const val MODEL_EFFICIENTDETV4 = 4
    }
}