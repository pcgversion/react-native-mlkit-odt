import CoreImage
import TensorFlowLite
import UIKit
import Accelerate
import MLImage


/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
    let inferences: [Inference]
}

/// Stores one formatted inference.
struct Inference {
    let confidenceScore: Float
    let rect: CGRect
    let trackingID: String
    let labels: [Label]
    let bounding: Bounding
}
struct Label: Codable {
    let index: String
    let text: String
    let confidence: String
}
struct Bounding: Codable {
    let width: Int
    let top: Int
    let left: Int
    let height: Int
}
/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `TFObjectDetector`.
class TFObjectDetectorHelper: NSObject {
    
    // MARK: Private properties
    
    /// TensorFlow Lite `Interpreter` object for performing object detection using a given model.
    private var interpreter: Interpreter?
    let threshold: Float = 0.5
    
    // MARK: Model parameters
    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 640 //width that used to train tensorflow model
    let inputHeight = 640 //widht that used to train tensorflow model
    private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
    private let rgbPixelChannels = 3
    var tfModelName = ""
    
    
    // MARK: - Initialization
    
    /// A failable initializer for `TFObjectDetectorHelper`.
    ///
    /// - Parameter modelFileInfo: The TFLite model to be used.
    /// - Parameter:
    ///
    ///   - threadCount: Number of threads to be used.
    ///   - scoreThreshold: Minimum score of objects to be include in the detection result.
    ///   - maxResults: Maximum number of objects to be include in the detection result.
    /// - Returns: A new instance is created if the model is successfully loaded from the app's main
    /// bundle.
    init?(modelPath: String, modelName:String, threadCount: Int = 2, scoreThreshold: Float, maxResults: Int = 3) {
        
        // Construct the path to the model file.
        tfModelName = modelName
        
        // Specify the options for the `Detector`.
        //    let options = ObjectDetectorOptions(modelPath: modelPath)
        //    options.classificationOptions.scoreThreshold = scoreThreshold
        //    options.classificationOptions.maxResults = maxResults
        //    options.baseOptions.computeSettings.cpuSettings.numThreads = Int32(threadCount)
        
        do {
            // Create the `Detector`.
            interpreter = try Interpreter(modelPath: modelPath)
            var options = Interpreter.Options()
            options.threadCount = threadCount
            do {
                // Create the `Interpreter`.
                interpreter = try Interpreter(modelPath: modelPath, options: options)
                // Allocate memory for the model's input `Tensor`s.
                try interpreter?.allocateTensors()
            } catch let error {
                print("Failed to create the interpreter with error: \(error.localizedDescription)")
                return nil
            }
            
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        super.init()
    }
    
    /// Detect objects from the given frame.
    ///
    /// This method handles all data preprocessing and makes calls to run inference on a given frame
    /// through the `Detector`. It then formats the inferences obtained and returns results
    /// for a successful inference.
    ///
    /// - Parameter pixelBuffer: The target frame.
    /// - Returns: The detected objects and other metadata of the inference.
    func detect(pixelBuffer pixelBuffer: CVPixelBuffer) -> Any? {
        
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        
        let imageChannels = 4
        assert(imageChannels >= inputChannels)
        
        // Crops the image to the biggest square in the center and scales it down to model dimensions.
        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
            return nil
        }
        
        let outputBoundingBox: Tensor?
        let outputClasses: Tensor?
        let outputScores: Tensor?
        let outputCount: Tensor?
        do {
            let inputTensor = try interpreter?.input(at: 0)
            
            // Remove the alpha component from the image buffer to get the RGB data.
            guard let rgbData = rgbDataFromBuffer(
                scaledPixelBuffer,
                byteCount: batchSize * inputWidth * inputHeight * inputChannels,
                isModelQuantized: inputTensor?.dataType == .uInt8
            ) else {
                print("Failed to convert the image buffer to RGB data.")
                return nil
            }
            
            // Copy the RGB data to the input `Tensor`.
            try interpreter?.copy(rgbData, toInputAt: 0)
            
            // Run inference by invoking the `Interpreter`.
            try interpreter?.invoke()
            
            outputBoundingBox = try interpreter?.output(at: 1)
            outputClasses = try interpreter?.output(at: 3)
            outputScores = try interpreter?.output(at:0)
            outputCount = try interpreter?.output(at: 2)
        } catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        // Formats the results
        let resultArray = formatResults(
            boundingBox: [Float](unsafeData: outputBoundingBox!.data) ?? [],
            outputClasses: [Float](unsafeData: outputClasses!.data) ?? [],
            outputScores: [Float](unsafeData: outputScores!.data) ?? [],
            outputCount: Int(([Float](unsafeData: outputCount!.data) ?? [0])[0]),
            width: CGFloat(imageWidth),
            height: CGFloat(imageHeight)
        )
        
        
        return resultArray
        
    }
    
    func formatResults(boundingBox: [Float], outputClasses: [Float], outputScores: [Float], outputCount: Int, width: CGFloat, height: CGFloat) -> [[String: Any]]{
//        print("Bounding box: ", boundingBox)
//        print("categories: ",outputClasses)
//        print("scores: ", outputScores)
//        print("total boxes: ", outputCount)
        var resultsArray: [Inference] = []
        var output: [[String: Any]] = []
        if (outputCount == 0) {
            return output
        }
        for i in 0...outputCount - 1 {
            
            let score = outputScores[i]
            // Filters results with confidence < threshold.
            guard score >= threshold else {
                continue
            }
            
            // Gets the output class names for detected classes from labels list.
            let outputClassIndex = Int(outputClasses[i])
            //let outputClass = labels[outputClassIndex + 1]
            
            var rect: CGRect = CGRect.zero
            
            // Translates the detected bounding box to CGRect.
            rect.origin.y = CGFloat(boundingBox[4*i])
            rect.origin.x = CGFloat(boundingBox[4*i+1])
            rect.size.height = CGFloat(boundingBox[4*i+2]) - rect.origin.y
            rect.size.width = CGFloat(boundingBox[4*i+3]) - rect.origin.x
            
            // The detected corners are for model dimensions. So we scale the rect with respect to the
            // actual image dimensions.
            let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))
            // Gets the color assigned for the class
            // let colorToAssign = UIColor.colorForClass(withIndex: outputClassIndex + 1)
            let bounding = Bounding(width:Int(newRect.size.width), top: Int(newRect.origin.y), left:Int(newRect.origin.x), height: Int(newRect.size.height))
            var labelText = "BoxLabel"
             if(tfModelName == "Box-Label-Detector"){
                labelText = (outputClasses[i] == 1) ? "Label" : "Box"
            }
            let trackingId: String = labelText + String(i)
            var labelsArray: [Label] = [];
            let labels = labelsArray.append(Label(index: "0", text: labelText, confidence: String(format: "%.2f", score)))
            let inference = Inference(confidenceScore: score,
                                      rect: newRect,
                                      trackingID: trackingId,
                                      labels: labelsArray,
                                      bounding: bounding
            )
            resultsArray.append(inference)
        }
        
        // Sort results in descending order of confidence.
        resultsArray.sort { (first, second) -> Bool in
            return first.confidenceScore > second.confidenceScore
        }
        for object in resultsArray {
            var detectedObject: [String: Any] = [:]
            detectedObject["bounding"] = ["left":object.bounding.left, "top":object.bounding.top, "width": object.bounding.width, "height":object.bounding.height]
            detectedObject["trackingID"] = object.trackingID
            var labels: [[String: String]] = []
            for label in object.labels {
                var resultLabel: [String: String] = [:]
                resultLabel["text"] = label.text
                resultLabel["confidence"] = String(label.confidence)
                resultLabel["index"] = String(label.index)
                labels.append(resultLabel)
            }
            detectedObject["labels"] = labels
            output.append(detectedObject)
        }
        return output
    }
    
    /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
    ///
    /// - Parameters
    ///   - buffer: The BGRA pixel buffer to convert to RGB data.
    ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
    ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
    ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
    ///       floating point values).
    /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
    ///     converted.
    private func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                         height: vImagePixelCount(height),
                                         width: vImagePixelCount(width),
                                         rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        
        defer {
            free(destinationData)
        }
        
        var destinationBuffer = vImage_Buffer(data: destinationData,
                                              height: vImagePixelCount(height),
                                              width: vImagePixelCount(width),
                                              rowBytes: destinationBytesPerRow)
        
        if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
            vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        }
        
        let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        if isModelQuantized {
            return byteData
        }
        
        // Not quantized, convert to floats
        let bytes = Array<UInt8>(unsafeData: byteData)!
        var floats = [Float]()
        for i in 0..<bytes.count {
            floats.append(Float(bytes[i]) / 255.0)
        }
        return Data(copyingBufferOf: floats)
    }
    
}




extension CVPixelBuffer {
    /// Returns thumbnail by cropping pixel buffer to biggest square and scaling the cropped image
    /// to model dimensions.
    func resized(to size: CGSize ) -> CVPixelBuffer? {
        
        let imageWidth = CVPixelBufferGetWidth(self)
        let imageHeight = CVPixelBufferGetHeight(self)
        
        let pixelBufferType = CVPixelBufferGetPixelFormatType(self)
        //print("CVPixelBuffer Function.......:", pixelBufferType, kCVPixelFormatType_32BGRA)
        //assert(pixelBufferType == kCVPixelFormatType_32BGRA)
        
        let inputImageRowBytes = CVPixelBufferGetBytesPerRow(self)
        let imageChannels = 4
        
        CVPixelBufferLockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))
        
        // Finds the biggest square in the pixel buffer and advances rows based on it.
        guard let inputBaseAddress = CVPixelBufferGetBaseAddress(self) else {
            return nil
        }
        
        // Gets vImage Buffer from input image
        var inputVImageBuffer = vImage_Buffer(data: inputBaseAddress, height: UInt(imageHeight), width: UInt(imageWidth), rowBytes: inputImageRowBytes)
        
        let scaledImageRowBytes = Int(size.width) * imageChannels
        guard  let scaledImageBytes = malloc(Int(size.height) * scaledImageRowBytes) else {
            return nil
        }
        
        // Allocates a vImage buffer for scaled image.
        var scaledVImageBuffer = vImage_Buffer(data: scaledImageBytes, height: UInt(size.height), width: UInt(size.width), rowBytes: scaledImageRowBytes)
        
        // Performs the scale operation on input image buffer and stores it in scaled image buffer.
        let scaleError = vImageScale_ARGB8888(&inputVImageBuffer, &scaledVImageBuffer, nil, vImage_Flags(0))
        
        CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))
        
        guard scaleError == kvImageNoError else {
            return nil
        }
        
        let releaseCallBack: CVPixelBufferReleaseBytesCallback = {mutablePointer, pointer in
            
            if let pointer = pointer {
                free(UnsafeMutableRawPointer(mutating: pointer))
            }
        }
        
        var scaledPixelBuffer: CVPixelBuffer?
        
        // Converts the scaled vImage buffer to CVPixelBuffer
        let conversionStatus = CVPixelBufferCreateWithBytes(nil, Int(size.width), Int(size.height), pixelBufferType, scaledImageBytes, scaledImageRowBytes, releaseCallBack, nil, nil, &scaledPixelBuffer)
        
        guard conversionStatus == kCVReturnSuccess else {
            
            free(scaledImageBytes)
            return nil
        }
        
        return scaledPixelBuffer
    }
    
}


// MARK: - Extensions

extension Data {
    /// Creates a new buffer by copying the buffer pointer of the given array.
    ///
    /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
    ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
    ///     data from the resulting buffer has undefined behavior.
    /// - Parameter array: An array with elements of type `T`.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}

extension Array {
    /// Creates a new array from the bytes of the given unsafe data.
    ///
    /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
    ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
    ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
    /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
    ///     `MemoryLayout<Element>.stride`.
    /// - Parameter unsafeData: The data containing the bytes to turn into an array.
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
#if swift(>=5.0)
        self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
#else
        self = unsafeData.withUnsafeBytes {
            .init(UnsafeBufferPointer<Element>(
                start: $0,
                count: unsafeData.count / MemoryLayout<Element>.stride
            ))
        }
#endif  // swift(>=5.0)
    }
}
