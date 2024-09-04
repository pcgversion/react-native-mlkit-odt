//
//  MlkitOdt.swift
//  MlkitOdt
//
//  Created by Ritesh Jariwala on 14/06/24.
//  Copyright © 2024 Facebook. All rights reserved.
//

import Foundation
import FirebaseMLModelDownloader
import MLKitObjectDetectionCustom
import MLKitObjectDetection
import MLKitVision
import MLKitCommon
import MLImage
import ImageIO
import MobileCoreServices

import CoreImage
import CoreVideo
import UIKit
import Accelerate

import Vision
import AVFoundation

@objc(MlkitOdtFrameProcessorPlugin)
public class MlkitOdtFrameProcessorPlugin: NSObject, FrameProcessorPluginBase {
    
    @objc
    public static func callback(_ frame: Frame!, withArgs args: [Any]!) -> Any! {
        
        // guard (CMSampleBufferGetImageBuffer(frame.buffer) != nil) else {
        //   print("Failed to get image buffer from sample buffer.")
        //   return nil
        // }
        // print("testing comes here...buffer data...2...",frame.buffer)
        // let visionImage = VisionImage(buffer: frame.buffer)
        var modelName: String = ""; 
        var customModel: String = "";
        var singleImage: Int? = 0;
        var classification: Bool = false;
        var multiDetect: Bool = false;
        if let options = args[0] as? [String: Any] {
            print("options....\(options["customModel"])")
            modelName = options["modelName"] as? String ?? ""
            customModel = options["customModel"] as? String ?? ""
            singleImage = options["singleImage"] as? Int ?? 0
            classification = options["classification"] as? Bool ?? false
            multiDetect = options["multiDetect"] as? Bool ?? false
        }
        let fileManager = FileManager.default
        
        // Get the document directory path
        var absoluteModelPath: String = "";
        if let documentDirectory = getDocumentsDirectory(){
            //NSLog("@document directory path...%@", documentDirectory.path);
            absoluteModelPath = documentDirectory.path + "/custom_models/" + modelName + ".tflite";
        }
        guard let imageBuffer = CMSampleBufferGetImageBuffer(frame.buffer) else {
          print("Failed to get image buffer from sample buffer.")
          return nil
        }

        var ciImage = CIImage(cvPixelBuffer: imageBuffer)
        var curDeviceOrientation = UIDevice.current.orientation
        let isLandscape = isDeviceInLandscapeWhenFaceUp()
        //print("current Device Orientation: \(curDeviceOrientation) \(isLandscape)")
        switch curDeviceOrientation {
            case UIDeviceOrientation.portraitUpsideDown:  // Device oriented vertically, Home button on the top
                ciImage = ciImage.oriented(forExifOrientation: 3)
            case UIDeviceOrientation.landscapeLeft:       // Device oriented horizontally, Home button on the right
                ciImage = ciImage.oriented(forExifOrientation: 3)
            case UIDeviceOrientation.landscapeRight:      // Device oriented horizontally, Home button on the left
                ciImage = ciImage.oriented(forExifOrientation: 3)
            case UIDeviceOrientation.portrait:            // Device oriented vertically, Home button on the bottom
                ciImage = ciImage.oriented(forExifOrientation: 1)
            case UIDeviceOrientation.faceUp:
            ciImage = ciImage.oriented(forExifOrientation: isLandscape ? 3 : 1)
            case UIDeviceOrientation.faceDown:
                ciImage = ciImage.oriented(forExifOrientation: isLandscape ? 3 : 1)
            case UIDeviceOrientation.unknown:
                ciImage = ciImage.oriented(forExifOrientation: 1)
            default:
                ciImage = ciImage.oriented(forExifOrientation: 1)
        }
        guard let cgImage = CIContext().createCGImage(ciImage, from: ciImage.extent) else {
            print("Failed to create bitmap from image.")
            return nil
        }
        
        let image = UIImage(cgImage: cgImage)
       
//         if let pngData = image.pngData() {
//             // Define file path where you want to save the image
//             let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
//             let fileURL = documentsDirectory.appendingPathComponent("image-ob-detection.png")
//            
//             do {
//                 // Write PNG data to file
//                 try pngData.write(to: fileURL)
//                 print("Image saved successfully at \(fileURL)")
//             } catch {
//                 print("Error saving image: \(error)")
//             }
//         } else {
//             print("Failed to convert UIImage to PNG data")
//         }
        //let visionImage = VisionImage(image: image)
        print("FrameData detect objects: \(image.size.width)x\(image.size.height)")
        print("args: \(args[0]) \(modelName) \(customModel) \(absoluteModelPath)")

         if customModel == "tensorflow" {
            let tfObjectDetectorHelper = TFObjectDetectorHelper(modelPath: absoluteModelPath, modelName: modelName, scoreThreshold: 0.5, maxResults: 3)
            var inputData: Data
            guard let visionImage = uiImageToCVPixelBuffer(image: image) else {return []}
            // Preprocess the image and prepare input tensor
           
            do {
                let outputData = try tfObjectDetectorHelper?.detect(pixelBuffer: visionImage)
                return outputData
            } catch let error as NSError{
                let errorString = error.localizedDescription
                let pData: [String: Any] = ["error": "On-Device object detection failed with error: \(errorString)"]
                return pData;
            }
        }else{
            let obtOptions = customModel == "automl"  ?  CustomObjectDetectorOptions(localModel:  LocalModel(path: absoluteModelPath)) : ObjectDetectorOptions()
            
            obtOptions.detectorMode = singleImage == 1 ? .singleImage : .stream
            obtOptions.shouldEnableClassification = classification
            obtOptions.shouldEnableMultipleObjects = multiDetect
            
            let detector = ObjectDetector.objectDetector(options: obtOptions)
            let visionImage = VisionImage(image: image)
            visionImage.orientation = image.imageOrientation
            
            do {
                let result = try detector.results(in: visionImage)
                let output = makeOutputResult(objects: result)
                return output
                 
            } catch let error {
                print("Failed to detect objects in image with error: \(error.localizedDescription).")
                return nil
            }
            // detector.process(visionImage) { result, error in
            //     do {
            //         if let error = error {
            //             throw NSError(domain: "ObjectDetectionError", code: 0, userInfo: [NSLocalizedDescriptionKey: error.localizedDescription])
            //         }
            //         guard let result = result else {
            //             throw NSError(domain: "ObjectDetectionError", code: 0, userInfo: [NSLocalizedDescriptionKey: "No results found"])
            //         }
                    
            //         let output :[[String: Any]] = makeOutputResult(objects: result)
            //         return output
            //     } catch let error as NSError {
            //         let errorString = error.localizedDescription
            //         let pData: [String: Any] = ["error": "On-Device object detection failed with error: \(errorString)"]
            //         return pData
            //     }
            // }
        }
        return []
    }
}

func getDocumentsDirectory() -> URL? {
    let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    return paths.first
}
    
 func uiImageToCVPixelBuffer(image: UIImage) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else {
            print("Failed to get cgImage from UIImage")
            return nil
        }
        
        let frameSize = CGSize(width: cgImage.width, height: cgImage.height)
        
        var pixelBuffer: CVPixelBuffer?
        let options: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ]
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(frameSize.width),
                                         Int(frameSize.height),
                                         kCVPixelFormatType_32ARGB,
                                         options as CFDictionary,
                                         &pixelBuffer)
        
        guard status == kCVReturnSuccess, let createdPixelBuffer = pixelBuffer else {
            print("Failed to create pixel buffer")
            return nil
        }
        
        CVPixelBufferLockBaseAddress(createdPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(createdPixelBuffer)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData,
                                width: Int(frameSize.width),
                                height: Int(frameSize.height),
                                bitsPerComponent: 8,
                                bytesPerRow: CVPixelBufferGetBytesPerRow(createdPixelBuffer),
                                space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        guard let context = context else {
            print("Failed to create CGContext")
            CVPixelBufferUnlockBaseAddress(createdPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
            return nil
        }
        
        context.draw(cgImage, in: CGRect(origin: .zero, size: frameSize))
        CVPixelBufferUnlockBaseAddress(createdPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return createdPixelBuffer
    }

    func makeOutputResult(objects: [Object]?) -> [[String: Any]] {
        var output: [[String: Any]] = []
        guard let objects = objects else {
            return output
        }
        for object in objects {
            var detectedObject: [String: Any] = [:]
            detectedObject["bounding"] = makeBoundingResult(frame: object.frame)

            if let trackingID = object.trackingID {
                detectedObject["trackingID"] = trackingID
            }

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

    func makeBoundingResult(frame: CGRect) -> [String: CGFloat] {
        return [
            "originX": frame.origin.x,
            "originY": frame.origin.y,
            "width": frame.size.width,
            "height": frame.size.height
        ]
    }
    
    func isDeviceInLandscapeWhenFaceUp() -> Bool {
        let orientation = UIDevice.current.orientation
        // If the device is face up, check the interface orientation
        if orientation == .faceUp {
            // Get the current interface orientation
            let interfaceOrientation = UIApplication.shared.windows.first?.windowScene?.interfaceOrientation
            if let interfaceOrientation = interfaceOrientation {
                return interfaceOrientation.isLandscape
            }
        }
        // Otherwise, check if the current device orientation is landscape
        return orientation == .landscapeLeft || orientation == .landscapeRight
    }
@objc(MlkitOdt)
class MlkitOdt: NSObject {
    @objc
    func downloadCustomModel(_ modelName: String, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
        let modelDownloader = ModelDownloader.modelDownloader()
        let conditions = FirebaseMLModelDownloader.ModelDownloadConditions()
        modelDownloader.getModel(name: modelName, downloadType:ModelDownloadType.latestModel, conditions: conditions) { result in
            switch (result) {
            case .success(let customModel):
                NSLog("@custom model donwloaded:, %@", customModel.path)
                let resultData:[String:Any] = ["success":true]
                
                // Example usage
                if let customModelsDirectory = self.createCustomModelsDirectory(){
                    let sourceDirectory = customModel.path;
                    let customModelsDirectoryWithFilename = customModelsDirectory + modelName  + ".tflite"
                    self.copyFile(from: sourceDirectory, to: customModelsDirectoryWithFilename)
                }
                resolve(resultData)
            case .failure(let error):
                reject("model download failed","Failed to download model", error)
                let errorData:[String:Any] = ["success":false]
                resolve(errorData)
            }
        }
    }
    @objc
    func detectFromUri(_ imagePath: String?,
                       singleImage singleImage: NSNumber,
                       classification classification: NSNumber,
                       multiDetect multiDetect: NSNumber,
                       modelName modelName: String,
                       customModel customModel: String,
                       resolver resolve: @escaping RCTPromiseResolveBlock,
                       rejecter reject: @escaping RCTPromiseRejectBlock) {

        guard let imagePath = imagePath, let imageURL = URL(string: imagePath) else {
            NSLog("@No image uri provided");
            reject("wrong_arguments", "No image uri provided", nil)
            return
        }

        guard let imageData = try? Data(contentsOf: imageURL), let image = UIImage(data: imageData) else {
            DispatchQueue.main.async {
                NSLog("@No image found %@",imagePath)
                reject("no_image", "No image path provided", nil)
            }
            return
        }
       
        let fileManager = FileManager.default
        
        // Get the document directory path
        var absoluteModelPath: String = "";
        if let documentDirectory = self.getDocumentsDirectory(){
            //NSLog("@document directory path...%@", documentDirectory.path);
            absoluteModelPath = documentDirectory.path + "/custom_models/" + modelName + ".tflite";
        }
        if customModel == "tensorflow" {
            let tfObjectDetectorHelper = TFObjectDetectorHelper(modelPath: absoluteModelPath, modelName: modelName, scoreThreshold: 0.5, maxResults: 3)
            var inputData: Data
            guard let visionImage = self.uiImageToCVPixelBuffer(image: image) else {return}
            // Preprocess the image and prepare input tensor
           
            do {
             
                let outputData = try tfObjectDetectorHelper?.detect(pixelBuffer: visionImage)
                DispatchQueue.main.async {
                    resolve(outputData)
                }
            } catch let error as NSError{
                let errorString = error.localizedDescription
                let pData: [String: Any] = ["error": "On-Device object detection failed with error: \(errorString)"]
                DispatchQueue.main.async {
                    resolve(pData)
                }
            }
        }else{
            let options = customModel == "automl"  ?  CustomObjectDetectorOptions(localModel:  LocalModel(path: absoluteModelPath)) : ObjectDetectorOptions()
            
            options.detectorMode = singleImage.boolValue ? .singleImage : .stream
            options.shouldEnableClassification = classification.boolValue
            options.shouldEnableMultipleObjects = multiDetect.boolValue
            
            let detector = ObjectDetector.objectDetector(options: options)
            let visionImage = VisionImage(image: image)
            visionImage.orientation = image.imageOrientation
            
            detector.process(visionImage) { result, error in
                do {
                    if let error = error {
                        throw NSError(domain: "ObjectDetectionError", code: 0, userInfo: [NSLocalizedDescriptionKey: error.localizedDescription])
                    }
                    guard let result = result else {
                        throw NSError(domain: "ObjectDetectionError", code: 0, userInfo: [NSLocalizedDescriptionKey: "No results found"])
                    }
                    
                    let output = self.makeOutputResult(objects: result)
                    DispatchQueue.main.async {
                        resolve(output)
                    }
                } catch let error as NSError {
                    let errorString = error.localizedDescription
                    let pData: [String: Any] = ["error": "On-Device object detection failed with error: \(errorString)"]
                    DispatchQueue.main.async {
                        resolve(pData)
                    }
                }
            }
        }
        
    }
    
  
    func getImageMetadata(from imageData: Data) -> (width: Int, height: Int, orientation: UIImage.Orientation)? {
        guard let imageSource = CGImageSourceCreateWithData(imageData as CFData, nil) else {
            print("Unable to create image source")
            return nil
        }

        guard let imageProperties = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any] else {
            print("Unable to get image properties")
            return nil
        }

        guard let pixelWidth = imageProperties[kCGImagePropertyPixelWidth] as? Int,
              let pixelHeight = imageProperties[kCGImagePropertyPixelHeight] as? Int,
              let orientationValue = imageProperties[kCGImagePropertyOrientation] as? Int,
              let orientation = UIImage.Orientation(rawValue: orientationValue) else {
            print("Missing or invalid image properties")
            return nil
        }

        return (width: pixelWidth, height: pixelHeight, orientation: orientation)
    }
    func uiImageToCVPixelBuffer(image: UIImage) -> CVPixelBuffer? {
        guard let cgImage = image.cgImage else {
            print("Failed to get cgImage from UIImage")
            return nil
        }
        
        let frameSize = CGSize(width: cgImage.width, height: cgImage.height)
        
        var pixelBuffer: CVPixelBuffer?
        let options: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ]
        
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(frameSize.width),
                                         Int(frameSize.height),
                                         kCVPixelFormatType_32ARGB,
                                         options as CFDictionary,
                                         &pixelBuffer)
        
        guard status == kCVReturnSuccess, let createdPixelBuffer = pixelBuffer else {
            print("Failed to create pixel buffer")
            return nil
        }
        
        CVPixelBufferLockBaseAddress(createdPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(createdPixelBuffer)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData,
                                width: Int(frameSize.width),
                                height: Int(frameSize.height),
                                bitsPerComponent: 8,
                                bytesPerRow: CVPixelBufferGetBytesPerRow(createdPixelBuffer),
                                space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        guard let context = context else {
            print("Failed to create CGContext")
            CVPixelBufferUnlockBaseAddress(createdPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
            return nil
        }
        
        context.draw(cgImage, in: CGRect(origin: .zero, size: frameSize))
        CVPixelBufferUnlockBaseAddress(createdPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return createdPixelBuffer
    }
    
    func pixelBufferToRGBData(pixelBuffer: CVPixelBuffer) -> Data? {
        // Lock the base address of the pixel buffer
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("Unable to get base address from pixel buffer")
            return nil
        }
        
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        
//        guard pixelFormat == kCVPixelFormatType_32BGRA else {
//            print("Pixel format not supported")
//            return nil
//        }
        
        var rgbData = Data(count: width * height * 3)
        
        rgbData.withUnsafeMutableBytes { rgbPtr in
            guard let rgbBaseAddress = rgbPtr.baseAddress else { return }
            let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
            
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIndex = y * bytesPerRow + x * 4
                    let rgbIndex = (y * width + x) * 3
                    
                    let blue = buffer[pixelIndex]
                    let green = buffer[pixelIndex + 1]
                    let red = buffer[pixelIndex + 2]
                    
                    rgbBaseAddress.storeBytes(of: red, toByteOffset: rgbIndex, as: UInt8.self)
                    rgbBaseAddress.storeBytes(of: green, toByteOffset: rgbIndex + 1, as: UInt8.self)
                    rgbBaseAddress.storeBytes(of: blue, toByteOffset: rgbIndex + 2, as: UInt8.self)
                }
            }
        }
        
        return rgbData
    }

   
    func createCustomModelsDirectory() -> String? {
        let fileManager = FileManager.default
        
        // Get the document directory path
        guard let documentDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("Could not find the document directory")
            return nil
        }
        
        // Create the custom_models directory path
        let customModelsDirectory = documentDirectory.appendingPathComponent("custom_models")
        
        // Create the custom_models directory if it doesn't exist
        do {
            try fileManager.createDirectory(at: customModelsDirectory, withIntermediateDirectories: true, attributes: nil)
            print("Custom models directory created at: \(customModelsDirectory)")
        } catch {
            print("Error creating custom_models directory: \(error.localizedDescription)")
            return nil
        }
        let resultPath = customModelsDirectory.path + "/"
        return resultPath
    }
    func getDocumentsDirectory() -> URL? {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths.first
    }
    func copyFile(from sourcePathString: String, to destinationPathString: String) {
        let fileManager = FileManager.default
        
        // Convert string paths to URL objects
        let sourceURL = URL(fileURLWithPath: sourcePathString)
        let destinationURL = URL(fileURLWithPath: destinationPathString)
        
        do {
            // Check if the file exists at the source path
            guard fileManager.fileExists(atPath: sourceURL.path) else {
                print("File does not exist at source path: \(sourceURL.path)")
                return
            }
            
            // Create the destination directory if it doesn't exist
            let destinationDirectory = destinationURL.deletingLastPathComponent()
            if !fileManager.fileExists(atPath: destinationDirectory.path) {
                try fileManager.createDirectory(at: destinationDirectory, withIntermediateDirectories: true, attributes: nil)
            }
            
            // Copy the file to the destination
            try fileManager.copyItem(at: sourceURL, to: destinationURL)
            print("File copied to: \(destinationURL.path)")
        } catch {
            print("Error copying file: \(error.localizedDescription)")
        }
    }
    
    
    func makeOutputResult(objects: [Object]?) -> [[String: Any]] {
        var output: [[String: Any]] = []
        guard let objects = objects else {
            return output
        }
               for object in objects {
            var detectedObject: [String: Any] = [:]
            detectedObject["bounding"] = self.makeBoundingResult(frame: object.frame)

            if let trackingID = object.trackingID {
                detectedObject["trackingID"] = trackingID
            }

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

    func makeBoundingResult(frame: CGRect) -> [String: CGFloat] {
        return [
            "originX": frame.origin.x,
            "originY": frame.origin.y,
            "width": frame.size.width,
            "height": frame.size.height
        ]
    }
    
   
}

