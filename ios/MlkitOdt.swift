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
    @objc
    func sayHello()->Void{
        print("Test")
    }
   
}

