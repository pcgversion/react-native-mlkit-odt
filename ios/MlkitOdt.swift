//
//  MlkitOdt.swift
//  MlkitOdt
//
//  Created by Ritesh Jariwala on 14/06/24.
//  Copyright © 2024 Facebook. All rights reserved.
//

import Foundation
import FirebaseMLModelDownloader

@objc public class MlkitOdtHelper: NSObject {
    
    @objc public static func downloadCustomModel(_ modelName: String, resolver: @escaping RCTPromiseResolveBlock, rejecter: @escaping RCTPromiseRejectBlock) {
        let modelDownloader = ModelDownloader.modelDownloader()
        let conditions = ModelDownloadConditions()
        print("comes here");
        modelDownloader.getModel(name: modelName, downloadType:ModelDownloadType.latestModel, conditions: conditions) { result in
            switch result {
            case .success(let customModel):
                resolver(customModel.path)
            case .failure(let error):
                rejecter("model download failed","Failed to download model", error)
            }
        }
    }
    
    @objc public static func sayHello() -> Void{
        print("Hello");
    }
}
