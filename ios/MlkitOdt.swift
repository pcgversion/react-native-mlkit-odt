//
//  MlkitOdt.swift
//  MlkitOdt
//
//  Created by Ritesh Jariwala on 14/06/24.
//  Copyright © 2024 Facebook. All rights reserved.
//

import Foundation
import FirebaseMLModelDownloader

@objc(MlkitOdt)
class MlkitOdt: NSObject {
  @objc
    func downloadCustomModel(_ modelName: String, resolver resolve: @escaping RCTPromiseResolveBlock, rejecter reject: @escaping RCTPromiseRejectBlock) {
        let modelDownloader = ModelDownloader.modelDownloader()
        let conditions = ModelDownloadConditions()
        print("comes here");
        modelDownloader.getModel(name: modelName, downloadType:ModelDownloadType.latestModel, conditions: conditions) { result in
            switch (result) {
            case .success(let customModel):
                NSLog("@custom model donwloaded:, %@", customModel.path);
                let resultData:[String:Any] = ["success":true];
                resolve(resultData)
            case .failure(let error):
                reject("model download failed","Failed to download model", error)
            }
        }
    }
    
}
