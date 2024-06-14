//
//  MlkitOdt.swift
//  MlkitOdt
//
//  Created by Ritesh Jariwala on 13/06/24.
//  Copyright © 2024 Facebook. All rights reserved.
//

import Foundation
import FirebaseMLModelDownloader

@objc public class MlkitOdt: NSObject {
    
    @objc public static func downloadModel(_ modelName: String, completion: @escaping (String?, NSError?) -> Void) {
        let modelDownloader = ModelDownloader.modelDownloader()
        let conditions = ModelDownloadConditions()
        
        modelDownloader.getModel(name: modelName, downloadType:ModelDownloadType.latestModel, cconditions: conditions) { result in
            switch result {
            case .success(let customModel):
                completion(customModel.path, nil)
            case .failure(let error):
                completion(nil, error as NSError)
            }
        }
    }
}

