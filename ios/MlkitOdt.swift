//
//  MlkitOdt.swift
//  MlkitOdt
//
//  Created by Ritesh Jariwala on 13/06/24.
//  Copyright © 2024 Facebook. All rights reserved.
//

import Foundation
import FirebaseMLModelDownloader

@objc public class FirebaseModelDownloaderHelper: NSObject {
    
    @objc public static func downloadModel(modelName: String, completion: @escaping (String?, NSError?) -> Void) {
        let conditions = ModelDownloadConditions(allowsCellularAccess: true, allowsBackgroundDownloading: true)
        let modelDownloader = ModelDownloader.modelDownloader()

        modelDownloader.getModel(name: modelName, conditions: conditions) { result in
            switch result {
            case .success(let customModel):
                completion(customModel.path, nil)
            case .failure(let error):
                completion(nil, error as NSError)
            }
        }
    }
}
