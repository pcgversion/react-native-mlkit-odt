//
//  MlkitOdtHelper.h
//  MlkitOdt
//
//  Created by Ritesh Jariwala on 17/06/24.
//  Copyright © 2024 Facebook. All rights reserved.
//

#ifndef MlkitOdtHelper_h
#define MlkitOdtHelper_h

@interface MlkitOdtHelper: NSOjbect
+(void)downloadModel(NSString *)modelName resolver:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock) reject;
@end

#endif /* MlkitOdtHelper_h */
