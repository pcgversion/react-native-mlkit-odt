require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "react-native-mlkit-odt"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "10.0" }
  #s.source       = { :git => "https://github.com/artikq/react-native-mlkit-odt.git", :tag => "#{s.version}" }
  s.source       = { :branch => "https://github.com/pcgversion/react-native-mlkit-odt/tree/react-native-mlkit-odt-v3"}

  s.source_files = "ios/**/*.{h,m,mm,swift}"
   # Explicitly define public headers.
  # This helps CocoaPods understand the module structure and ensures
  # that 'vision_camera_ocr.h' is findable within the module.
  s.public_header_files = "ios/react_native_mlkit_odt.h"
  s.dependency "React-Core"
  s.dependency "VisionCamera"
  s.dependency 'GoogleMLKit/ObjectDetection'
  s.dependency 'FirebaseMLModelDownloader', '10.12.0-beta'
  s.dependency "TensorFlowLiteSwift"
end
