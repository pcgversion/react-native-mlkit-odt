package com.reactnativemlkitodt

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

import com.mrousavy.camera.frameprocessor.FrameProcessorPlugin

class MlkitOdtPackage : ReactPackage {
    override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> {
        FrameProcessorPlugin.register(MlkitOdtFrameProcessorPlugin(reactContext))
        return listOf(MlkitOdtModule(reactContext))
    }

    override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> {
        return emptyList()
    }
}
