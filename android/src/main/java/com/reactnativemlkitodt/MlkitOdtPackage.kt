package com.reactnativemlkitodt

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry

class MlkitOdtPackage : ReactPackage {
    override fun createNativeModules(reactContext: ReactApplicationContext): List<NativeModule> {
        //FrameProcessorPlugin.register(MlkitOdtFrameProcessorPlugin(reactContext))
        FrameProcessorPluginRegistry.addFrameProcessorPlugin("detectObjects") { proxy, options ->
            MlkitOdtFrameProcessorPlugin(reactContext, proxy, options)
        }
        return listOf(MlkitOdtModule(reactContext))
    }

    override fun createViewManagers(reactContext: ReactApplicationContext): List<ViewManager<*, *>> {
        return emptyList()
    }
}
