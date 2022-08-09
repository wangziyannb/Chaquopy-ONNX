package com.wzy.test4onnx

import android.content.Context

internal class ModelUtil {

    fun readModel(context: Context): ByteArray {
        return context.resources.openRawResource(R.raw.comictextdetector).readBytes();
    }
}