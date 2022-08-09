package com.wzy.test4onnx

import android.content.Context

internal class ABC {

    fun readModel(context: Context): ByteArray {
        return context.resources.openRawResource(R.raw.comictextdetector_with_runtime_opt).readBytes();
    }
}