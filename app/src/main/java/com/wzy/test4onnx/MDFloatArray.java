package com.wzy.test4onnx;

import java.io.Serializable;

public class MDFloatArray implements Serializable {
    public float[] oneDValue;
    public float[][] twoDValue;
    public float[][][] ThreeDValue;
    public float[][][][] FourDValue;

    public MDFloatArray(float[] array) {
        oneDValue = array;
    }

    public MDFloatArray(float[][] array) {
        twoDValue = array;
    }

    public MDFloatArray(float[][][] array) {
        ThreeDValue = array;
    }

    public MDFloatArray(float[][][][] array) {
        FourDValue = array;
    }
}
