package com.wzy.test4onnx;

import java.io.Serializable;

public class MDIntArray implements Serializable {
    public int[] oneDValue;
    public int[][] twoDValue;
    public int[][][] ThreeDValue;
    public int[][][][] FourDValue;

    public MDIntArray(int[] array) {
        oneDValue = array;
    }

    public MDIntArray(int[][] array) {
        twoDValue = array;
    }

    public MDIntArray(int[][][] array) {
        ThreeDValue = array;
    }

    public MDIntArray(int[][][][] array) {
        FourDValue = array;
    }
}
