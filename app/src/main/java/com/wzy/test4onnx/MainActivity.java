package com.wzy.test4onnx;

import ai.onnxruntime.*;

import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.res.ResourcesCompat;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.wzy.test4onnx.databinding.ActivityMainBinding;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding binding;
    private Python py;
    private static final String TAG = "MainActivity";
    private final Handler handler = new Handler(Looper.getMainLooper()) {
        @Override
        public void handleMessage(@NonNull Message msg) {
            switch (msg.what) {
                // use onnx to infer
                case 0x01:
                    Bundle data = msg.getData();
                    Log.d("MainActivityHandler", "handleMessage: preprocessing result inbound");
                    new Thread(() -> {
                        try {
                            MDFloatArray img_in_wrapper = (MDFloatArray) data.getSerializable("img_in");
                            float[][][][] img_in = img_in_wrapper.FourDValue;
                            // obtain an ortEnvironment and use this environment to get session for infer
                            OrtEnvironment ortEnvironment = OrtEnvironment.getEnvironment();
                            // a wrapper to use kotlin method
                            ModelUtil modelUtil = new ModelUtil();
                            OrtSession session = ortEnvironment.createSession(modelUtil.readModel(getApplicationContext()));
                            // build floatbuffer for onnxruntime
                            // this will be the input
                            FloatBuffer buffer = FloatBuffer.allocate(3 * 1024 * 1024);
                            // remember to rewind twice
                            buffer.rewind();
                            for (int i = 0; i < img_in.length; i++) {
                                for (int j = 0; j < img_in[0].length; j++) {
                                    for (int k = 0; k < img_in[0][0].length; k++) {
                                        for (int l = 0; l < img_in[0][0][0].length; l++) {
                                            buffer.put(img_in[i][j][k][l]);
                                        }
                                    }
                                }
                            }
                            buffer.rewind();
                            // OnnxTensor is the real form of ort. Last param is shape of the backend buffer
                            OnnxTensor input = OnnxTensor.createTensor(ortEnvironment, buffer, new long[]{1, 3, 1024, 1024});
                            // output of onnx
                            OrtSession.Result output = session.run(Collections.singletonMap(session.getInputNames().iterator().next(), input));
                            // the demo onnx model have 1 input and 3 output. See raw/comictextdetector.onnx
                            // origin:    https://github.com/zyddnys/manga-image-translator
                            float[][][] blks = (float[][][]) output.get(0).getValue();
                            float[][][][] mask = (float[][][][]) output.get(1).getValue();
                            float[][][][] lines_map = (float[][][][]) output.get(2).getValue();
                            int limit = blks[0].length / 2;
                            float[][][] firstHalfBlks = new float[blks.length][limit][blks[0][0].length];
                            float[][][] lastHalfBlks = new float[blks.length][blks[0].length - limit][blks[0][0].length];
                            // the size of global ref table is 51200, but the shape of our onnx output(blks) is (1, 64512, 7)
                            // so directly use this as input will cause overflow. Need to split
                            for (int i = 0; i < blks.length; i++) {
                                System.arraycopy(blks[i], 0, firstHalfBlks[i], 0, limit);
                                if (blks[0].length - limit >= 0) {
                                    System.arraycopy(blks[i], limit, lastHalfBlks[i], 0, blks[0].length - limit);
                                }
                            }

                            output.close();
                            session.close();
                            ortEnvironment.close();
                            Message message = Message.obtain();
                            message.what = 0x02;
                            // add result and pass to main thread
                            data.putSerializable("firstHalfBlks", new MDFloatArray(firstHalfBlks));
                            data.putSerializable("lastHalfBlks", new MDFloatArray(lastHalfBlks));
                            data.putSerializable("mask", new MDFloatArray(mask));
                            data.putSerializable("lines_map", new MDFloatArray(lines_map));
                            message.setData(data);
                            handler.sendMessage(message);
                        } catch (OrtException e) {
                            e.printStackTrace();
                        }
                        Log.d(TAG, "handleMessage: onnx fin");
                    }).start();
                    break;
                // use python to perform postprocessing
                case 0x02:
                    Bundle data2 = msg.getData();
                    Log.d("MainActivityHandler", "handleMessage: onnx result inbound");
                    new Thread(() -> {
                        float[][][] firstHalfBlks = ((MDFloatArray) data2.getSerializable("firstHalfBlks")).ThreeDValue;
                        float[][][] lastHalfBlks = ((MDFloatArray) data2.getSerializable("lastHalfBlks")).ThreeDValue;
                        float[][][][] mask = ((MDFloatArray) data2.getSerializable("mask")).FourDValue;
                        float[][][][] lines_map = ((MDFloatArray) data2.getSerializable("lines_map")).FourDValue;
                        PyObject np = py.getModule("numpy");
                        PyObject postprocessingResult = py.getModule("main")
                                .callAttr("postprocessing",
                                        data2.getInt("im_h"),
                                        data2.getInt("im_w"),
                                        data2.getInt("dw"),
                                        data2.getInt("dh"),
                                        data2.getString("img_path"),
                                        np.callAttr("array", (Object) firstHalfBlks),
                                        np.callAttr("array", (Object) lastHalfBlks),
                                        np.callAttr("array", (Object) mask),
                                        np.callAttr("array", (Object) lines_map));
                        // we can see the result (python console output) from logcat
                        Log.d(TAG, "onCreate: success!");
                    }).start();
                    break;
            }
        }
    };

    /**
     * preprocessing for onnx input in a new thread
     */
    private final Thread preprocessing = new Thread(() -> {
        // release demo picture to private folder for python
        Drawable drawable = ResourcesCompat.getDrawable(getResources(), R.drawable.screenshot, null);
        Bitmap bitmap = ((BitmapDrawable) drawable).getBitmap();
        File testDir = this.getExternalFilesDir("test");
        File file = new File(testDir.getAbsolutePath() + "/screenshot.jpg");
        FileOutputStream out = null;
        try {
            out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
            out.close();
            // use chaquopy to run python code
            // return is a list of pyObject and need toJava() to transform to a readable java object
            PyObject preprocessingResult = py.getModule("main").callAttr("preprocessing", file.getAbsolutePath());
            List<PyObject> preResult = preprocessingResult.asList();
            float[][][][] img_in = preResult.get(0).toJava(float[][][][].class);
            // unable to convert uint8 to int array in java like float array does
            // so only pass the file path to main thread
//                    int[][][] img = preResult.get(5).toJava(int[][][].class);
            Message message = Message.obtain();
            message.what = 0x01;
            Bundle data = new Bundle();
            data.putInt("im_h", preResult.get(1).toJava(int.class));
            data.putInt("im_w", preResult.get(2).toJava(int.class));
            data.putInt("dw", preResult.get(3).toJava(int.class));
            data.putInt("dh", preResult.get(4).toJava(int.class));
            data.putString("img_path", file.getAbsolutePath());
            // MDFloatArray is a wrapper of multi-dimensional float array
            // There might be a more elegant way to implement that.
            data.putSerializable("img_in", new MDFloatArray(img_in));
//                    data.putSerializable("img", new MDIntArray(img));
            message.setData(data);
            handler.sendMessage(message);
        } catch (IOException e) {
            e.printStackTrace();
        }
    });

    /**
     * This is the one thread version.
     * But due to unknown reason (might be conflict between onnx and chaquopy), postprocessing in python will cause
     * chaquopy no response.
     * Only for an overview.
     * For the detail comment please see other standalone thread.
     */
    private final Thread all = new Thread(() -> {
        Drawable drawable = ResourcesCompat.getDrawable(getResources(), R.drawable.screenshot, null);
        Bitmap bitmap = ((BitmapDrawable) drawable).getBitmap();
        File testDir = this.getExternalFilesDir("test");
        File file = new File(testDir.getAbsolutePath() + "/screenshot.jpg");
        FileOutputStream out = null;
        try {
            out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
            out.close();
            PyObject preprocessingResult = py.getModule("main").callAttr("preprocessing", file.getAbsolutePath());
            List<PyObject> preResult = preprocessingResult.asList();
            float[][][][] img_in = preResult.get(0).toJava(float[][][][].class);
            OrtEnvironment ortEnvironment = OrtEnvironment.getEnvironment();
            ModelUtil model = new ModelUtil();
            OrtSession session = ortEnvironment.createSession(model.readModel(getApplicationContext()));
            FloatBuffer buffer = FloatBuffer.allocate(3 * 1024 * 1024);
            buffer.rewind();
            for (int i = 0; i < img_in.length; i++) {
                for (int j = 0; j < img_in[0].length; j++) {
                    for (int k = 0; k < img_in[0][0].length; k++) {
                        for (int l = 0; l < img_in[0][0][0].length; l++) {
                            buffer.put(img_in[i][j][k][l]);
                        }
                    }
                }
            }
            buffer.rewind();
            OnnxTensor input = OnnxTensor.createTensor(ortEnvironment, buffer, new long[]{1, 3, 1024, 1024});
            OrtSession.Result output = session.run(Collections.singletonMap(session.getInputNames().iterator().next(), input));
            float[][][] blks = (float[][][]) output.get(0).getValue();
            float[][][][] mask = (float[][][][]) output.get(1).getValue();
            float[][][][] lines_map = (float[][][][]) output.get(2).getValue();
            int limit = blks[0].length / 2;
            float[][][] firstHalfBlks = new float[blks.length][limit][blks[0][0].length];
            float[][][] lastHalfBlks = new float[blks.length][blks[0].length - limit][blks[0][0].length];
            for (int i = 0; i < blks.length; i++) {
                System.arraycopy(blks[i], 0, firstHalfBlks[i], 0, limit);
                if (blks[0].length - limit >= 0) {
                    System.arraycopy(blks[i], limit, lastHalfBlks[i], 0, blks[0].length - limit);
                }

            }
            output.close();
            session.close();
            ortEnvironment.close();
            PyObject np = py.getModule("numpy");
            //bug : nms in postprocessing not return
            // may be conflict with onnx
            PyObject postprocessingResult = py.getModule("main")
                    .callAttr("postprocessing",
                            preResult.get(1),
                            preResult.get(2),
                            preResult.get(3),
                            preResult.get(4),
                            preResult.get(5),
                            np.callAttr("array", (Object) firstHalfBlks),
                            np.callAttr("array", (Object) lastHalfBlks),
                            np.callAttr("array", (Object) mask),
                            np.callAttr("array", (Object) lines_map));
        } catch (IOException | OrtException e) {
            e.printStackTrace();
        }
    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        py = Python.getInstance();

        setContentView(binding.getRoot());

        binding.button.setOnClickListener(v -> {
            preprocessing.start();
        });

    }
}