package com.example.projectvoice;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class WhisperHelper {

    private static final String TAG = "WhisperHelper";
    private Interpreter interpreter;

    // --- IMPORTANT ---
    // These are placeholders. You MUST determine the correct input/output shapes
    // and data types from the specific Whisper TFLite model documentation you are using.
    // Input is typically Mel Spectrogram data. Output is token IDs.
    private final int inputTensorIndex = 0; // Usually 0
    private final int outputTensorIndex = 0; // Usually 0
    // Example shapes (replace with actual):
    // private final int[] inputShape = new int[]{1, 80, 3000}; // e.g., [batch, mel_bins, frames]
    // private final int[] outputShape = new int[]{1, 200};    // e.g., [batch, max_tokens]


    public WhisperHelper(Context context, String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            modelPath = "whisper-tiny.tflite"; // Default model
        }
        try {
            Interpreter.Options options = new Interpreter.Options();
            // options.setNumThreads(4); // Optional: Configure threads
            // Consider adding delegates (GPU, NNAPI) for performance later
            interpreter = new Interpreter(loadModelFile(context, modelPath), options);
            Log.i(TAG, "TensorFlow Lite interpreter loaded successfully.");

            // Optional: Log input/output tensor details to verify
            logTensorDetails();

        } catch (IOException e) {
            Log.e(TAG, "Error loading TFLite model: " + e.getMessage(), e);
            interpreter = null; // Ensure interpreter is null if loading failed
        } catch (Exception e) {
            Log.e(TAG, "Unexpected error initializing interpreter: " + e.getMessage(), e);
            interpreter = null;
        }
    }

    // Overloaded constructor using default model path
    public WhisperHelper(Context context) {
         this(context, "whisper-tiny.tflite");
    }


    // Standard TFLite model loading utility
    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        inputStream.close(); // Close stream after mapping
        fileDescriptor.close();
        return mappedByteBuffer;
    }

    private void logTensorDetails() {
        if (interpreter != null) {
             try {
                 int inputCount = interpreter.getInputTensorCount();
                 int outputCount = interpreter.getOutputTensorCount();
                 Log.d(TAG, "Input Tensor Count: " + inputCount);
                 Log.d(TAG, "Output Tensor Count: " + outputCount);

                  if (inputCount > inputTensorIndex) {
                      Tensor inputTensor = interpreter.getInputTensor(inputTensorIndex);
                      Log.d(TAG, "Input Tensor (" + inputTensorIndex + ") Name: " + inputTensor.name());
                      Log.d(TAG, "Input Tensor (" + inputTensorIndex + ") Shape: " + Arrays.toString(inputTensor.shape()));
                      Log.d(TAG, "Input Tensor (" + inputTensorIndex + ") Type: " + inputTensor.dataType());
                  } else {
                       Log.w(TAG, "Input Tensor Index " + inputTensorIndex + " out of bounds (Count: " + inputCount + ")");
                  }


                  if (outputCount > outputTensorIndex) {
                      Tensor outputTensor = interpreter.getOutputTensor(outputTensorIndex);
                      Log.d(TAG, "Output Tensor (" + outputTensorIndex + ") Name: " + outputTensor.name());
                      Log.d(TAG, "Output Tensor (" + outputTensorIndex + ") Shape: " + Arrays.toString(outputTensor.shape()));
                      Log.d(TAG, "Output Tensor (" + outputTensorIndex + ") Type: " + outputTensor.dataType());
                  } else {
                       Log.w(TAG, "Output Tensor Index " + outputTensorIndex + " out of bounds (Count: " + outputCount + ")");
                  }

             } catch (Exception e) {
                  Log.e(TAG, "Error getting tensor details: " + e.getMessage(), e);
             }
        } else {
            Log.e(TAG, "Interpreter is null, cannot get tensor details.");
        }
    }

    /**
     * Transcribes audio data.
     *
     * @param audioData Preprocessed audio data (e.g., Mel Spectrogram) matching the model's input requirements.
     *                  This needs to be a Buffer (e.g., FloatBuffer, ByteBuffer) depending on the model's input type.
     * @return The raw output tensor buffer from the model (e.g., token IDs) as a Map, or null if inference fails.
     *         The map key is the output tensor index (usually 0).
     */
    public Map<Integer, Object> transcribe(Object audioData) {
        if (interpreter == null) {
            Log.e(TAG, "Interpreter not initialized.");
            return null;
        }
         if (audioData == null) {
             Log.e(TAG, "Input audio data is null.");
             return null;
         }


        try {
            // --- TODO: Prepare Output Buffer ---
            // You need to allocate the output buffer based on the output tensor shape and type.
            Tensor outputTensor = interpreter.getOutputTensor(outputTensorIndex);
            DataType outputDataType = outputTensor.dataType();
            int[] outputShape = outputTensor.shape();
            int outputBytes = outputTensor.numBytes(); // Size in bytes

            if (outputBytes <= 0) {
                 Log.e(TAG, "Calculated output tensor size is invalid: " + outputBytes);
                 return null; // Cannot allocate buffer
            }


            ByteBuffer outputBuffer = ByteBuffer.allocateDirect(outputBytes);
            outputBuffer.order(ByteOrder.nativeOrder()); // Crucial for direct buffers

            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(outputTensorIndex, outputBuffer);

            // --- Prepare Inputs ---
            Object[] inputs = new Object[]{audioData};

            // --- Run Inference ---
            Log.d(TAG, "Running inference...");
            interpreter.runForMultipleInputsOutputs(inputs, outputs);
            Log.d(TAG, "Inference complete.");

            // --- TODO: Process Output ---
            // The result is in the 'outputs' map, associated with outputTensorIndex.
            // The value is the ByteBuffer ('outputBuffer') allocated above, now filled with data.
            // You need to extract the data (e.g., token IDs) from this buffer based on its
            // data type (outputDataType) and shape (outputShape), and then convert it to text
            // using the Whisper vocabulary. This is a complex step involving token decoding.

            return outputs; // Return the map containing the raw output buffer

        } catch (IllegalArgumentException e) {
            Log.e(TAG, "IllegalArgumentException during inference. Check input data type/shape: " + e.getMessage(), e);
             return null;
        } catch (Exception e) {
            Log.e(TAG, "Error during inference: " + e.getMessage(), e);
            return null;
        }
    }

    // Helper method (example) - must match your model's input type
    public DataType getInputDataType() {
        if (interpreter != null && interpreter.getInputTensorCount() > inputTensorIndex) {
            return interpreter.getInputTensor(inputTensorIndex).dataType();
        }
        return null; // Or a default/error value
    }

    // Helper method (example) - must match your model's input shape
     public int[] getInputShape() {
         if (interpreter != null && interpreter.getInputTensorCount() > inputTensorIndex) {
             return interpreter.getInputTensor(inputTensorIndex).shape();
         }
         return null;
     }

    // Call this when the helper is no longer needed (e.g., in Activity's onDestroy)
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
            Log.i(TAG, "Interpreter closed.");
        }
    }
}