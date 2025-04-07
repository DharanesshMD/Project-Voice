package com.example.projectvoice;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;
import org.tensorflow.lite.DataType;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
// Consider adding InterpreterApi and TensorApi if using newer TFLite features

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

    // These indices usually remain 0, but check your specific model if needed.
    private final int inputTensorIndex = 0;
    private final int outputTensorIndex = 0;

    // Optional: Store tensor details after loading for quicker access
    private DataType inputDataType = null;
    private int[] inputShape = null;
    private DataType outputDataType = null;
    private int[] outputShape = null;
    private int outputTensorSizeInBytes = -1;


    public WhisperHelper(Context context, String modelPath) throws IOException {
        if (modelPath == null || modelPath.isEmpty()) {
            Log.w(TAG, "Model path is null or empty, using default 'whisper-tiny.tflite'");
            modelPath = "whisper-tiny.tflite"; // Default model
        }
        try {
            Interpreter.Options options = new Interpreter.Options();
            // Optional: Configure threads, delegates (GPU, NNAPI), etc.
             options.setNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() / 2)); // Example: Use half available cores
            // options.addDelegate(new GpuDelegate()); // Requires GPU delegate dependency
            // options.addDelegate(new NnApiDelegate()); // Requires NNAPI delegate dependency

            MappedByteBuffer modelBuffer = loadModelFile(context, modelPath);
            interpreter = new Interpreter(modelBuffer, options);
            Log.i(TAG, "TensorFlow Lite interpreter loaded successfully from: " + modelPath);

            // Get and store tensor details
            logAndStoreTensorDetails();

        } catch (IOException e) {
            Log.e(TAG, "IOException loading TFLite model '" + modelPath + "': " + e.getMessage());
            interpreter = null; // Ensure interpreter is null if loading failed
            throw e; // Re-throw exception so caller knows initialization failed
        } catch (Exception e) { // Catch other potential runtime errors during initialization
            Log.e(TAG, "Unexpected error initializing interpreter: " + e.getMessage(), e);
            interpreter = null;
            // Wrap in IOException or a custom exception if needed
             throw new IOException("Failed to initialize TFLite interpreter", e);
        }
    }

    // Overloaded constructor using default model path
    public WhisperHelper(Context context) throws IOException {
         this(context, "whisper-tiny.tflite");
    }


    // Standard TFLite model loading utility
    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = null;
        FileInputStream inputStream = null;
        FileChannel fileChannel = null;
        try {
            fileDescriptor = context.getAssets().openFd(modelPath);
            inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } finally {
            // Close resources in reverse order of opening
            if (fileChannel != null) try { fileChannel.close(); } catch (IOException e) { Log.e(TAG, "Error closing FileChannel", e); }
            if (inputStream != null) try { inputStream.close(); } catch (IOException e) { Log.e(TAG, "Error closing FileInputStream", e); }
            if (fileDescriptor != null) try { fileDescriptor.close(); } catch (IOException e) { Log.e(TAG, "Error closing AssetFileDescriptor", e); }
        }
    }

    private void logAndStoreTensorDetails() {
        if (interpreter == null) {
            Log.e(TAG, "Interpreter is null, cannot get tensor details.");
            return;
        }
        try {
            int inputCount = interpreter.getInputTensorCount();
            int outputCount = interpreter.getOutputTensorCount();
            Log.d(TAG, "Input Tensor Count: " + inputCount);
            Log.d(TAG, "Output Tensor Count: " + outputCount);

            if (inputCount > inputTensorIndex) {
                Tensor inputTensor = interpreter.getInputTensor(inputTensorIndex);
                inputDataType = inputTensor.dataType();
                inputShape = inputTensor.shape().clone(); // Clone shape array
                Log.d(TAG, "Input Tensor (" + inputTensorIndex + ") Name: " + inputTensor.name());
                Log.d(TAG, "Input Tensor (" + inputTensorIndex + ") Shape: " + Arrays.toString(inputShape));
                Log.d(TAG, "Input Tensor (" + inputTensorIndex + ") Type: " + inputDataType);
            } else {
                 Log.w(TAG, "Input Tensor Index " + inputTensorIndex + " out of bounds (Count: " + inputCount + ")");
            }

            if (outputCount > outputTensorIndex) {
                Tensor outputTensor = interpreter.getOutputTensor(outputTensorIndex);
                outputDataType = outputTensor.dataType();
                outputShape = outputTensor.shape().clone(); // Clone shape array
                outputTensorSizeInBytes = outputTensor.numBytes(); // Store size
                Log.d(TAG, "Output Tensor (" + outputTensorIndex + ") Name: " + outputTensor.name());
                Log.d(TAG, "Output Tensor (" + outputTensorIndex + ") Shape: " + Arrays.toString(outputShape));
                Log.d(TAG, "Output Tensor (" + outputTensorIndex + ") Type: " + outputDataType);
                Log.d(TAG, "Output Tensor (" + outputTensorIndex + ") Size (bytes): " + outputTensorSizeInBytes);

                 if (outputTensorSizeInBytes <= 0) {
                     Log.e(TAG, "Output tensor size calculation resulted in <= 0 bytes. Check model output.");
                 }

            } else {
                 Log.w(TAG, "Output Tensor Index " + outputTensorIndex + " out of bounds (Count: " + outputCount + ")");
            }

        } catch (Exception e) {
             Log.e(TAG, "Error getting tensor details: " + e.getMessage(), e);
             // Reset stored details on error
             inputDataType = null;
             inputShape = null;
             outputDataType = null;
             outputShape = null;
             outputTensorSizeInBytes = -1;
        }
    }

    /**
     * Transcribes preprocessed audio data.
     *
     * @param preprocessedAudioData A Buffer (usually ByteBuffer or FloatBuffer) containing the
     *                              audio data already converted to the model's required format
     *                              (e.g., Mel Spectrogram), shape, and data type.
     *                              Ensure the buffer is rewound if necessary before passing.
     * @return A Map containing the raw output tensor buffer(s) from the model, or null if inference fails.
     *         The map key is the output tensor index (e.g., 0). The value is typically a ByteBuffer.
     */
    public Map<Integer, Object> transcribe(Object preprocessedAudioData) {
        if (interpreter == null) {
            Log.e(TAG, "Interpreter not initialized.");
            return null;
        }
        if (preprocessedAudioData == null) {
             Log.e(TAG, "Input preprocessedAudioData is null.");
             return null;
        }
        if (outputTensorSizeInBytes <= 0 || outputDataType == null) {
            Log.e(TAG, "Output tensor details not available or invalid. Cannot prepare output buffer.");
            return null;
        }

        try {
            // --- Prepare Output Buffer ---
            // Allocate the output buffer based on stored details. Use direct buffer.
            ByteBuffer outputBuffer = ByteBuffer.allocateDirect(outputTensorSizeInBytes);
            outputBuffer.order(ByteOrder.nativeOrder()); // Crucial for direct buffers

            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(outputTensorIndex, outputBuffer);

            // --- Prepare Inputs ---
            // Assuming single input at inputTensorIndex
            Object[] inputs = new Object[]{preprocessedAudioData};

            // --- Run Inference ---
            // Log.d(TAG, "Running inference..."); // Moved logging to MainActivity for timing
            interpreter.runForMultipleInputsOutputs(inputs, outputs);
            // Log.d(TAG, "Inference complete.");

            // Rewind the output buffer before returning so the caller can read from the start
            outputBuffer.rewind();

            return outputs; // Return the map containing the raw output buffer

        } catch (IllegalArgumentException e) {
             // This often indicates a mismatch between the input data provided (shape/type)
             // and what the model expects.
            Log.e(TAG, "IllegalArgumentException during inference. Check input data format/shape/type: " + e.getMessage(), e);
            // Log input buffer details if possible (be careful with large data)
            if (preprocessedAudioData instanceof ByteBuffer) {
                ByteBuffer bb = (ByteBuffer) preprocessedAudioData;
                Log.e(TAG, "Input Buffer details: capacity=" + bb.capacity() + ", limit=" + bb.limit() + ", position=" + bb.position() + ", isDirect=" + bb.isDirect());
            }
            return null;
        } catch (Exception e) { // Catch other runtime TFLite errors
            Log.e(TAG, "Error during model inference: " + e.getMessage(), e);
            return null;
        }
    }

    // Helper method to get stored input data type
    public DataType getInputDataType() {
        return inputDataType;
    }

    // Helper method to get stored input shape (returns a copy)
     public int[] getInputShape() {
         return (inputShape != null) ? inputShape.clone() : null;
     }

    // Helper method to get stored output data type
    public DataType getOutputDataType() {
        return outputDataType;
    }

    // Helper method to get stored output shape (returns a copy)
     public int[] getOutputShape() {
         return (outputShape != null) ? outputShape.clone() : null;
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