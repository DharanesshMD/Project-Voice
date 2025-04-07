package com.example.projectvoice;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.projectvoice.R;

import org.tensorflow.lite.DataType; // Import DataType

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private final int outputTensorIndex = 0;

    private WhisperHelper whisperHelper;
    private TextView textViewStatus;
    private TextView textViewResult;
    private Button buttonStartRecord;
    private Button buttonStopRecord;

    // --- Audio Recording Setup ---
    private AudioRecord audioRecord;
    private boolean isRecording = false;
    private final int sampleRate = 16000; // Whisper models typically expect 16kHz
    private final int channelConfig = AudioFormat.CHANNEL_IN_MONO;
    private final int audioFormat = AudioFormat.ENCODING_PCM_16BIT; // 16-bit PCM
    private int bufferSize = AudioRecord.ERROR_BAD_VALUE; // Initialize with error

    // ExecutorService for background tasks (replacement for Kotlin Coroutines)
    private final ExecutorService executorService = Executors.newSingleThreadExecutor();
    // Handler to post results back to the main thread
    private final Handler mainHandler = new Handler(Looper.getMainLooper());


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main); // Link to your layout file

        textViewStatus = findViewById(R.id.textViewStatus);
        textViewResult = findViewById(R.id.textViewResult);
        buttonStartRecord = findViewById(R.id.buttonStartRecord);
        buttonStopRecord = findViewById(R.id.buttonStopRecord);

        // Initialize Whisper Helper
        // Consider running this in a background thread if model loading is very slow
        try {
            whisperHelper = new WhisperHelper(this);
            textViewStatus.setText("Status: Model Loaded");
        } catch (Exception e) {
            textViewStatus.setText("Status: Error loading model");
            Log.e(TAG, "Error initializing WhisperHelper", e);
            Toast.makeText(this, "Failed to load model: " + e.getMessage(), Toast.LENGTH_LONG).show();
            // Disable buttons if model fails to load
            buttonStartRecord.setEnabled(false);
            buttonStopRecord.setEnabled(false);
        }

        buttonStartRecord.setOnClickListener(v -> startRecording());
        buttonStopRecord.setOnClickListener(v -> stopRecordingAndTranscribe());

        // Calculate buffer size
        bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat);
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            Log.w(TAG, "Min buffer size calculation failed. Using default.");
            bufferSize = sampleRate * 2; // Set a default buffer size (e.g., 2 seconds)
        }
         Log.d(TAG,"AudioRecord buffer size: " + bufferSize);
    }

    private void requestAudioPermission() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.RECORD_AUDIO},
                REQUEST_RECORD_AUDIO_PERMISSION);
    }

    private boolean checkAudioPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
    }

    private void startRecording() {
        if (whisperHelper == null) {
             Toast.makeText(this, "Model not ready.", Toast.LENGTH_SHORT).show();
             return;
        }
        if (!checkAudioPermission()) {
            requestAudioPermission();
            return;
        }

        if (isRecording) {
            Toast.makeText(this, "Already recording.", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            // Check buffer size again before creating AudioRecord
             if (bufferSize <= 0) {
                 Log.e(TAG, "Invalid bufferSize: " + bufferSize);
                 Toast.makeText(this, "Audio recording configuration error.", Toast.LENGTH_SHORT).show();
                 return;
             }

            audioRecord = new AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    channelConfig,
                    audioFormat,
                    bufferSize
            );

            if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                 Log.e(TAG, "AudioRecord initialization failed. State: " + audioRecord.getState());
                 Toast.makeText(this, "Audio recording failed to initialize.", Toast.LENGTH_SHORT).show();
                 releaseAudioRecord(); // Clean up if failed
                 return;
            }

            audioRecord.startRecording();
            isRecording = true;
            textViewStatus.setText("Status: Recording...");
            textViewResult.setText(""); // Clear previous results
            buttonStartRecord.setEnabled(false);
            buttonStopRecord.setEnabled(true);
            Log.i(TAG, "Recording started.");

            // Note: Actual reading will happen in stopRecordingAndTranscribe on background thread

        } catch (SecurityException e) {
             Log.e(TAG, "SecurityException starting recording: " + e.getMessage(), e);
             Toast.makeText(this, "Permission denied.", Toast.LENGTH_SHORT).show();
             resetRecordingState();
        } catch (IllegalStateException e) {
            Log.e(TAG, "IllegalStateException starting recording: " + e.getMessage(), e);
            Toast.makeText(this, "Failed to start recording.", Toast.LENGTH_SHORT).show();
            resetRecordingState();
        } catch (IllegalArgumentException e) {
             Log.e(TAG, "IllegalArgumentException starting recording (check params): " + e.getMessage(), e);
             Toast.makeText(this, "Audio recording configuration error.", Toast.LENGTH_SHORT).show();
             resetRecordingState();
        } catch (Exception e) { // Catch unexpected errors
             Log.e(TAG, "Unexpected error starting recording: " + e.getMessage(), e);
             Toast.makeText(this, "An error occurred.", Toast.LENGTH_SHORT).show();
             resetRecordingState();
        }
    }

     private void stopRecordingAndTranscribe() {
        if (!isRecording || audioRecord == null) {
            if (audioRecord == null && isRecording) {
                 // State inconsistency, reset
                 Log.w(TAG, "isRecording is true but audioRecord is null. Resetting state.");
                 resetRecordingState();
            } else {
                Toast.makeText(this, "Not recording.", Toast.LENGTH_SHORT).show();
            }
            return;
        }

        textViewStatus.setText("Status: Stopping and Processing...");
        buttonStopRecord.setEnabled(false); // Disable stop button during processing

        // Submit the processing task to the background thread
        executorService.submit(() -> {
            // Keep local copy of audioRecord to avoid race conditions if resetRecordingState is called early
            AudioRecord recorderToProcess = audioRecord;
            boolean recordingStoppedSuccessfully = false;

            // Stop recording and release resources
            try {
                if (recorderToProcess != null && recorderToProcess.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                    recorderToProcess.stop();
                     recordingStoppedSuccessfully = true;
                    Log.i(TAG, "AudioRecord stopped.");
                } else {
                    Log.w(TAG,"AudioRecord was not recording or null when stop was called.");
                }
            } catch (IllegalStateException e) {
                 Log.e(TAG, "IllegalStateException stopping recording: " + e.getMessage(), e);
                  // Attempt to continue processing if possible, but log the error
            } finally {
                 // Release MUST happen after stop, even if stop throws error
                 releaseAudioRecord(); // Use synchronized method
                 // Set isRecording false *after* releasing the recorder instance used by this task
                 isRecording = false;
                 Log.i(TAG,"Finished stopping/releasing AudioRecord in background task.");
            }


            // --- TODO: Read Audio Data ---
            // If you need to read data AFTER stopping, it must be done before releaseAudioRecord().
            // Usually, you read *while* recording in a separate thread loop.
            // Since we are doing a simple record-then-process, we assume no data reading is needed here.
            // If you were continuously recording, you'd accumulate data in a buffer before this point.


             // --- TODO: Preprocess Audio Data ---
             // CRITICAL STEP: Convert raw PCM audio (if you captured it) into the
             // format expected by the Whisper model (Mel Spectrogram).
             // This requires FFT, Mel filterbank, normalization, padding/truncation.
             // You need the EXACT input shape and type from your model!
             if (whisperHelper == null) {
                  Log.e(TAG, "WhisperHelper is null, cannot preprocess or transcribe.");
                  updateUI("Error: Model helper not available.", "Status: Error");
                  return; // Exit background task
             }

             DataType inputDataType = whisperHelper.getInputDataType();
             int[] inputShape = whisperHelper.getInputShape();

             if (inputDataType == null || inputShape == null) {
                  Log.e(TAG, "Could not get model input details from WhisperHelper.");
                  updateUI("Error: Failed to get model input info.", "Status: Error");
                  return; // Exit background task
             }
              Log.d(TAG, "Model Input Shape: " + Arrays.toString(inputShape) + ", Type: " + inputDataType);


             // Placeholder: Create a dummy input buffer (REPLACE THIS with real preprocessing)
             ByteBuffer inputBuffer = createDummyInputBuffer(inputShape, inputDataType);


             // --- Run Transcription ---
             Map<Integer, Object> transcriptionOutput = null;
             String resultText;

             if (inputBuffer != null) { // Only run if dummy input created (or real preprocessing succeeded)
                 Log.d(TAG, "Calling whisperHelper.transcribe...");
                 transcriptionOutput = whisperHelper.transcribe(inputBuffer); // Pass the *real* preprocessed buffer
                 Log.d(TAG, "Transcription attempt finished.");
             } else {
                  Log.e(TAG,"Cannot run transcription, input buffer is null");
             }


             // --- TODO: Postprocess Output ---
             // The 'transcriptionOutput' map contains the raw output tensor buffer(s).
             // Extract token IDs from the ByteBuffer associated with the outputTensorIndex.
             // Use the Whisper vocabulary (from .bin file) to decode these IDs into text.
             if (transcriptionOutput != null && transcriptionOutput.containsKey(outputTensorIndex)) {
                  Object rawOutput = transcriptionOutput.get(outputTensorIndex);
                  // TODO: Implement real token decoding here based on the type of rawOutput (ByteBuffer)
                  resultText = "Transcription Raw Output Received (Needs Decoding):\n" + rawOutput.toString();
             } else {
                  resultText = "Transcription failed or input was not processed.";
             }

             // Update UI on the main thread
             updateUI(resultText, "Status: Idle");
             Log.i(TAG,"Background processing complete.");
        });
    }

     // Helper to create a dummy buffer - REPLACE with actual audio preprocessing
     private ByteBuffer createDummyInputBuffer(int[] shape, DataType dataType) {
         if (shape == null || shape.length == 0) {
             Log.e(TAG, "Invalid shape for dummy buffer creation.");
             return null;
         }
         try {
             long numElements = 1;
             for (int dim : shape) {
                 if (dim <= 0) throw new IllegalArgumentException("Dimension must be positive: " + dim);
                 numElements *= dim;
             }

             int elementSize = getDataTypeSizeBytes(dataType);
             if (elementSize <= 0) {
                  Log.e(TAG, "Unsupported data type for dummy buffer: " + dataType);
                  return null;
             }

             long totalBytes = numElements * elementSize;
             if (totalBytes > Integer.MAX_VALUE) {
                  Log.e(TAG, "Required buffer size exceeds Integer.MAX_VALUE: " + totalBytes);
                  return null; // ByteBuffer uses int size
             }


             ByteBuffer buffer = ByteBuffer.allocateDirect((int)totalBytes);
             buffer.order(ByteOrder.nativeOrder());

             // Optional: Fill with dummy data (e.g., zeros) - depends on model tolerance
             // For FLOAT32:
             // FloatBuffer floatBuffer = buffer.asFloatBuffer();
             // for (int i = 0; i < numElements; i++) { floatBuffer.put(0.0f); }

             Log.d(TAG, "Created dummy input buffer of size " + totalBytes + " bytes for shape " + Arrays.toString(shape));
             return buffer;

         } catch (IllegalArgumentException e) {
              Log.e(TAG, "Error creating dummy input buffer: " + e.getMessage());
              return null;
         } catch (OutOfMemoryError e) {
             Log.e(TAG, "OutOfMemoryError creating dummy input buffer of size. Model input might be too large.");
             return null;
         }
     }

    // Helper function to get size of TFLite DataType in bytes
     private int getDataTypeSizeBytes(DataType dataType) {
         if (dataType == DataType.FLOAT32) return 4;
         if (dataType == DataType.INT32) return 4;
         if (dataType == DataType.UINT8) return 1;
         if (dataType == DataType.INT64) return 8;
         // if (dataType == DataType.STRING) return -1; // Variable size, needs special handling
         if (dataType == DataType.BOOL) return 1;
         if (dataType == DataType.INT16) return 2;
        //  if (dataType == DataType.FLOAT16) return 2;
         if (dataType == DataType.INT8) return 1;
         // Add other types if needed
         // throw new IllegalArgumentException("Unsupported data type: " + dataType);
         Log.w(TAG, "Unsupported or unknown data type encountered: " + dataType);
         return -1; // Indicate error or unsupported type
     }

    // Helper method to update UI components from any thread
    private void updateUI(final String result, final String status) {
         mainHandler.post(() -> {
             textViewResult.setText(result);
             textViewStatus.setText(status);
             // Reset buttons only when processing is fully complete (Idle or Error status)
              if ("Status: Idle".equals(status) || status.startsWith("Status: Error")) {
                 resetRecordingStateUI(); // Reset button states on UI thread
             }
         });
    }

    // Resets button states, typically called from updateUI on Main thread
    private void resetRecordingStateUI() {
        buttonStartRecord.setEnabled(true);
        buttonStopRecord.setEnabled(false);
        if (textViewStatus.getText().toString().startsWith("Status: Recording")) {
             // If status didn't update correctly before this, force it back to Idle
              textViewStatus.setText("Status: Idle");
        }
         Log.d(TAG,"UI state reset.");
    }

    // Only resets internal state variables, call from appropriate thread
     private synchronized void resetRecordingState() {
          releaseAudioRecord(); // Ensure recorder is released if still held
          isRecording = false;
          Log.d(TAG,"Internal recording state reset.");
          // UI update should happen separately via updateUI/mainHandler
     }


    // Synchronized method to safely release the AudioRecord instance
    private synchronized void releaseAudioRecord() {
        if (audioRecord != null) {
            try {
                 // Check state before stopping, some states might throw exception on stop()
                 if (audioRecord.getState() == AudioRecord.STATE_INITIALIZED ||
                     audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                     // Only stop if it was actually recording or initialized and possibly started
                     if (audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                         audioRecord.stop();
                         Log.d(TAG, "AudioRecord stopped in releaseAudioRecord.");
                     }
                 }
                audioRecord.release();
                Log.i(TAG, "AudioRecord released.");
            } catch (IllegalStateException e) {
                 Log.e(TAG, "IllegalStateException while stopping/releasing AudioRecord: " + e.getMessage());
            } catch (Exception e) { // Catch any other potential issues during release
                 Log.e(TAG, "Exception releasing AudioRecord: " + e.getMessage());
            } finally {
                 audioRecord = null; // Ensure it's null after attempting release
            }
        }
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permission Granted! Click Start again.", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Permission Denied.", Toast.LENGTH_SHORT).show();
                // Handle the case where the user denies the permission. Maybe disable functionality.
            }
        }
        // Add other permission results handling here if needed
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i(TAG,"onDestroy called.");
        // Shutdown executor service gracefully
        executorService.shutdown();
        // Release resources
        releaseAudioRecord(); // Ensure AudioRecord is released

        if (whisperHelper != null) {
            whisperHelper.close(); // Close the TFLite interpreter
        }
         // Remove callbacks from handler to prevent memory leaks
         mainHandler.removeCallbacksAndMessages(null);
         Log.i(TAG,"Activity destroyed, resources released.");
    }
}