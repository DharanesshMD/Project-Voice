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
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.DataType;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.zip.GZIPInputStream; // Import GZIPInputStream

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private final int outputTensorIndex = 0; // Assuming output is at index 0

    private WhisperHelper whisperHelper;
    private TextView textViewStatus;
    private TextView textViewResult;
    private Button buttonStartRecord;
    private Button buttonStopRecord;

    // --- Audio Recording Setup ---
    private AudioRecord audioRecord;
    private boolean isRecording = false;
    private Thread recordingThread; // Thread for reading audio data
    private ByteArrayOutputStream recordingBuffer; // To store recorded audio bytes

    private final int sampleRate = 16000; // Whisper models typically expect 16kHz
    private final int channelConfig = AudioFormat.CHANNEL_IN_MONO;
    private final int audioFormat = AudioFormat.ENCODING_PCM_16BIT; // 16-bit PCM
    private int bufferSizeInBytes = AudioRecord.ERROR_BAD_VALUE;

    // ExecutorService for background tasks (inference)
    private final ExecutorService inferenceExecutorService = Executors.newSingleThreadExecutor();
    // Handler to post results back to the main thread
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    // --- Vocabulary and Decoding ---
    private Map<Integer, String> idToTokenMap = null; // To store loaded vocabulary
    // !!! Choose the correct vocabulary file for your model !!!
    private static final String VOCAB_FILENAME = "filters_vocab_en.bin"; // Or "filters_vocab_multilingual.bin"

    // Define common special tokens to filter out (add more if needed based on vocab file)
    private static final Set<String> SPECIAL_TOKENS = new HashSet<>(Arrays.asList(
            "<|startoftranscript|>",
            "<|endoftext|>",
            "<|notimestamps|>",
            "<|transcribe|>",
            "<|translate|>",
            "<|nocaptions|>", // Common in vocab
            "<|nospeech|>",   // Common in vocab
            // Add language tokens if using multilingual model, e.g., "<|en|>", "<|fr|>" etc.
            // Add timestamp tokens if you want to ignore them "<|0.00|>", etc. (usually handled differently)
            "" // Empty token if present
    ));


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textViewStatus = findViewById(R.id.textViewStatus);
        textViewResult = findViewById(R.id.textViewResult);
        buttonStartRecord = findViewById(R.id.buttonStartRecord);
        buttonStopRecord = findViewById(R.id.buttonStopRecord);

        // Disable button initially
        buttonStartRecord.setEnabled(false);

        // Initialize Whisper Helper and Load Vocabulary
        try {
            whisperHelper = new WhisperHelper(this);
            Log.i(TAG, "WhisperHelper initialized successfully.");

            // --- Load Vocabulary ---
            loadVocabulary(); // Try to load the vocabulary
            // --- End Load Vocabulary ---

            // Enable start button only if *both* model and vocab loaded successfully
            if (whisperHelper != null && idToTokenMap != null && !idToTokenMap.isEmpty()) {
                textViewStatus.setText("Status: Ready");
                buttonStartRecord.setEnabled(true);
            } else {
                // If vocab failed, keep status as error
                textViewStatus.setText("Status: Error loading model/vocab");
                Toast.makeText(this, "Model or vocabulary failed to load. Check logs.", Toast.LENGTH_LONG).show();
                buttonStartRecord.setEnabled(false); // Ensure button remains disabled
            }

        } catch (Exception e) { // Catch errors from WhisperHelper constructor OR loadVocabulary
            textViewStatus.setText("Status: Error loading model/vocab");
            Log.e(TAG, "Error during initialization (Model or Vocab)", e); // Log the exception 'e'
            Toast.makeText(this, "Initialization failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
            buttonStartRecord.setEnabled(false);
            buttonStopRecord.setEnabled(false);
        }

        buttonStartRecord.setOnClickListener(v -> startRecording());
        buttonStopRecord.setOnClickListener(v -> stopRecordingAndTranscribe());

        // Calculate buffer size
        bufferSizeInBytes = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat);
        if (bufferSizeInBytes == AudioRecord.ERROR || bufferSizeInBytes == AudioRecord.ERROR_BAD_VALUE) {
            Log.w(TAG, "Min buffer size calculation failed. Using default (1 sec).");
            bufferSizeInBytes = sampleRate * 2 * 1; // sampleRate * bytes_per_sample * seconds
        }
        Log.d(TAG,"AudioRecord minimum buffer size: " + bufferSizeInBytes + " bytes");
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
        // Re-check dependencies just in case state changes somehow, though unlikely here
        if (whisperHelper == null || idToTokenMap == null || idToTokenMap.isEmpty()) {
            Toast.makeText(this, "Model or vocabulary not ready.", Toast.LENGTH_SHORT).show();
            Log.w(TAG, "Start recording called but model/vocab not ready.");
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
            if (bufferSizeInBytes <= 0) {
                Log.e(TAG, "Invalid bufferSizeInBytes: " + bufferSizeInBytes);
                Toast.makeText(this, "Audio recording configuration error.", Toast.LENGTH_SHORT).show();
                return;
            }

            int recordingBufferSize = bufferSizeInBytes * 2;
            audioRecord = new AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    channelConfig,
                    audioFormat,
                    recordingBufferSize
            );

            if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord initialization failed. State: " + audioRecord.getState());
                Toast.makeText(this, "Audio recording failed to initialize.", Toast.LENGTH_SHORT).show();
                releaseAudioRecord();
                return;
            }

            recordingBuffer = new ByteArrayOutputStream();
            isRecording = true;

            recordingThread = new Thread(() -> {
                byte[] audioDataBuffer = new byte[bufferSizeInBytes];
                Log.d(TAG, "Recording thread started. Reading in chunks of " + bufferSizeInBytes + " bytes.");
                while (isRecording && audioRecord != null && audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                    int bytesRead = audioRecord.read(audioDataBuffer, 0, audioDataBuffer.length);
                    if (bytesRead > 0) {
                        recordingBuffer.write(audioDataBuffer, 0, bytesRead);
                    } else if (bytesRead < 0) {
                        Log.e(TAG, "Error reading audio data: " + bytesRead);
                        if (bytesRead == AudioRecord.ERROR_INVALID_OPERATION || bytesRead == AudioRecord.ERROR_BAD_VALUE) {
                            Log.e(TAG, "Stopping recording thread due to read error: " + bytesRead);
                            // Consider signaling UI thread about the error
                            mainHandler.post(() -> updateUI("Error reading audio", "Status: Error"));
                            break;
                        }
                    }
                }
                Log.d(TAG,"Recording thread finished.");
            }, "AudioRecorder Thread");


            audioRecord.startRecording();
            recordingThread.start();

            updateUI(null, "Status: Recording...");
            Log.i(TAG, "Recording started.");

        } catch (SecurityException | IllegalStateException | IllegalArgumentException e) {
            Log.e(TAG, "Exception starting recording: " + e.getMessage(), e);
            Toast.makeText(this, "Failed to start recording: "+e.getMessage(), Toast.LENGTH_SHORT).show();
            resetRecordingState();
        } catch (Exception e) {
            Log.e(TAG, "Unexpected error starting recording: " + e.getMessage(), e);
            Toast.makeText(this, "An error occurred.", Toast.LENGTH_SHORT).show();
            resetRecordingState();
        }
    }

    private void stopRecordingAndTranscribe() {
        if (!isRecording) {
            Log.w(TAG, "Stop called but not in recording state.");
            return;
        }

        updateUI(null, "Status: Stopping and Processing...");

        isRecording = false;
        if (recordingThread != null) {
            try {
                recordingThread.join(500);
                if (recordingThread.isAlive()) {
                    Log.w(TAG, "Recording thread did not finish within timeout. Interrupting.");
                    recordingThread.interrupt();
                }
            } catch (InterruptedException e) {
                Log.e(TAG, "Interrupted while waiting for recording thread", e);
                Thread.currentThread().interrupt();
            }
            recordingThread = null;
        }

        releaseAudioRecord();

        final byte[] recordedAudioBytes = (recordingBuffer != null) ? recordingBuffer.toByteArray() : null;
        try {
            if (recordingBuffer != null) recordingBuffer.close();
        } catch (IOException e) {
            Log.e(TAG, "Error closing recording buffer stream", e);
        }
        recordingBuffer = null;

        if (recordedAudioBytes == null || recordedAudioBytes.length == 0) {
            Log.w(TAG,"No audio data captured.");
            updateUI("No audio data captured.", "Status: Idle");
            return;
        }

        Log.i(TAG, "Recorded " + recordedAudioBytes.length + " bytes of audio (" + (recordedAudioBytes.length / (float)(sampleRate*2)) + " seconds).");

        inferenceExecutorService.submit(() -> {
            if (whisperHelper == null) {
                Log.e(TAG, "WhisperHelper is null, cannot preprocess or transcribe.");
                updateUI("Error: Model helper not available.", "Status: Error");
                return;
            }

            // Get input details from WhisperHelper
            DataType inputDataType = whisperHelper.getInputDataType();
            int[] inputShape = whisperHelper.getInputShape();
            if (inputDataType == null || inputShape == null) {
                Log.e(TAG, "Could not get model input details from WhisperHelper.");
                updateUI("Error: Failed to get model input info.", "Status: Error");
                return;
            }
            Log.d(TAG, "Model Expected Input Shape: " + Arrays.toString(inputShape) + ", Type: " + inputDataType);

            // Process audio data into the format expected by the model
            ByteBuffer inputBuffer = preprocessAudio(recordedAudioBytes);

            // Check if preprocessing was successful
            if (inputBuffer == null) {
                Log.e(TAG, "Audio preprocessing failed");
                updateUI("Error: Audio preprocessing failed", "Status: Error");
                return;
            }

            Map<Integer, Object> transcriptionOutput = null;
            String resultText = "Processing failed";

            Log.d(TAG, "Calling whisperHelper.transcribe...");
            long startTime = System.currentTimeMillis();
            transcriptionOutput = whisperHelper.transcribe(inputBuffer);
            long endTime = System.currentTimeMillis();
            Log.d(TAG, "Transcription finished in " + (endTime-startTime) + " ms.");

            // Decode Output
            if (transcriptionOutput != null && transcriptionOutput.containsKey(outputTensorIndex)) {
                Object rawOutput = transcriptionOutput.get(outputTensorIndex);
                DataType outputDataType = whisperHelper.getOutputDataType();

                if (rawOutput instanceof ByteBuffer) {
                    ByteBuffer outputByteBuffer = (ByteBuffer) rawOutput;
                    Log.d(TAG, "Decoding output buffer. Size: " + outputByteBuffer.remaining() + " bytes. Type: " + outputDataType);
                    resultText = decodeOutputBuffer(outputByteBuffer, outputDataType);
                } else {
                    resultText = "Transcription output format unexpected: " + (rawOutput != null ? rawOutput.getClass().getName() : "null");
                    Log.e(TAG, resultText);
                }
            } else {
                resultText = "Transcription failed or output tensor not found.";
                Log.e(TAG, resultText);
            }

            // Update UI on the main thread
            updateUI(resultText, "Status: Idle");
            Log.i(TAG,"Background processing complete.");
        });
    }


    // --- Function to Load Vocabulary ---
    // Attempts GZip first, then plain text.
    // !!! Assumes text format is "ID<space>TOKEN" !!!
    // !!! MUST BE MODIFIED IF YOUR .bin FORMAT IS DIFFERENT !!!
    private void loadVocabulary() {
        idToTokenMap = new HashMap<>();
        InputStream inputStream = null;
        BufferedReader reader = null;
        boolean loaded = false;

        // --- Attempt 1: Try reading as GZipped Text ---
        try {
            Log.i(TAG, "Attempting to load vocabulary as GZip from: " + VOCAB_FILENAME);
            inputStream = getAssets().open(VOCAB_FILENAME);
            // Wrap the asset stream in a GZIPInputStream
            InputStream gzipStream = new GZIPInputStream(inputStream);
            reader = new BufferedReader(new InputStreamReader(gzipStream)); // Read the decompressed stream
            parseVocabulary(reader); // Call helper to parse lines
            Log.i(TAG, "Vocabulary loaded successfully (GZip mode). Size: " + idToTokenMap.size());
            loaded = true;

        } catch (IOException gzipException) {
            Log.w(TAG, "Failed to load vocabulary as GZip (" + gzipException.getMessage() + "). Trying as plain text...");
            // Close potentially opened streams from failed attempt
            try { if (reader != null) reader.close(); } catch (IOException e) { /* ignore */ }
            try { if (inputStream != null) inputStream.close(); } catch (IOException e) { /* ignore */ }
            reader = null;
            inputStream = null;
            idToTokenMap.clear(); // Clear map before trying plain text

            // --- Attempt 2: Try reading as Plain Text ---
            try {
                Log.i(TAG, "Attempting to load vocabulary as plain text from: " + VOCAB_FILENAME);
                inputStream = getAssets().open(VOCAB_FILENAME);
                reader = new BufferedReader(new InputStreamReader(inputStream));
                parseVocabulary(reader); // Call helper to parse lines
                Log.i(TAG, "Vocabulary loaded successfully (Plain Text mode). Size: " + idToTokenMap.size());
                loaded = true;

            } catch (IOException textException) {
                Log.e(TAG, "Failed to load vocabulary as plain text as well.", textException);
                Toast.makeText(this, "Failed to load vocabulary.", Toast.LENGTH_SHORT).show();
                idToTokenMap = null; // Ensure map is null on failure
                loaded = false; // Ensure loaded flag is false
            }
        } finally {
            try {
                if (reader != null) reader.close();
                if (inputStream != null) inputStream.close();
            } catch (IOException e) {
                Log.e(TAG, "Error closing vocabulary stream", e);
            }
        }

        // Final check if loading failed completely or resulted in empty map
        if (!loaded || (idToTokenMap != null && idToTokenMap.isEmpty()) ) {
            Log.e(TAG, "Vocabulary map is null or empty after attempting load! Check file format and content: " + VOCAB_FILENAME);
            if (loaded) { // Only show toast if parsing succeeded but result was empty
                Toast.makeText(this, "Vocabulary loaded but empty. Check format?", Toast.LENGTH_LONG).show();
            }
            idToTokenMap = null; // Ensure it's null
        }
    }

    // Helper function to parse lines from the vocabulary reader
    private void parseVocabulary(BufferedReader reader) throws IOException, NumberFormatException {
        String line;
        int count = 0;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.trim().split("\\s+", 2); // Split on first whitespace block
            if (parts.length == 2) {
                int id = Integer.parseInt(parts[0]); // Can throw NumberFormatException
                String token = parts[1];
                idToTokenMap.put(id, token);
                count++;
            } else {
                Log.w(TAG, "Skipping malformed vocab line (parts != 2): " + line);
            }
        }
        Log.d(TAG, "Parsed " + count + " vocabulary entries.");
    }


    // --- Function to Decode Output Buffer ---
    private String decodeOutputBuffer(ByteBuffer outputBuffer, DataType outputDataType) {
        if (outputBuffer == null || idToTokenMap == null || idToTokenMap.isEmpty()) {
            Log.e(TAG, "Cannot decode: Output buffer is null or vocabulary not loaded.");
            return "Decoding Error: Vocab not loaded or null buffer.";
        }
        if (outputDataType == null) {
            outputDataType = whisperHelper != null ? whisperHelper.getOutputDataType() : null;
            if (outputDataType == null) {
                Log.e(TAG, "Cannot decode: Output data type is unknown.");
                return "Decoding Error: Unknown output data type.";
            }
        }

        outputBuffer.rewind();

        List<Integer> tokenIds = new ArrayList<>();
        int bufferLimit = outputBuffer.limit();
        Log.d(TAG, "Decoding output buffer. Type: " + outputDataType + ", Size (bytes): " + bufferLimit);

        try {
            switch (outputDataType) {
                case INT32:
                    if (bufferLimit % 4 != 0) { /* ... error handling ... */ return "Decoding Error: Invalid INT32 buffer size."; }
                    IntBuffer intBuffer = outputBuffer.asIntBuffer();
                    int numInts = intBuffer.remaining();
                    Log.d(TAG, "Decoding INT32 buffer with " + numInts + " elements.");
                    for (int i = 0; i < numInts; i++) tokenIds.add(intBuffer.get(i));
                    break;

                case FLOAT32:
                    if (bufferLimit % 4 != 0) { /* ... error handling ... */ return "Decoding Error: Invalid FLOAT32 buffer size."; }
                    FloatBuffer floatBuffer = outputBuffer.asFloatBuffer();
                    int numFloats = floatBuffer.remaining();
                    int vocabSize = idToTokenMap.size();
                    if (vocabSize <= 0) { /* ... error handling ... */ return "Decoding Error: Invalid vocab size."; }
                    if (numFloats % vocabSize != 0) { /* ... error handling ... */ }
                    int sequenceLength = numFloats / vocabSize;
                    Log.d(TAG, "Decoding FLOAT32 buffer. Float count: " + numFloats + ", SeqLen(est): " + sequenceLength + ", VocabSize: "+ vocabSize);
                    for (int i = 0; i < sequenceLength; i++) {
                        int startIndex = i * vocabSize;
                        if (startIndex + vocabSize > numFloats) break;
                        int bestTokenId = findMaxIndex(floatBuffer, startIndex, vocabSize);
                        if (bestTokenId != -1) tokenIds.add(bestTokenId);
                    }
                    break;

                case INT64:
                    if (bufferLimit % 8 != 0) { /* ... error handling ... */ return "Decoding Error: Invalid INT64 buffer size."; }
                    java.nio.LongBuffer longBuffer = outputBuffer.asLongBuffer();
                    int numLongs = longBuffer.remaining();
                    Log.d(TAG, "Decoding INT64 buffer with " + numLongs + " elements.");
                    for (int i = 0; i < numLongs; i++) {
                        long id = longBuffer.get(i);
                        if (id > Integer.MAX_VALUE || id < Integer.MIN_VALUE) { Log.w(TAG, "Warning: INT64 token ID " + id + " outside int range."); tokenIds.add(-1); }
                        else { tokenIds.add((int) id); }
                    }
                    break;

                default:
                    Log.e(TAG, "Unsupported output data type for decoding: " + outputDataType);
                    return "Decoding Error: Unsupported output type " + outputDataType;
            }
        } catch (Exception e) {
            Log.e(TAG, "Error reading data from output buffer: " + e.getMessage(), e);
            return "Decoding Error: Buffer read failed.";
        }

        StringBuilder transcript = new StringBuilder();
        boolean firstToken = true;
        for (int tokenId : tokenIds) {
            if (tokenId == -1) continue;
            String token = idToTokenMap.getOrDefault(tokenId, "[UNK:" + tokenId + "]");

            if (token.equals("<|endoftext|>")) { Log.d(TAG,"End-of-text token encountered."); break; }
            if (SPECIAL_TOKENS.contains(token)) { Log.d(TAG,"Skipping special token: "+token); continue; }

            if (token.startsWith("Ġ")) {
                if (!firstToken && transcript.length() > 0 && transcript.charAt(transcript.length() - 1) != ' ') {
                    transcript.append(" ");
                }
                transcript.append(token.substring(1));
            } else {
                transcript.append(token);
            }
            firstToken = false;
        }

        Log.d(TAG, "Raw decoded IDs: " + tokenIds.toString());
        String finalTranscript = transcript.toString().trim();
        Log.i(TAG, "Final Transcript: " + finalTranscript);
        return finalTranscript;
    }

    // Helper function to find the index of the maximum value in a section of a FloatBuffer
    // Helper function to find the index of the maximum value in a section of a FloatBuffer
    private int findMaxIndex(FloatBuffer buffer, int startIndex, int length) {
        if (buffer == null) {
            Log.e(TAG, "Buffer is null in findMaxIndex");
            return -1;
        }

        if (startIndex < 0 || length <= 0 || startIndex + length > buffer.limit()) {
            Log.e(TAG, "Invalid range for findMaxIndex. Start: " + startIndex +
                    ", Length: " + length + ", Buffer limit: " + buffer.limit());
            return -1;
        }

        float maxVal = Float.NEGATIVE_INFINITY;
        int maxIdx = -1;

        for (int i = 0; i < length; i++) {
            float currentVal;
            try {
                currentVal = buffer.get(startIndex + i);
            } catch (IndexOutOfBoundsException e) {
                Log.e(TAG, "Index out of bounds in findMaxIndex: " + (startIndex + i), e);
                break;
            }

            // Skip NaN values
            if (Float.isNaN(currentVal)) {
                continue;
            }

            if (currentVal > maxVal) {
                maxVal = currentVal;
                maxIdx = i;
            }
        }

        if (maxIdx == -1) {
            Log.w(TAG, "No valid maximum found in buffer section");
        }

        return maxIdx;
    }


    // Helper to create a dummy buffer - REPLACE with actual audio preprocessing
    private ByteBuffer createDummyInputBuffer(int[] shape, DataType dataType) {
        // --- THIS IS A PLACEHOLDER - DO NOT USE FOR REAL TRANSCRIPTION ---
        if (shape == null || shape.length == 0) { /* ... error handling ... */ return null; }
        try {
            long numElements = 1;
            for (int i = 0; i < shape.length; i++) {
                int dim = shape[i];
                int actualDim = dim;
                if (dim <= 0) {
                    if (i == shape.length - 1) { // Often last dim is time/frames
                        actualDim = 3000; // Default Whisper frame count for 30s
                        Log.w(TAG, "Shape dim " + i + " is non-positive (" + dim + "), using fixed size " + actualDim + " for dummy buffer (assuming time dimension).");
                    } else { actualDim = 1; Log.w(TAG, "Shape dim " + i + " is non-positive (" + dim + "), using 1 for dummy buffer."); }
                }
                if (Long.MAX_VALUE / actualDim < numElements) throw new IllegalArgumentException("Dummy buffer size overflow.");
                numElements *= actualDim;
            }
            if (numElements == 0) { /* ... error handling ... */ return null; }
            int elementSize = getDataTypeSizeBytes(dataType);
            if (elementSize <= 0) { /* ... error handling ... */ return null; }
            long totalBytesLong = numElements * elementSize;
            if (totalBytesLong > Integer.MAX_VALUE) { /* ... error handling ... */ return null; }
            int totalBytes = (int) totalBytesLong;
            ByteBuffer buffer = ByteBuffer.allocateDirect(totalBytes);
            buffer.order(ByteOrder.nativeOrder());
            // Filling with zeros is usually okay for dummy run
            byte[] zeros = new byte[totalBytes];
            buffer.put(zeros);
            buffer.rewind();
            Log.d(TAG, "Created DUMMY input buffer of size " + totalBytes + " bytes for shape " + Arrays.toString(shape) + ", Type: " + dataType);
            return buffer;
        } catch (IllegalArgumentException | OutOfMemoryError e) {
            Log.e(TAG, "Error creating dummy input buffer: " + e.getMessage());
            return null;
        }
    }

    // Helper function to get size of TFLite DataType in bytes
    private int getDataTypeSizeBytes(DataType dataType) {
        switch (dataType) {
            case FLOAT32: return 4; case INT32: return 4; case UINT8: return 1;
            case INT64: return 8; case BOOL: return 1; case INT16: return 2;
            case INT8: return 1;
            default: Log.w(TAG, "Unsupported data type size calc: " + dataType); return -1;
        }
    }

    // Helper method to update UI components from any thread
    private void updateUI(final String result, final String status) {
        mainHandler.post(() -> {
            if (result != null) textViewResult.setText(result);
            if (status != null) textViewStatus.setText(status);

            boolean modelReady = whisperHelper != null && idToTokenMap != null && !idToTokenMap.isEmpty();
            switch (status != null ? status : "") {
                case "Status: Recording...":
                    buttonStartRecord.setEnabled(false);
                    buttonStopRecord.setEnabled(true);
                    textViewResult.setText("");
                    break;
                case "Status: Stopping and Processing...":
                    buttonStartRecord.setEnabled(false);
                    buttonStopRecord.setEnabled(false);
                    break;
                case "Status: Ready": // Explicit ready state
                    buttonStartRecord.setEnabled(true);
                    buttonStopRecord.setEnabled(false);
                    break;
                case "Status: Idle":
                case "Status: Error":
                default:
                    buttonStartRecord.setEnabled(modelReady);
                    buttonStopRecord.setEnabled(false);
                    if (status != null && !status.startsWith("Status: Error") && modelReady && !isRecording) {
                        textViewStatus.setText("Status: Ready"); // Reset to ready if idle and model is ok
                    } else if (status != null && !status.startsWith("Status: Error") && !modelReady) {
                        textViewStatus.setText("Status: Error loading model/vocab"); // Reflect underlying issue
                    }
                    break;
            }
        });
    }

    // Only resets internal state variables, called from appropriate thread
    private synchronized void resetRecordingState() {
        isRecording = false;
        if (recordingThread != null) {
            if (recordingThread.isAlive()) { Log.w(TAG,"Interrupting recording thread during reset."); recordingThread.interrupt(); }
            recordingThread = null;
        }
        releaseAudioRecord();
        if (recordingBuffer != null) { try { recordingBuffer.close(); } catch (IOException e) { /* Ignored */ } recordingBuffer = null; }
        Log.d(TAG,"Internal recording state reset.");
        updateUI(null, "Status: Idle"); // Update UI to idle/ready state
    }

    // Synchronized method to safely release the AudioRecord instance
    private synchronized void releaseAudioRecord() {
        if (audioRecord != null) {
            Log.d(TAG, "Attempting to release AudioRecord...");
            try {
                if (audioRecord.getState() == AudioRecord.STATE_INITIALIZED) {
                    if (audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                        try { audioRecord.stop(); Log.d(TAG, "AudioRecord stopped."); }
                        catch (IllegalStateException e) { Log.e(TAG, "IllegalStateException on stop(): " + e.getMessage()); }
                    }
                    audioRecord.release(); Log.i(TAG, "AudioRecord released.");
                } else if (audioRecord.getState() == AudioRecord.STATE_UNINITIALIZED) {
                    Log.w(TAG,"AudioRecord was uninitialized, attempting release anyway.");
                    audioRecord.release(); Log.i(TAG, "AudioRecord released from uninitialized state.");
                } else { Log.w(TAG,"AudioRecord not initialized, skipping release. State: "+audioRecord.getState()); }
            } catch (Exception e) { Log.e(TAG, "Exception releasing AudioRecord: " + e.getMessage(), e); }
            finally { audioRecord = null; }
        }
    }

    private ByteBuffer preprocessAudio(byte[] recordedAudioBytes) {
        if (recordedAudioBytes == null || recordedAudioBytes.length == 0) {
            Log.e(TAG, "No audio data to preprocess");
            return null;
        }

        Log.d(TAG, "Preprocessing " + recordedAudioBytes.length + " bytes of audio data");

        // Get input shape and data type from WhisperHelper
        int[] inputShape = whisperHelper.getInputShape();
        DataType inputDataType = whisperHelper.getInputDataType();

        if (inputShape == null || inputDataType == null) {
            Log.e(TAG, "Invalid input shape or data type");
            return null;
        }

        Log.d(TAG, "Model expects input shape: " + Arrays.toString(inputShape) + ", type: " + inputDataType);

        try {
            // Convert PCM audio bytes to float array (assuming 16-bit PCM)
            short[] shorts = new short[recordedAudioBytes.length / 2];
            ByteBuffer.wrap(recordedAudioBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts);

            float[] floatAudio = new float[shorts.length];
            for (int i = 0; i < shorts.length; i++) {
                // Normalize to range [-1, 1]
                floatAudio[i] = shorts[i] / 32768.0f;
            }

            // Calculate mel spectrogram (simplified approach)
            int melFeatureCount = inputShape[inputShape.length - 1]; // Usually 80 for Whisper models
            int frameLength = 400; // 25ms at 16kHz
            int frameShift = 160; // 10ms at 16kHz
            int fftSize = 512; // Typically power of 2 >= frameLength

            // Calculate number of frames
            int numFrames = 1 + (floatAudio.length - frameLength) / frameShift;
            if (numFrames <= 0) {
                Log.e(TAG, "Audio too short for processing");
                return null;
            }

            Log.d(TAG, "Audio will produce " + numFrames + " frames with " + melFeatureCount + " mel features each");

            // Pre-allocate mel spectrogram array
            float[][] melSpectrogram = new float[numFrames][melFeatureCount];

            // Windows for FFT
            float[] hammingWindow = createHammingWindow(frameLength);

            // Process each frame
            for (int frameIndex = 0; frameIndex < numFrames; frameIndex++) {
                int startSample = frameIndex * frameShift;

                // Apply window and prepare for FFT
                float[] windowedFrame = new float[fftSize];
                for (int i = 0; i < frameLength && (startSample + i) < floatAudio.length; i++) {
                    windowedFrame[i] = floatAudio[startSample + i] * hammingWindow[i];
                }

                // Compute power spectrum (magnitude squared of FFT)
                float[] powerSpectrum = computePowerSpectrum(windowedFrame, fftSize);

                // Apply mel filterbank
                melSpectrogram[frameIndex] = applyMelFilterbank(powerSpectrum, melFeatureCount, sampleRate, fftSize);
            }

            // Take log of mel spectrogram
            for (int i = 0; i < melSpectrogram.length; i++) {
                for (int j = 0; j < melSpectrogram[i].length; j++) {
                    // Add small constant to avoid log(0)
                    melSpectrogram[i][j] = (float)Math.log(melSpectrogram[i][j] + 1e-10);
                }
            }

            // Normalize (simple version - center and scale)
            float mean = 0f, stddev = 0f;
            int totalElements = numFrames * melFeatureCount;

            // Calculate mean
            for (int i = 0; i < melSpectrogram.length; i++) {
                for (int j = 0; j < melSpectrogram[i].length; j++) {
                    mean += melSpectrogram[i][j];
                }
            }
            mean /= totalElements;

            // Calculate stddev
            for (int i = 0; i < melSpectrogram.length; i++) {
                for (int j = 0; j < melSpectrogram[i].length; j++) {
                    float diff = melSpectrogram[i][j] - mean;
                    stddev += diff * diff;
                }
            }
            stddev = (float)Math.sqrt(stddev / totalElements);

            // Normalize
            for (int i = 0; i < melSpectrogram.length; i++) {
                for (int j = 0; j < melSpectrogram[i].length; j++) {
                    melSpectrogram[i][j] = (melSpectrogram[i][j] - mean) / (stddev + 1e-5f);
                }
            }

            // Pad or truncate to expected model input shape
            int expectedFrames = inputShape[2]; // Assuming shape is [batch, channels, time, features]
            float[][] finalMel;

            if (numFrames > expectedFrames) {
                // Truncate
                Log.d(TAG, "Truncating " + numFrames + " frames to " + expectedFrames);
                finalMel = new float[expectedFrames][melFeatureCount];
                for (int i = 0; i < expectedFrames; i++) {
                    System.arraycopy(melSpectrogram[i], 0, finalMel[i], 0, melFeatureCount);
                }
            } else {
                // Pad with zeros
                Log.d(TAG, "Padding " + numFrames + " frames to " + expectedFrames);
                finalMel = new float[expectedFrames][melFeatureCount];
                for (int i = 0; i < numFrames; i++) {
                    System.arraycopy(melSpectrogram[i], 0, finalMel[i], 0, melFeatureCount);
                }
                // Rest of the array remains zeros (padding)
            }

            // Create ByteBuffer with the appropriate size and fill it
            int totalSize = 1 * 1 * expectedFrames * melFeatureCount * 4; // batch=1, channels=1, float=4 bytes
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(totalSize);
            inputBuffer.order(ByteOrder.nativeOrder());

            for (int i = 0; i < expectedFrames; i++) {
                for (int j = 0; j < melFeatureCount; j++) {
                    inputBuffer.putFloat(finalMel[i][j]);
                }
            }

            inputBuffer.rewind();
            Log.d(TAG, "Created preprocessed input buffer of size " + totalSize + " bytes");
            return inputBuffer;

        } catch (Exception e) {
            Log.e(TAG, "Error preprocessing audio: " + e.getMessage(), e);
            return null;
        }
    }

    // Helper methods for mel spectrogram calculation
    private float[] createHammingWindow(int length) {
        float[] window = new float[length];
        for (int i = 0; i < length; i++) {
            window[i] = 0.54f - 0.46f * (float)Math.cos(2 * Math.PI * i / (length - 1));
        }
        return window;
    }

    private float[] computePowerSpectrum(float[] frame, int fftSize) {
        // Simple FFT implementation for power spectrum
        // In a real app, you'd use a proper FFT library

        // For this simplified example, we'll use a dummy implementation
        // that approximates power spectrum shape
        int halfFFT = fftSize / 2 + 1;
        float[] powerSpec = new float[halfFFT];

        for (int i = 0; i < halfFFT; i++) {
            float freq = i * (sampleRate / (float)fftSize);
            float sumSquares = 0;

            // Approximate by summing nearby samples with weighting
            for (int j = 0; j < frame.length; j++) {
                float weight = (float)Math.exp(-0.5 * Math.pow((j - frame.length/2) / (frame.length/8.0), 2));
                sumSquares += frame[j] * frame[j] * weight;
            }

            // Apply some frequency shaping typical for speech
            float freqShape = (float)(4000 / (freq + 400) * Math.exp(-freq/4000));
            powerSpec[i] = sumSquares * freqShape;
        }

        return powerSpec;
    }

    private float[] applyMelFilterbank(float[] powerSpectrum, int numMelBins, int sampleRate, int fftSize) {
        float[] melEnergies = new float[numMelBins];

        // Mel scale conversion functions
        float minMel = 0;
        float maxMel = hzToMel(sampleRate / 2);
        float melStep = (maxMel - minMel) / (numMelBins + 1);

        // For each mel bin
        for (int bin = 0; bin < numMelBins; bin++) {
            float leftMel = minMel + bin * melStep;
            float centerMel = leftMel + melStep;
            float rightMel = centerMel + melStep;

            // Convert to Hz and then to FFT bin indices
            int leftBin = Math.round(melToHz(leftMel) * fftSize / sampleRate);
            int centerBin = Math.round(melToHz(centerMel) * fftSize / sampleRate);
            int rightBin = Math.round(melToHz(rightMel) * fftSize / sampleRate);

            // Ensure we're within bounds
            leftBin = Math.max(0, Math.min(leftBin, powerSpectrum.length - 1));
            centerBin = Math.max(0, Math.min(centerBin, powerSpectrum.length - 1));
            rightBin = Math.max(0, Math.min(rightBin, powerSpectrum.length - 1));

            // Apply triangular filter
            float melEnergy = 0;

            // Left side of triangle
            for (int i = leftBin; i < centerBin; i++) {
                float weight = (i - leftBin) / (float)(centerBin - leftBin);
                melEnergy += powerSpectrum[i] * weight;
            }

            // Right side of triangle
            for (int i = centerBin; i < rightBin; i++) {
                float weight = 1.0f - (i - centerBin) / (float)(rightBin - centerBin);
                melEnergy += powerSpectrum[i] * weight;
            }

            melEnergies[bin] = melEnergy;
        }

        return melEnergies;
    }

    // Convert Hz to Mel
    private float hzToMel(float hz) {
        return 2595 * (float)Math.log10(1 + hz/700);
    }

    // Convert Mel to Hz
    private float melToHz(float mel) {
        return 700 * ((float)Math.pow(10, mel/2595) - 1);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            boolean modelReady = whisperHelper != null && idToTokenMap != null && !idToTokenMap.isEmpty();
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permission Granted! Click Start again.", Toast.LENGTH_SHORT).show();
                buttonStartRecord.setEnabled(modelReady); // Enable only if model/vocab also ready
            } else {
                Toast.makeText(this, "Permission Denied. Cannot record audio.", Toast.LENGTH_SHORT).show();
                buttonStartRecord.setEnabled(false);
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i(TAG,"onDestroy called.");
        if (isRecording) { Log.w(TAG, "Activity destroyed while recording. Forcing stop/release."); resetRecordingState(); }
        inferenceExecutorService.shutdown();
        try { if (!inferenceExecutorService.awaitTermination(500, java.util.concurrent.TimeUnit.MILLISECONDS)) { inferenceExecutorService.shutdownNow(); } }
        catch (InterruptedException e) { inferenceExecutorService.shutdownNow(); Thread.currentThread().interrupt(); }
        if (whisperHelper != null) { whisperHelper.close(); whisperHelper = null; }
        mainHandler.removeCallbacksAndMessages(null);
        Log.i(TAG,"Activity destroyed, resources released.");
    }
} // End of MainActivity class