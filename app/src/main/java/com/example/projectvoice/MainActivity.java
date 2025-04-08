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

import org.jtransforms.fft.FloatFFT_1D; // Import JTransforms FFT
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
    // --- UPDATED: Use multilingual vocab for the default multilingual model ---
    private static final String VOCAB_FILENAME = "filters_vocab_multilingual.bin"; // Or "filters_vocab_en.bin" for english-only model

    // Define common special tokens to filter out (add more if needed based on vocab file)
    // --- UPDATED: Added more common special/language tokens ---
    private static final Set<String> SPECIAL_TOKENS = new HashSet<>(Arrays.asList(
            "<|startoftranscript|>",
            "<|endoftext|>",
            "<|notimestamps|>",
            "<|transcribe|>",
            "<|translate|>",
            "<|nocaptions|>",
            "<|nospeech|>",
            // Common language tokens for multilingual model
            "<|en|>", "<|zh|>", "<|de|>", "<|es|>", "<|ru|>", "<|ko|>", "<|fr|>", "<|ja|>",
            "<|pt|>", "<|tr|>", "<|pl|>", "<|ca|>", "<|nl|>", "<|ar|>", "<|sv|>", "<|it|>",
            "<|id|>", "<|hi|>", "<|fi|>", "<|vi|>", "<|he|>", "<|uk|>", "<|el|>", "<|ms|>",
            "<|cs|>", "<|ro|>", "<|da|>", "<|hu|>", "<|ta|>", "<|no|>", "<|th|>", "<|ur|>",
            "<|hr|>", "<|bg|>", "<|lt|>", "<|la|>", "<|mi|>", "<|ml|>", "<|cy|>", "<|sk|>",
            "<|te|>", "<|fa|>", "<|lv|>", "<|bn|>", "<|sr|>", "<|az|>", "<|sl|>", "<|kn|>",
            "<|et|>", "<|mk|>", "<|br|>", "<|eu|>", "<|is|>", "<|hy|>", "<|ne|>", "<|mn|>",
            "<|bs|>", "<|kk|>", "<|sq|>", "<|sw|>", "<|gl|>", "<|mr|>", "<|pa|>", "<|si|>",
            "<|km|>", "<|sn|>", "<|yo|>", "<|so|>", "<|af|>", "<|oc|>", "<|ka|>", "<|be|>",
            "<|tg|>", "<|sd|>", "<|gu|>", "<|am|>", "<|yi|>", "<|lo|>", "<|uz|>", "<|fo|>",
            "<|ht|>", "<|ps|>", "<|tk|>", "<|ny|>", "<|mg|>", "<|as|>", "<|tt|>", "<|haw|>",
            "<|ln|>", "<|ha|>", "<|ba|>", "<|jw|>", "<|su|>",
            // Timestamp tokens (optional, filter if not needed)
            // Example: "<|0.00|>", "<|0.02|>", ... up to "<|30.00|>"
            // Simpler to filter based on pattern later if needed, or add common ones here.
            "<|startoflm|>", // Some models use this
            "<|startofprev|>",
            "<|nospeechprob|>",
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
            // Try loading the default model (whisper-tiny.tflite - multilingual)
            whisperHelper = new WhisperHelper(this);
            Log.i(TAG, "WhisperHelper initialized successfully.");

            // --- Load Vocabulary ---
            loadVocabulary(); // Try to load the vocabulary

            // Enable start button only if *both* model and vocab loaded successfully
            if (whisperHelper != null && idToTokenMap != null && !idToTokenMap.isEmpty()) {
                textViewStatus.setText("Status: Ready");
                buttonStartRecord.setEnabled(true);
            } else {
                textViewStatus.setText("Status: Error loading model/vocab");
                Toast.makeText(this, "Model or vocabulary failed to load. Check logs.", Toast.LENGTH_LONG).show();
                buttonStartRecord.setEnabled(false); // Ensure button remains disabled
            }

        } catch (Exception e) { // Catch errors from WhisperHelper constructor OR loadVocabulary
            textViewStatus.setText("Status: Error loading model/vocab");
            Log.e(TAG, "Error during initialization (Model or Vocab)", e);
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

            int recordingBufferSize = bufferSizeInBytes * 2; // Use a larger buffer for recording
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
                android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO); // Request higher priority
                byte[] audioDataBuffer = new byte[bufferSizeInBytes]; // Read in smaller chunks
                Log.d(TAG, "Recording thread started. Reading in chunks of " + bufferSizeInBytes + " bytes.");
                while (isRecording && audioRecord != null && audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                    int bytesRead = audioRecord.read(audioDataBuffer, 0, audioDataBuffer.length);
                    if (bytesRead > 0) {
                        try {
                            recordingBuffer.write(audioDataBuffer, 0, bytesRead);
                        } catch (Exception e) {
                            Log.e(TAG, "Error writing to recording buffer", e);
                            // Maybe stop recording?
                        }
                    } else if (bytesRead < 0) {
                        Log.e(TAG, "Error reading audio data: " + bytesRead);
                        // Handle specific errors if needed (e.g., ERROR_INVALID_OPERATION, ERROR_BAD_VALUE)
                        if (bytesRead == AudioRecord.ERROR_INVALID_OPERATION || bytesRead == AudioRecord.ERROR_BAD_VALUE) {
                            Log.e(TAG, "Stopping recording thread due to read error: " + bytesRead);
                            mainHandler.post(() -> updateUI("Error reading audio", "Status: Error"));
                            break; // Exit loop on critical errors
                        }
                        // Other errors might be recoverable, maybe just log them?
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
        } catch (Exception e) { // Catch unexpected errors
            Log.e(TAG, "Unexpected error starting recording: " + e.getMessage(), e);
            Toast.makeText(this, "An unexpected error occurred.", Toast.LENGTH_SHORT).show();
            resetRecordingState();
        }
    }

    private void stopRecordingAndTranscribe() {
        if (!isRecording) {
            Log.w(TAG, "Stop called but not in recording state.");
            return;
        }

        updateUI(null, "Status: Stopping and Processing...");

        isRecording = false; // Signal thread to stop
        if (recordingThread != null) {
            try {
                recordingThread.join(500); // Wait briefly for thread to finish reading
                if (recordingThread.isAlive()) {
                    Log.w(TAG, "Recording thread did not finish within timeout. Interrupting.");
                    recordingThread.interrupt(); // Interrupt if needed
                }
            } catch (InterruptedException e) {
                Log.e(TAG, "Interrupted while waiting for recording thread", e);
                Thread.currentThread().interrupt();
            }
            recordingThread = null;
        }

        releaseAudioRecord(); // Stop and release hardware resources

        final byte[] recordedAudioBytes = (recordingBuffer != null) ? recordingBuffer.toByteArray() : null;
        try {
            if (recordingBuffer != null) recordingBuffer.close();
        } catch (IOException e) {
            Log.e(TAG, "Error closing recording buffer stream", e);
        }
        recordingBuffer = null; // Release memory

        if (recordedAudioBytes == null || recordedAudioBytes.length == 0) {
            Log.w(TAG,"No audio data captured.");
            updateUI("No audio data captured.", "Status: Idle");
            return;
        }

        Log.i(TAG, "Recorded " + recordedAudioBytes.length + " bytes of audio (" + (recordedAudioBytes.length / (float)(sampleRate*2)) + " seconds).");

        // --- Submit for Preprocessing & Inference ---
        inferenceExecutorService.submit(() -> {
            if (whisperHelper == null) {
                Log.e(TAG, "WhisperHelper is null, cannot preprocess or transcribe.");
                updateUI("Error: Model helper not available.", "Status: Error");
                return;
            }

            // Get input details from WhisperHelper
            DataType inputDataType = whisperHelper.getInputDataType();
            int[] inputShape = whisperHelper.getInputShape();
            if (inputDataType == null || inputShape == null || inputDataType != DataType.FLOAT32) { // Whisper expects FLOAT32 Mel Spectrogram
                Log.e(TAG, "Could not get model input details or type is not FLOAT32. Shape: " + Arrays.toString(inputShape) + ", Type: " + inputDataType);
                updateUI("Error: Failed to get model input info or wrong type.", "Status: Error");
                return;
            }
            Log.d(TAG, "Model Expected Input Shape: " + Arrays.toString(inputShape) + ", Type: " + inputDataType);

            // --- Preprocess Audio ---
            long preprocessStart = System.currentTimeMillis();
            ByteBuffer inputBuffer = preprocessAudio(recordedAudioBytes);
            long preprocessEnd = System.currentTimeMillis();
            Log.d(TAG, "Audio preprocessing took: " + (preprocessEnd - preprocessStart) + " ms");


            // Check if preprocessing was successful
            if (inputBuffer == null) {
                Log.e(TAG, "Audio preprocessing failed");
                updateUI("Error: Audio preprocessing failed", "Status: Error");
                return;
            }

            // --- Transcribe ---
            Map<Integer, Object> transcriptionOutput = null;
            String resultText = "Processing failed";

            Log.d(TAG, "Calling whisperHelper.transcribe...");
            long startTime = System.currentTimeMillis();
            transcriptionOutput = whisperHelper.transcribe(inputBuffer);
            long endTime = System.currentTimeMillis();
            Log.d(TAG, "Transcription finished in " + (endTime-startTime) + " ms.");

            // --- Decode Output ---
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


    // --- Function to Load Vocabulary --- (Unchanged from previous version)
    private void loadVocabulary() {
        idToTokenMap = new HashMap<>();
        InputStream inputStream = null;
        BufferedReader reader = null;
        boolean loaded = false;

        try {
            Log.i(TAG, "Attempting to load vocabulary as GZip from: " + VOCAB_FILENAME);
            inputStream = getAssets().open(VOCAB_FILENAME);
            InputStream gzipStream = new GZIPInputStream(inputStream);
            reader = new BufferedReader(new InputStreamReader(gzipStream));
            parseVocabulary(reader);
            Log.i(TAG, "Vocabulary loaded successfully (GZip mode). Size: " + idToTokenMap.size());
            loaded = true;

        } catch (IOException gzipException) {
            Log.w(TAG, "Failed to load vocabulary as GZip (" + gzipException.getMessage() + "). Trying as plain text...");
            try { if (reader != null) reader.close(); } catch (IOException e) { /* ignore */ }
            try { if (inputStream != null) inputStream.close(); } catch (IOException e) { /* ignore */ }
            reader = null; inputStream = null; idToTokenMap.clear();

            try {
                Log.i(TAG, "Attempting to load vocabulary as plain text from: " + VOCAB_FILENAME);
                inputStream = getAssets().open(VOCAB_FILENAME);
                reader = new BufferedReader(new InputStreamReader(inputStream));
                parseVocabulary(reader);
                Log.i(TAG, "Vocabulary loaded successfully (Plain Text mode). Size: " + idToTokenMap.size());
                loaded = true;

            } catch (IOException textException) {
                Log.e(TAG, "Failed to load vocabulary as plain text as well.", textException);
                Toast.makeText(this, "Failed to load vocabulary.", Toast.LENGTH_SHORT).show();
                idToTokenMap = null; loaded = false;
            }
        } finally {
            try { if (reader != null) reader.close(); } catch (IOException e) { Log.e(TAG, "Error closing vocab reader", e); }
            try { if (inputStream != null) inputStream.close(); } catch (IOException e) { Log.e(TAG, "Error closing vocab input stream", e); }
        }

        if (!loaded || (idToTokenMap != null && idToTokenMap.isEmpty()) ) {
            Log.e(TAG, "Vocabulary map is null or empty after attempting load! Check file format and content: " + VOCAB_FILENAME);
            if (loaded) { Toast.makeText(this, "Vocabulary loaded but empty. Check format?", Toast.LENGTH_LONG).show(); }
            idToTokenMap = null;
        }
    }

    // --- Helper function to parse lines from the vocabulary reader --- (Unchanged)
    private void parseVocabulary(BufferedReader reader) throws IOException, NumberFormatException {
        String line;
        int count = 0;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.trim().split("\\s+", 2); // Split on first whitespace block
            if (parts.length == 2) {
                try {
                    int id = Integer.parseInt(parts[0]); // Can throw NumberFormatException
                    String token = parts[1];
                    idToTokenMap.put(id, token);
                    count++;
                } catch (NumberFormatException e) {
                    Log.w(TAG, "Skipping malformed vocab line (ID not integer): " + line);
                }
            } else {
                Log.w(TAG, "Skipping malformed vocab line (parts != 2): " + line);
            }
        }
        Log.d(TAG, "Parsed " + count + " vocabulary entries.");
    }


    // --- Function to Decode Output Buffer --- (Minor logging improvements)
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

        outputBuffer.order(ByteOrder.nativeOrder()).rewind(); // Ensure native order and rewind

        List<Integer> tokenIds = new ArrayList<>();
        int bufferLimit = outputBuffer.limit();
        Log.d(TAG, "Decoding output buffer. Type: " + outputDataType + ", Size (bytes): " + bufferLimit);

        try {
            switch (outputDataType) {
                case INT32:
                    if (bufferLimit % 4 != 0) { Log.e(TAG,"INT32 buffer size (" + bufferLimit + ") not divisible by 4"); return "Decoding Error: Invalid INT32 buffer size."; }
                    IntBuffer intBuffer = outputBuffer.asIntBuffer();
                    int numInts = intBuffer.remaining();
                    Log.d(TAG, "Decoding INT32 buffer with " + numInts + " elements.");
                    for (int i = 0; i < numInts; i++) tokenIds.add(intBuffer.get(i));
                    break;

                case FLOAT32:
                    // This case assumes output is logits [SeqLen, VocabSize] or [Batch, SeqLen, VocabSize]
                    // We perform greedy decoding here.
                    if (bufferLimit % 4 != 0) { Log.e(TAG,"FLOAT32 buffer size (" + bufferLimit + ") not divisible by 4"); return "Decoding Error: Invalid FLOAT32 buffer size."; }
                    FloatBuffer floatBuffer = outputBuffer.asFloatBuffer();
                    int numFloats = floatBuffer.remaining();
                    int vocabSize = idToTokenMap.size();
                    if (vocabSize <= 0) { Log.e(TAG,"Invalid vocab size: " + vocabSize); return "Decoding Error: Invalid vocab size."; }
                    if (numFloats % vocabSize != 0) { Log.w(TAG, "Warning: FLOAT32 buffer size (" + numFloats + ") not divisible by vocab size (" + vocabSize + "). Decoding might be partial."); }

                    int sequenceLength = numFloats / vocabSize;
                    Log.d(TAG, "Decoding FLOAT32 buffer (greedy). Float count: " + numFloats + ", SeqLen(est): " + sequenceLength + ", VocabSize: "+ vocabSize);

                    for (int i = 0; i < sequenceLength; i++) {
                        int startIndex = i * vocabSize;
                        if (startIndex + vocabSize > numFloats) { Log.w(TAG,"Buffer underflow at sequence step "+i); break; } // Prevent reading past buffer
                        int bestTokenId = findMaxIndex(floatBuffer, startIndex, vocabSize);
                        if (bestTokenId != -1) {
                            tokenIds.add(bestTokenId);
                            // Optional: Check for end token here to stop early
                            // String token = idToTokenMap.get(bestTokenId);
                            // if (token != null && token.equals("<|endoftext|>")) break;
                        } else {
                            Log.w(TAG,"No valid max found at sequence step "+i);
                        }
                    }
                    break;

                case INT64:
                    if (bufferLimit % 8 != 0) { Log.e(TAG,"INT64 buffer size (" + bufferLimit + ") not divisible by 8"); return "Decoding Error: Invalid INT64 buffer size."; }
                    java.nio.LongBuffer longBuffer = outputBuffer.asLongBuffer();
                    int numLongs = longBuffer.remaining();
                    Log.d(TAG, "Decoding INT64 buffer with " + numLongs + " elements.");
                    for (int i = 0; i < numLongs; i++) {
                        long id = longBuffer.get(i);
                        if (id > Integer.MAX_VALUE || id < Integer.MIN_VALUE) { Log.w(TAG, "Warning: INT64 token ID " + id + " outside int range. Skipping."); tokenIds.add(-1); } // Treat as invalid
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

        // --- Reconstruct Text ---
        StringBuilder transcript = new StringBuilder();
        boolean firstToken = true; // To handle leading spaces correctly
        int eotTokenId = -1; // Find the EOT token ID if needed for early stopping

        // Find EOT token ID once (optimization)
        for (Map.Entry<Integer, String> entry : idToTokenMap.entrySet()) {
            if ("<|endoftext|>".equals(entry.getValue())) {
                eotTokenId = entry.getKey();
                break;
            }
        }

        Log.d(TAG, "Raw decoded IDs: " + tokenIds.toString());

        for (int tokenId : tokenIds) {
            if (tokenId == -1) continue; // Skip invalid tokens (e.g., out-of-range INT64)
            if (tokenId == eotTokenId) { Log.d(TAG,"End-of-text token ID encountered."); break; } // Stop decoding at EOT

            String token = idToTokenMap.get(tokenId);

            if (token == null) {
                Log.w(TAG,"Token ID " + tokenId + " not found in vocabulary. Treating as UNK.");
                token = "[UNK:" + tokenId + "]";
            }

            // Filter special tokens (like <|en|>, <|notimestamps|>, etc.)
            // Also filter timestamp tokens if present (they look like <|0.00|>)
            if (SPECIAL_TOKENS.contains(token) || (token.startsWith("<|") && token.endsWith("|>") && token.length() > 4 && Character.isDigit(token.charAt(2)))) {
                Log.d(TAG,"Skipping special/timestamp token: "+token);
                continue;
            }

            // Handle BPE spaces (specific to SentencePiece/Whisper tokenizers)
            // 'Ġ' (U+0120) often indicates a space before the word.
            if (token.startsWith("Ġ")) {
                // Add a space unless it's the very first token or the previous token ended with a space
                if (!firstToken && transcript.length() > 0 && transcript.charAt(transcript.length() - 1) != ' ') {
                    transcript.append(" ");
                }
                transcript.append(token.substring(1)); // Append token without the leading 'Ġ'
            } else {
                transcript.append(token); // Append token directly
            }
            firstToken = false; // No longer the first token
        }

        String finalTranscript = transcript.toString().trim(); // Trim leading/trailing whitespace
        Log.i(TAG, "Final Transcript: " + finalTranscript);
        return finalTranscript;
    }

    // --- Helper function to find the index of the maximum value in a section of a FloatBuffer --- (Unchanged)
    private int findMaxIndex(FloatBuffer buffer, int startIndex, int length) {
        if (buffer == null) { Log.e(TAG, "Buffer is null in findMaxIndex"); return -1; }
        if (startIndex < 0 || length <= 0 || startIndex + length > buffer.limit()) {
            Log.e(TAG, "Invalid range for findMaxIndex. Start: " + startIndex + ", Length: " + length + ", Buffer limit: " + buffer.limit());
            return -1;
        }
        float maxVal = Float.NEGATIVE_INFINITY;
        int maxIdx = -1;
        for (int i = 0; i < length; i++) {
            float currentVal;
            try { currentVal = buffer.get(startIndex + i); }
            catch (IndexOutOfBoundsException e) { Log.e(TAG, "Index out of bounds in findMaxIndex: " + (startIndex + i), e); break; }
            if (Float.isNaN(currentVal)) continue; // Skip NaN values
            if (currentVal > maxVal) { maxVal = currentVal; maxIdx = i; }
        }
        if (maxIdx == -1) { Log.w(TAG, "No valid maximum found in buffer section [start=" + startIndex + ", len=" + length + "]"); }
        return maxIdx; // Return the index relative to the start of the segment (0 to length-1)
    }

    // --- Placeholder: createDummyInputBuffer --- (Unchanged, NOT used for actual transcription)
    private ByteBuffer createDummyInputBuffer(int[] shape, DataType dataType) {
        if (shape == null || shape.length == 0) { return null; }
        try {
            long numElements = 1;
            for (int i = 0; i < shape.length; i++) {
                int dim = shape[i]; int actualDim = (dim <= 0) ? ((i == shape.length - 1) ? 3000 : 1) : dim;
                if (dim <= 0) Log.w(TAG, "Shape dim "+i+" is non-positive ("+dim+"), using fixed size "+actualDim+" for dummy buffer.");
                if (Long.MAX_VALUE / actualDim < numElements) throw new IllegalArgumentException("Dummy buffer size overflow.");
                numElements *= actualDim;
            }
            if (numElements == 0) { return null; }
            int elementSize = getDataTypeSizeBytes(dataType); if (elementSize <= 0) { return null; }
            long totalBytesLong = numElements * elementSize;
            if (totalBytesLong > Integer.MAX_VALUE) { return null; }
            int totalBytes = (int) totalBytesLong;
            ByteBuffer buffer = ByteBuffer.allocateDirect(totalBytes);
            buffer.order(ByteOrder.nativeOrder());
            byte[] zeros = new byte[totalBytes]; buffer.put(zeros); buffer.rewind();
            Log.d(TAG, "Created DUMMY input buffer of size " + totalBytes + " bytes for shape " + Arrays.toString(shape) + ", Type: " + dataType);
            return buffer;
        } catch (IllegalArgumentException | OutOfMemoryError e) { Log.e(TAG, "Error creating dummy input buffer: " + e.getMessage()); return null; }
    }
    // --- Helper: getDataTypeSizeBytes --- (Unchanged)
    private int getDataTypeSizeBytes(DataType dataType) {
        switch (dataType) {
            case FLOAT32: return 4; case INT32: return 4; case UINT8: return 1;
            case INT64: return 8; case BOOL: return 1; case INT16: return 2;
            case INT8: return 1; default: Log.w(TAG, "Unsupported data type size calc: " + dataType); return -1;
        }
    }

    // --- Helper method to update UI components --- (Unchanged)
    private void updateUI(final String result, final String status) {
        mainHandler.post(() -> {
            if (result != null) textViewResult.setText(result);
            if (status != null) textViewStatus.setText(status);

            boolean modelReady = whisperHelper != null && idToTokenMap != null && !idToTokenMap.isEmpty();
            switch (status != null ? status : "") {
                case "Status: Recording...":
                    buttonStartRecord.setEnabled(false);
                    buttonStopRecord.setEnabled(true);
                    textViewResult.setText(""); // Clear previous result
                    break;
                case "Status: Stopping and Processing...":
                    buttonStartRecord.setEnabled(false);
                    buttonStopRecord.setEnabled(false);
                    break;
                case "Status: Ready":
                    buttonStartRecord.setEnabled(modelReady); // Should be true if we got here
                    buttonStopRecord.setEnabled(false);
                    break;
                case "Status: Idle":
                case "Status: Error":
                default:
                    buttonStartRecord.setEnabled(modelReady);
                    buttonStopRecord.setEnabled(false);
                    // Reset status to Ready if idle and model is ok, otherwise reflect error
                    if (status != null && !status.startsWith("Status: Error") && modelReady && !isRecording) {
                        textViewStatus.setText("Status: Ready");
                    } else if (status != null && !status.startsWith("Status: Error") && !modelReady) {
                        textViewStatus.setText("Status: Error loading model/vocab");
                    }
                    break;
            }
        });
    }

    // --- Helper: resetRecordingState --- (Unchanged)
    private synchronized void resetRecordingState() {
        isRecording = false;
        if (recordingThread != null) {
            if (recordingThread.isAlive()) { Log.w(TAG,"Interrupting recording thread during reset."); recordingThread.interrupt(); }
            recordingThread = null;
        }
        releaseAudioRecord();
        if (recordingBuffer != null) { try { recordingBuffer.close(); } catch (IOException e) { /* Ignored */ } recordingBuffer = null; }
        Log.d(TAG,"Internal recording state reset.");
        // Don't immediately update UI here, let the caller (e.g., stopRecording) handle the final state update
    }

    // --- Helper: releaseAudioRecord --- (Unchanged)
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
                } else { Log.w(TAG,"AudioRecord not initialized or already released, skipping release actions. State: "+audioRecord.getState()); }
            } catch (Exception e) { Log.e(TAG, "Exception releasing AudioRecord: " + e.getMessage(), e); }
            finally { audioRecord = null; }
        }
    }

    // --- *** UPDATED: Audio Preprocessing using JTransforms *** ---
    private ByteBuffer preprocessAudio(byte[] recordedAudioBytes) {
        if (recordedAudioBytes == null || recordedAudioBytes.length == 0) {
            Log.e(TAG, "No audio data to preprocess");
            return null;
        }
        if (whisperHelper == null) {
            Log.e(TAG, "Whisper helper is null in preprocessAudio");
            return null;
        }

        Log.d(TAG, "Preprocessing " + recordedAudioBytes.length + " bytes of audio data");

        int[] inputShape = whisperHelper.getInputShape(); // e.g., [1, 80, 3000] or [1, 1, 3000, 80]
        DataType inputDataType = whisperHelper.getInputDataType();

        if (inputShape == null || inputShape.length < 3 || inputDataType != DataType.FLOAT32) {
            Log.e(TAG, "Invalid input shape or data type for preprocessing. Shape: " + Arrays.toString(inputShape) + ", Type: " + inputDataType);
            return null;
        }

        // Determine Mel feature count and expected frame count from shape
        // Assuming shape is [Batch, Features, Frames] OR [Batch, Channels, Frames, Features]
        // Adapt this logic if your model uses a different layout (e.g., [Batch, Frames, Features])
        final int melFeatureCount = (inputShape.length == 4) ? inputShape[3] : inputShape[1]; // Typically 80
        final int expectedFrames = (inputShape.length == 4) ? inputShape[2] : inputShape[2]; // Typically 3000

        if (melFeatureCount <= 0 || expectedFrames <= 0) {
            Log.e(TAG, "Invalid dimensions extracted from shape: Features=" + melFeatureCount + ", Frames=" + expectedFrames);
            return null;
        }

        Log.d(TAG, "Model expects input features: " + melFeatureCount + ", frames: " + expectedFrames);


        try {
            // 1. Convert PCM 16-bit bytes to float array [-1.0, 1.0]
            short[] shorts = new short[recordedAudioBytes.length / 2];
            ByteBuffer.wrap(recordedAudioBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts);
            float[] floatAudio = new float[shorts.length];
            for (int i = 0; i < shorts.length; i++) {
                floatAudio[i] = shorts[i] / 32768.0f; // Normalize to [-1.0, 1.0]
            }
            Log.d(TAG, "Converted audio to " + floatAudio.length + " float samples.");

            // --- Parameters for STFT ---
            final int frameLength = 400; // 25ms at 16kHz (Whisper default) = 0.025 * 16000
            final int frameShift = 160;  // 10ms hop at 16kHz (Whisper default) = 0.010 * 16000
            final int fftSize = frameLength; // Use frameLength for FFT size (common practice)
            final int nFFT = fftSize; // Alias for clarity in calculations
            final int nMel = melFeatureCount; // From model input shape

            // --- Calculate number of frames ---
            // Note: Whisper might implicitly pad, so we calculate frames based on input length.
            // Padding/truncation of the *final mel spectrogram* will happen later.
            int numFrames = 0;
            if (floatAudio.length >= frameLength) {
                numFrames = 1 + (floatAudio.length - frameLength) / frameShift;
            }

            if (numFrames <= 0) {
                Log.e(TAG, "Audio too short ("+floatAudio.length+" samples) for STFT processing with frameLength="+frameLength);
                return null;
            }
            Log.d(TAG, "Audio will produce " + numFrames + " frames");

            // --- Precompute Hann Window & Mel Filterbank ---
            float[] hannWindow = createHannWindow(frameLength);
            float[][] melFilterbank = createMelFilterbank(nMel, nFFT, sampleRate); // [nMel, nFFT/2 + 1]

            // --- Allocate Mel Spectrogram array ---
            float[][] melSpectrogram = new float[numFrames][nMel];

            // --- Initialize FFT ---
            FloatFFT_1D fft = new FloatFFT_1D(nFFT);
            float[] fftInput = new float[nFFT]; // Buffer for windowed frame
            float[] fftOutput = new float[nFFT * 2]; // JTransforms requires 2*n for real fft (stores complex result)

            // --- Process each frame ---
            for (int frameIndex = 0; frameIndex < numFrames; frameIndex++) {
                int startSample = frameIndex * frameShift;
                int endSample = Math.min(startSample + frameLength, floatAudio.length); // Ensure we don't go out of bounds

                // --- Apply window and prepare for FFT ---
                Arrays.fill(fftInput, 0.0f); // Zero pad if frame is shorter than fftSize
                for (int i = 0; i < (endSample - startSample); i++) {
                    fftInput[i] = floatAudio[startSample + i] * hannWindow[i];
                }

                // --- Compute FFT ---
                // JTransforms realForward modifies input, copy if needed: System.arraycopy(fftInput, 0, fftOutput, 0, nFFT);
                // Or use realForwardFull which takes a separate output buffer.
                // We will use realForward which overwrites fftInput.
                fft.realForward(fftInput);

                // The result is stored in fftInput in a packed format:
                // fftInput[0] = Re[0], fftInput[1] = Re[n/2]
                // fftInput[2*k] = Re[k], fftInput[2*k+1] = Im[k] for k=1..n/2-1

                // --- Compute Power Spectrum (Magnitude Squared) ---
                // Size is nFFT/2 + 1 bins
                int numSpectrumBins = nFFT / 2 + 1;
                float[] powerSpectrum = new float[numSpectrumBins];

                powerSpectrum[0] = fftInput[0] * fftInput[0]; // DC component (Re[0]^2)
                for (int k = 1; k < nFFT / 2; k++) {
                    float real = fftInput[2 * k];
                    float imag = fftInput[2 * k + 1];
                    powerSpectrum[k] = real * real + imag * imag;
                }
                if (nFFT % 2 == 0) { // Nyquist freq for even nFFT
                    powerSpectrum[nFFT / 2] = fftInput[1] * fftInput[1]; // Nyquist component (Re[n/2]^2)
                }


                // --- Apply Mel Filterbank ---
                for (int melIndex = 0; melIndex < nMel; melIndex++) {
                    float melEnergy = 0;
                    for (int specIndex = 0; specIndex < numSpectrumBins; specIndex++) {
                        melEnergy += powerSpectrum[specIndex] * melFilterbank[melIndex][specIndex];
                    }
                    melSpectrogram[frameIndex][melIndex] = melEnergy; // Store linear Mel energy
                }
            }

            // --- Take Log of Mel Spectrogram ---
            final float logOffset = 1e-10f; // Small offset to avoid log(0)
            for (int i = 0; i < numFrames; i++) {
                for (int j = 0; j < nMel; j++) {
                    // Whisper typically uses natural log (Math.log)
                    melSpectrogram[i][j] = (float) Math.log(Math.max(melSpectrogram[i][j], 0.0f) + logOffset);
                    // Using Math.max to ensure non-negative input to log, though energies should be >= 0
                }
            }
            Log.d(TAG, "Calculated Log Mel Spectrogram.");

            // --- Normalize (Simple Z-Score - may need refinement based on Whisper's exact method) ---
            // Calculate mean and stddev across *all* log mel values
            float mean = 0f;
            int totalElements = numFrames * nMel;
            for (int i = 0; i < numFrames; i++) { for (int j = 0; j < nMel; j++) { mean += melSpectrogram[i][j]; } }
            mean /= totalElements;

            float stddev = 0f;
            for (int i = 0; i < numFrames; i++) {
                for (int j = 0; j < nMel; j++) {
                    float diff = melSpectrogram[i][j] - mean;
                    stddev += diff * diff;
                }
            }
            stddev = (float) Math.sqrt(stddev / totalElements);
            Log.d(TAG, "Log Mel Mean: " + mean + ", StdDev: " + stddev);

            final float stdEps = 1e-5f; // Epsilon to prevent division by zero
            for (int i = 0; i < numFrames; i++) {
                for (int j = 0; j < nMel; j++) {
                    melSpectrogram[i][j] = (melSpectrogram[i][j] - mean) / (stddev + stdEps);
                }
            }
            Log.d(TAG, "Normalized Log Mel Spectrogram.");

            // --- Pad or Truncate Mel Spectrogram to expected model input size ---
            float[][] finalMel; // Shape [expectedFrames][melFeatureCount]

            if (numFrames > expectedFrames) {
                // Truncate (take the first 'expectedFrames')
                Log.d(TAG, "Truncating " + numFrames + " frames to " + expectedFrames);
                finalMel = new float[expectedFrames][nMel];
                for (int i = 0; i < expectedFrames; i++) {
                    // System.arraycopy(source, srcPos, dest, destPos, length)
                    System.arraycopy(melSpectrogram[i], 0, finalMel[i], 0, nMel);
                }
            } else if (numFrames < expectedFrames) {
                // Pad with zeros (or a suitable padding value, often reflects silence ~min log value)
                Log.d(TAG, "Padding " + numFrames + " frames to " + expectedFrames);
                finalMel = new float[expectedFrames][nMel];
                // Determine padding value (e.g., Z-score of logOffset, or just zero after normalization)
                // float paddingValue = (float) (Math.log(logOffset) - mean) / (stddev + stdEps); // Approx padding value
                float paddingValue = 0.0f; // Simpler: pad with zero after normalization

                for (int i = 0; i < expectedFrames; i++) {
                    if (i < numFrames) {
                        // Copy existing frame data
                        System.arraycopy(melSpectrogram[i], 0, finalMel[i], 0, nMel);
                    } else {
                        // Fill padding frame
                        Arrays.fill(finalMel[i], paddingValue);
                    }
                }
            } else {
                // Exact match, just assign
                finalMel = melSpectrogram;
            }

            // --- Create ByteBuffer for TFLite input ---
            // Expected shape is likely [1, nMel, expectedFrames] or [1, expectedFrames, nMel] or [1, 1, expectedFrames, nMel] etc.
            // We need to flatten finalMel [expectedFrames][nMel] into the buffer in the correct order.
            // TFLite usually expects NCHW or NHWC. Let's assume the model wants [1, nMel, expectedFrames] for now.
            // If it's [1, expectedFrames, nMel], the inner/outer loops below need swapping.
            // Check WhisperHelper log for input tensor shape if unsure.

            // Total size = Batch * Channels * Height * Width * sizeof(float)
            // Or Batch * Frames * Features * sizeof(float) etc.
            // Size should match inputTensor.numBytes() from WhisperHelper.
            long numInputElements = 1;
            for(int dim : inputShape) { if(dim > 0) numInputElements *= dim; } // Calculate elements ignoring batch if -1
            int totalBytes = (int) numInputElements * 4; // 4 bytes per float

            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(totalBytes);
            inputBuffer.order(ByteOrder.nativeOrder()); // Use native byte order

            // Assuming input shape [Batch=1, MelFeatures, Frames] format
            if (inputShape.length == 3 && inputShape[0] == 1 && inputShape[1] == nMel && inputShape[2] == expectedFrames) {
                Log.d(TAG, "Filling buffer assuming [1, MelFeatures, Frames] layout");
                for (int j = 0; j < nMel; j++) {        // Iterate features (rows in finalMel after transpose logic)
                    for (int i = 0; i < expectedFrames; i++) { // Iterate frames (columns in finalMel after transpose logic)
                        // Need data from finalMel[i][j]
                        inputBuffer.putFloat(finalMel[i][j]);
                    }
                }
                // Assuming input shape [Batch=1, Frames, MelFeatures] format
            } else if (inputShape.length == 3 && inputShape[0] == 1 && inputShape[1] == expectedFrames && inputShape[2] == nMel) {
                Log.d(TAG, "Filling buffer assuming [1, Frames, MelFeatures] layout");
                for (int i = 0; i < expectedFrames; i++) {        // Iterate frames
                    for (int j = 0; j < nMel; j++) { // Iterate features
                        inputBuffer.putFloat(finalMel[i][j]);
                    }
                }
                // Assuming input shape [Batch=1, Channels=1, Frames, MelFeatures] format
            } else if (inputShape.length == 4 && inputShape[0] == 1 && inputShape[1] == 1 && inputShape[2] == expectedFrames && inputShape[3] == nMel) {
                Log.d(TAG, "Filling buffer assuming [1, 1, Frames, MelFeatures] layout");
                for (int i = 0; i < expectedFrames; i++) {        // Iterate frames
                    for (int j = 0; j < nMel; j++) { // Iterate features
                        inputBuffer.putFloat(finalMel[i][j]);
                    }
                }
            }
            else {
                Log.e(TAG, "Input shape " + Arrays.toString(inputShape) + " does not match expected layouts for filling buffer. ABORTING.");
                return null; // Or attempt a default filling, but it's likely wrong.
            }


            inputBuffer.rewind(); // Prepare buffer for reading by TFLite
            Log.d(TAG, "Created preprocessed input buffer of size " + totalBytes + " bytes. Capacity: " + inputBuffer.capacity() + ", Limit: " + inputBuffer.limit());

            if (totalBytes != inputBuffer.capacity()) {
                Log.e(TAG, "Mismatch between calculated byte size ("+totalBytes+") and buffer capacity ("+inputBuffer.capacity()+")!");
                // This might happen if inputShape has dynamic dimensions (-1) not handled correctly.
                // Check WhisperHelper logs for the concrete input tensor byte size.
            }

            return inputBuffer;

        } catch (Exception e) {
            Log.e(TAG, "Error during audio preprocessing: " + e.getMessage(), e);
            return null;
        }
    }

    // --- *** NEW/UPDATED HELPER METHODS FOR MEL SPECTROGRAM *** ---

    // Create Hann Window
    private float[] createHannWindow(int length) {
        float[] window = new float[length];
        for (int i = 0; i < length; i++) {
            // Hann formula: 0.5 * (1 - cos(2 * PI * i / (N - 1)))
            window[i] = (float) (0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / (length - 1))));
        }
        return window;
    }

    // Create Mel Filterbank Matrix
    private float[][] createMelFilterbank(int numMelBins, int fftSize, int sampleRate) {
        int numSpectrumBins = fftSize / 2 + 1;
        float[][] filterbank = new float[numMelBins][numSpectrumBins];

        float minMel = hzToMel(0); // Typically 0 Hz minimum
        float maxMel = hzToMel(sampleRate / 2); // Nyquist frequency

        // Calculate Mel frequency points (linearly spaced in Mel scale)
        float[] melPoints = new float[numMelBins + 2]; // We need N+2 points to define N filters
        for (int i = 0; i < numMelBins + 2; i++) {
            melPoints[i] = minMel + i * (maxMel - minMel) / (numMelBins + 1);
        }

        // Convert Mel points back to Hz and then to FFT bin indices
        int[] binIndices = new int[numMelBins + 2];
        for (int i = 0; i < numMelBins + 2; i++) {
            binIndices[i] = Math.round(melToHz(melPoints[i]) * fftSize / sampleRate);
            // Ensure indices are within the valid range [0, numSpectrumBins - 1]
            binIndices[i] = Math.max(0, Math.min(binIndices[i], numSpectrumBins - 1));
        }

        // Create triangular filters
        for (int i = 0; i < numMelBins; i++) {
            int startBin = binIndices[i];
            int centerBin = binIndices[i + 1];
            int endBin = binIndices[i + 2];

            // Calculate slopes for the triangle
            float risingSlope = (centerBin - startBin == 0) ? 0 : 1.0f / (centerBin - startBin);
            float fallingSlope = (endBin - centerBin == 0) ? 0 : 1.0f / (endBin - centerBin);

            // Apply rising edge
            for (int k = startBin; k < centerBin; k++) {
                filterbank[i][k] = (k - startBin) * risingSlope;
            }
            // Apply falling edge
            for (int k = centerBin; k < endBin; k++) {
                filterbank[i][k] = 1.0f - (k - centerBin) * fallingSlope;
            }
            // Ensure the peak is exactly 1 at center bin if start/center/end coincide
            if (startBin == centerBin && centerBin == endBin && centerBin < numSpectrumBins) {
                filterbank[i][centerBin] = 1.0f;
            } else if (startBin == centerBin && centerBin < endBin && centerBin < numSpectrumBins) {
                // Handle case where start and center are the same: Only falling slope applies from center
                filterbank[i][centerBin] = 1.0f; // Peak is at center
            } else if (startBin < centerBin && centerBin == endBin && centerBin < numSpectrumBins) {
                // Handle case where center and end are the same: Only rising slope applies up to center
                // Peak is achieved at centerBin, which is filterbank[i][centerBin-1] via rising slope logic
            }

        }
        Log.d(TAG, "Created Mel Filterbank with " + numMelBins + " bins.");
        return filterbank;
    }

    // Convert Hz to Mel (HTK formula - commonly used)
    private float hzToMel(float hz) {
        return (float) (2595.0 * Math.log10(1.0 + hz / 700.0));
    }

    // Convert Mel to Hz (HTK formula)
    private float melToHz(float mel) {
        return (float) (700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0));
    }
    // --- END OF NEW/UPDATED HELPERS ---


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
                buttonStartRecord.setEnabled(false); // Ensure button stays disabled if permission denied
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i(TAG,"onDestroy called.");
        if (isRecording) {
            Log.w(TAG, "Activity destroyed while recording. Forcing stop/release.");
            // Force stop recording process safely
            isRecording = false; // Signal recording thread to stop
            releaseAudioRecord(); // Release hardware
            if (recordingThread != null) {
                recordingThread.interrupt(); // Interrupt if still alive
                recordingThread = null;
            }
            if (recordingBuffer != null) {
                try { recordingBuffer.close(); } catch (IOException e) {/*ignore*/}
                recordingBuffer = null;
            }
        }
        // Shutdown executor service
        inferenceExecutorService.shutdown();
        try {
            if (!inferenceExecutorService.awaitTermination(500, java.util.concurrent.TimeUnit.MILLISECONDS)) {
                inferenceExecutorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            inferenceExecutorService.shutdownNow();
            Thread.currentThread().interrupt();
        }
        // Close TFLite interpreter
        if (whisperHelper != null) {
            whisperHelper.close();
            whisperHelper = null;
        }
        // Remove any pending UI updates
        mainHandler.removeCallbacksAndMessages(null);
        Log.i(TAG,"Activity destroyed, resources released.");
    }
} // End of MainActivity class