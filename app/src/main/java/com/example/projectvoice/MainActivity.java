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

import java.io.BufferedReader; // Added
import java.io.ByteArrayOutputStream;
import java.io.IOException; // Added
import java.io.InputStream; // Added
import java.io.InputStreamReader; // Added
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer; // Added
import java.nio.IntBuffer; // Added
import java.util.ArrayList; // Added
import java.util.Arrays;
import java.util.HashMap; // Added
import java.util.HashSet; // Added
import java.util.List; // Added
import java.util.Map;
import java.util.Set; // Added
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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
    private int bufferSizeInBytes = AudioRecord.ERROR_BAD_VALUE; // Renamed for clarity

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
        setContentView(R.layout.activity_main); // Ensure res/layout/activity_main.xml exists

        textViewStatus = findViewById(R.id.textViewStatus);
        textViewResult = findViewById(R.id.textViewResult);
        buttonStartRecord = findViewById(R.id.buttonStartRecord);
        buttonStopRecord = findViewById(R.id.buttonStopRecord);

        // Initialize Whisper Helper and Load Vocabulary
        try {
            // Using default model "whisper-tiny.tflite" or provide specific path in constructor
            whisperHelper = new WhisperHelper(this);
            textViewStatus.setText("Status: Model Loaded (Vocab Skipped)");

            // loadVocabulary();
        
            buttonStartRecord.setEnabled(whisperHelper != null);

            // Enable start button only if both model and vocab loaded successfully
            // buttonStartRecord.setEnabled(whisperHelper != null && idToTokenMap != null && !idToTokenMap.isEmpty());
            // if (!buttonStartRecord.isEnabled()) {
            //      textViewStatus.setText("Status: Error loading model/vocab");
            //      Toast.makeText(this, "Model or vocabulary failed to load.", Toast.LENGTH_LONG).show();
            // }

        } catch (Exception e) {
            textViewStatus.setText("Status: Error loading model/vocab");
            Log.e(TAG, "Error initializing WhisperHelper or loading Vocabulary", e);
            Toast.makeText(this, "Failed to load model/vocab: " + e.getMessage(), Toast.LENGTH_LONG).show();
            buttonStartRecord.setEnabled(false);
            buttonStopRecord.setEnabled(false);
        }

        buttonStartRecord.setOnClickListener(v -> startRecording());
        buttonStopRecord.setOnClickListener(v -> stopRecordingAndTranscribe());

        // Calculate buffer size
        bufferSizeInBytes = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat);
        if (bufferSizeInBytes == AudioRecord.ERROR || bufferSizeInBytes == AudioRecord.ERROR_BAD_VALUE) {
            Log.w(TAG, "Min buffer size calculation failed. Using default (1 sec).");
            // Use a default buffer size (e.g., 1 second of data) if calculation fails
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

            // Use a buffer size larger than the minimum for reading efficiency
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
                 releaseAudioRecord(); // Clean up if failed
                 return;
            }

            // --- Start Reading Audio in a Separate Thread ---
            recordingBuffer = new ByteArrayOutputStream();
            isRecording = true; // Set flag before starting thread

            recordingThread = new Thread(() -> {
                // Use the minimum buffer size for reading chunks
                byte[] audioDataBuffer = new byte[bufferSizeInBytes];
                Log.d(TAG, "Recording thread started. Reading in chunks of " + bufferSizeInBytes + " bytes.");
                // Check isRecording flag within the loop
                while (isRecording && audioRecord != null && audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                    int bytesRead = audioRecord.read(audioDataBuffer, 0, audioDataBuffer.length);
                    if (bytesRead > 0) {
                        // Write the valid data to our ByteArrayOutputStream
                        recordingBuffer.write(audioDataBuffer, 0, bytesRead);
                    } else if (bytesRead < 0) {
                        Log.e(TAG, "Error reading audio data: " + bytesRead);
                        // Optionally break or handle error based on the error code
                        // e.g., AudioRecord.ERROR_INVALID_OPERATION, ERROR_BAD_VALUE, ERROR_DEAD_OBJECT, ERROR
                         if (bytesRead == AudioRecord.ERROR_INVALID_OPERATION || bytesRead == AudioRecord.ERROR_BAD_VALUE) {
                              // Consider stopping if fundamental error occurs
                              // isRecording = false; // Signal loop to stop
                              Log.e(TAG, "Stopping recording thread due to read error: " + bytesRead);
                              break;
                         }
                    }
                    // Small sleep to prevent busy-waiting if needed, but read() should block
                    // try { Thread.sleep(10); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                }
                Log.d(TAG,"Recording thread finished.");
            }, "AudioRecorder Thread");


            audioRecord.startRecording(); // Start hardware recording
            recordingThread.start(); // Start the software reading thread

            updateUI(null, "Status: Recording..."); // Update UI on main thread
            Log.i(TAG, "Recording started.");

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
        if (!isRecording) {
            Toast.makeText(this, "Not currently recording.", Toast.LENGTH_SHORT).show();
            return;
        }

        updateUI(null, "Status: Stopping and Processing..."); // Update status immediately

        // Signal the recording thread to stop and wait for it
        isRecording = false; // Set flag first
        if (recordingThread != null) {
            try {
                recordingThread.join(500); // Wait max 500ms for thread to finish
                 if (recordingThread.isAlive()) {
                    Log.w(TAG, "Recording thread did not finish within timeout. Interrupting.");
                    recordingThread.interrupt();
                 }
            } catch (InterruptedException e) {
                Log.e(TAG, "Interrupted while waiting for recording thread", e);
                Thread.currentThread().interrupt(); // Restore interrupt status
            }
            recordingThread = null;
        }

        // Stop and release AudioRecord instance (must happen after thread finishes reading)
        releaseAudioRecord(); // Use synchronized method

        final byte[] recordedAudioBytes = (recordingBuffer != null) ? recordingBuffer.toByteArray() : null;
        try {
            if (recordingBuffer != null) recordingBuffer.close(); // Close the stream
        } catch (IOException e) {
            Log.e(TAG, "Error closing recording buffer stream", e);
        }
        recordingBuffer = null; // Release the buffer memory reference

        if (recordedAudioBytes == null || recordedAudioBytes.length == 0) {
             Log.w(TAG,"No audio data captured.");
             updateUI("No audio data captured.", "Status: Idle");
             return;
        }

        Log.i(TAG, "Recorded " + recordedAudioBytes.length + " bytes of audio (" + (recordedAudioBytes.length / (float)(sampleRate*2)) + " seconds).");

        // Submit the inference task to the background thread
        inferenceExecutorService.submit(() -> {
            if (whisperHelper == null) {
                Log.e(TAG, "WhisperHelper is null, cannot preprocess or transcribe.");
                updateUI("Error: Model helper not available.", "Status: Error");
                return; // Exit background task
            }

            // --- TODO: Preprocess Audio Data ---
            // CRITICAL STEP: Convert `recordedAudioBytes` (raw 16-bit PCM) into the
            // format expected by the Whisper model (Mel Spectrogram FloatBuffer/ByteBuffer).
            // You MUST know the exact input shape and type required by your .tflite model.
            DataType inputDataType = whisperHelper.getInputDataType();
            int[] inputShape = whisperHelper.getInputShape();
             if (inputDataType == null || inputShape == null) {
                  Log.e(TAG, "Could not get model input details from WhisperHelper.");
                  updateUI("Error: Failed to get model input info.", "Status: Error");
                  return; // Exit background task
             }
             Log.d(TAG, "Model Expected Input Shape: " + Arrays.toString(inputShape) + ", Type: " + inputDataType);

            // --- Placeholder: Create a dummy input buffer (REPLACE THIS) ---
            Log.w(TAG, "!!! Using DUMMY input buffer. Preprocessing implementation needed! !!!");
            ByteBuffer inputBuffer = createDummyInputBuffer(inputShape, inputDataType);
            // --- End Preprocessing Placeholder ---


            // --- Run Transcription ---
            Map<Integer, Object> transcriptionOutput = null;
            String resultText = "Preprocessing Failed / Dummy Data Used"; // Default message

            if (inputBuffer != null) { // Only run if dummy input created (or real preprocessing succeeded)
                Log.d(TAG, "Calling whisperHelper.transcribe with " + (inputBuffer.isDirect() ? "Direct" : "Heap") + " ByteBuffer, Size: " + inputBuffer.capacity());
                long startTime = System.currentTimeMillis();
                transcriptionOutput = whisperHelper.transcribe(inputBuffer); // Pass the *real* preprocessed buffer eventually
                long endTime = System.currentTimeMillis();
                Log.d(TAG, "Transcription attempt finished in " + (endTime-startTime) + " ms.");

                // --- Decode Output ---
                if (transcriptionOutput != null && transcriptionOutput.containsKey(outputTensorIndex)) {
                     Object rawOutput = transcriptionOutput.get(outputTensorIndex);
                     DataType outputDataType = whisperHelper.getOutputDataType(); // Get output type from helper

                      if (rawOutput instanceof ByteBuffer) {
                           ByteBuffer outputByteBuffer = (ByteBuffer) rawOutput;
                           Log.d(TAG, "Attempting to decode output buffer. Size: " + outputByteBuffer.remaining() + " bytes. Type: " + outputDataType);
                           // Call the decoding function
                           resultText = decodeOutputBuffer(outputByteBuffer, outputDataType);

                      } else {
                           resultText = "Transcription output format unexpected: " + (rawOutput != null ? rawOutput.getClass().getName() : "null");
                           Log.e(TAG, resultText);
                      }
                } else {
                     resultText = "Transcription failed or output tensor not found.";
                     Log.e(TAG, resultText);
                }
                // --- End Decode Output ---

            } else {
                 Log.e(TAG,"Cannot run transcription, input buffer is null (Preprocessing likely failed)");
                 resultText = "Error: Preprocessing failed."; // Update result text
            }

            // Update UI on the main thread
            updateUI(resultText, "Status: Idle");
            Log.i(TAG,"Background processing complete.");
        });
    }


    // --- Function to Load Vocabulary ---
    // !!! IMPORTANT: This assumes a VERY simple format for the .bin file: !!!
    // !!! Each line is "TOKEN_ID<space>TOKEN_STRING"                 !!!
    // !!! You MUST adapt this based on the ACTUAL format of your .bin file !!!
    // !!! It might be gzipped, use tabs, be binary, etc.               !!!
    private void loadVocabulary() {
        idToTokenMap = new HashMap<>();
        InputStream inputStream = null;
        BufferedReader reader = null;
        try {
            Log.i(TAG, "Attempting to load vocabulary from: " + VOCAB_FILENAME);
            inputStream = getAssets().open(VOCAB_FILENAME);
            // Consider checking if it's GZipped: new GZIPInputStream(inputStream)
            reader = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            int count = 0;
            while ((line = reader.readLine()) != null) {
                // Example: Split by space, assuming ID then Token
                String[] parts = line.trim().split("\\s+", 2);
                if (parts.length == 2) {
                    try {
                        int id = Integer.parseInt(parts[0]);
                        String token = parts[1];
                        idToTokenMap.put(id, token);
                        count++;
                        // Log first few and last few to verify
                        // if (count < 5 || count > 51860) { Log.d(TAG, "Loaded vocab ID: " + id + " -> '" + token + "'"); }
                    } catch (NumberFormatException e) {
                        Log.w(TAG, "Skipping invalid vocab line (ID parse failed): " + line + " - Error: " + e.getMessage());
                    }
                } else {
                    Log.w(TAG, "Skipping malformed vocab line (parts != 2): " + line);
                }
            }
            if (idToTokenMap.isEmpty()) {
                 Log.e(TAG, "Vocabulary map is empty after loading! Check file format and content: " + VOCAB_FILENAME);
                 Toast.makeText(this, "Vocabulary empty. Check format.", Toast.LENGTH_LONG).show();
                 idToTokenMap = null; // Set to null if loading failed
            } else {
                 Log.i(TAG, "Vocabulary loaded successfully. Size: " + idToTokenMap.size());
            }

        } catch (IOException e) {
            Log.e(TAG, "Error loading vocabulary file: " + VOCAB_FILENAME, e);
            Toast.makeText(this, "Failed to load vocabulary.", Toast.LENGTH_SHORT).show();
            idToTokenMap = null; // Ensure map is null on error
        } finally {
            try {
                if (reader != null) reader.close();
                if (inputStream != null) inputStream.close();
            } catch (IOException e) {
                Log.e(TAG, "Error closing vocabulary stream", e);
            }
        }
    }

    // --- Function to Decode Output Buffer ---
    private String decodeOutputBuffer(ByteBuffer outputBuffer, DataType outputDataType) {
        if (outputBuffer == null || idToTokenMap == null || idToTokenMap.isEmpty()) {
            Log.e(TAG, "Cannot decode: Output buffer is null or vocabulary not loaded.");
            return "Decoding Error: Vocab not loaded or null buffer.";
        }
        if (outputDataType == null) {
            // Try getting it from helper again if not passed correctly
             outputDataType = whisperHelper != null ? whisperHelper.getOutputDataType() : null;
             if (outputDataType == null) {
                Log.e(TAG, "Cannot decode: Output data type is unknown.");
                return "Decoding Error: Unknown output data type.";
             }
        }

        outputBuffer.rewind(); // Ensure we read from the beginning

        List<Integer> tokenIds = new ArrayList<>();
        int bufferLimit = outputBuffer.limit(); // How many bytes are in the buffer
        Log.d(TAG, "Decoding output buffer. Type: " + outputDataType + ", Size (bytes): " + bufferLimit);


        // Read token IDs based on the data type
        try { // Add try-catch for buffer operations
            switch (outputDataType) {
                case INT32:
                    if (bufferLimit % 4 != 0) {
                        Log.e(TAG, "INT32 buffer size (" + bufferLimit + ") not divisible by 4.");
                        return "Decoding Error: Invalid INT32 buffer size.";
                    }
                    IntBuffer intBuffer = outputBuffer.asIntBuffer();
                    int numInts = intBuffer.remaining(); // Use remaining() after asIntBuffer()
                    Log.d(TAG, "Decoding INT32 buffer with " + numInts + " elements.");
                    for (int i = 0; i < numInts; i++) {
                        tokenIds.add(intBuffer.get(i));
                    }
                    break;

                case FLOAT32:
                    if (bufferLimit % 4 != 0) {
                        Log.e(TAG, "FLOAT32 buffer size (" + bufferLimit + ") not divisible by 4.");
                        return "Decoding Error: Invalid FLOAT32 buffer size.";
                    }
                    FloatBuffer floatBuffer = outputBuffer.asFloatBuffer();
                    int numFloats = floatBuffer.remaining();
                    int vocabSize = idToTokenMap.size(); // Or get from WhisperHelper output shape if possible
                    if (vocabSize <= 0) {
                        Log.e(TAG,"Vocab size is invalid ("+vocabSize+"). Cannot decode FLOAT32 output.");
                        return "Decoding Error: Invalid vocab size.";
                    }

                    // Assuming output shape is [1, sequenceLength, vocabSize] or similar
                    // Check if total floats is divisible by vocab size
                     if (numFloats % vocabSize != 0) {
                         Log.w(TAG, "Warning: Output buffer float count ("+numFloats+") is not perfectly divisible by presumed vocab size ("+vocabSize+"). Shape might be different than expected.");
                         // Proceed cautiously or return error depending on model specifics
                     }
                     int sequenceLength = numFloats / vocabSize; // Estimated sequence length

                     Log.d(TAG, "Decoding FLOAT32 buffer. Float count: " + numFloats + ", SeqLen(est): " + sequenceLength + ", VocabSize: "+ vocabSize);

                    for (int i = 0; i < sequenceLength; i++) {
                        int startIndex = i * vocabSize;
                        if (startIndex + vocabSize > numFloats) { // Boundary check
                             Log.w(TAG,"Attempting to read past buffer limit during argmax. Stopping.");
                             break;
                        }
                        int bestTokenId = findMaxIndex(floatBuffer, startIndex, vocabSize);
                        if (bestTokenId != -1) { // Check if findMaxIndex succeeded
                            tokenIds.add(bestTokenId);
                        }
                    }
                    break;

                // Add cases for INT64, INT8, etc. if your model might output those
                 case INT64:
                     if (bufferLimit % 8 != 0) {
                         Log.e(TAG, "INT64 buffer size (" + bufferLimit + ") not divisible by 8.");
                         return "Decoding Error: Invalid INT64 buffer size.";
                     }
                     java.nio.LongBuffer longBuffer = outputBuffer.asLongBuffer();
                     int numLongs = longBuffer.remaining();
                     Log.d(TAG, "Decoding INT64 buffer with " + numLongs + " elements.");
                     for (int i = 0; i < numLongs; i++) {
                         // Whisper IDs usually fit in int, but handle potential large IDs if needed
                          long id = longBuffer.get(i);
                          if (id > Integer.MAX_VALUE || id < Integer.MIN_VALUE) {
                               Log.w(TAG, "Warning: INT64 token ID " + id + " outside standard int range.");
                               tokenIds.add(-1); // Or handle appropriately
                          } else {
                               tokenIds.add((int) id);
                          }
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


        // Convert token IDs to text using the vocabulary
        StringBuilder transcript = new StringBuilder();
        boolean firstToken = true;
        for (int tokenId : tokenIds) {
             if (tokenId == -1) continue; // Skip invalid tokens from INT64 case or other errors
            String token = idToTokenMap.getOrDefault(tokenId, "[UNK:" + tokenId + "]"); // Handle unknown tokens

            // Stop decoding at the end-of-text token
            if (token.equals("<|endoftext|>")) {
                 Log.d(TAG,"End-of-text token encountered.");
                break;
            }

            // Filter out other special tokens
            if (SPECIAL_TOKENS.contains(token)) {
                Log.d(TAG,"Skipping special token: "+token);
                continue; // Skip other special tokens
            }

            // Basic detokenization (replace G2P markers, handle spaces) - This needs improvement for perfect GPT-2!
            // Simplification: Treat "Ġ" as a space marker at the beginning of a word.
            if (token.startsWith("Ġ")) {
                // Append space only if it's not the very first token AND the previous didn't end in space
                if (!firstToken && transcript.length() > 0 && transcript.charAt(transcript.length() - 1) != ' ') {
                    transcript.append(" ");
                }
                transcript.append(token.substring(1)); // Append token without the marker
            } else {
                transcript.append(token); // Append token directly
            }
            firstToken = false; // No longer the first token
        }

        Log.d(TAG, "Raw decoded IDs: " + tokenIds.toString());
        String finalTranscript = transcript.toString().trim();
        Log.i(TAG, "Final Transcript: " + finalTranscript);
        return finalTranscript; // Return the final text
    }

    // Helper function to find the index of the maximum value in a section of a FloatBuffer
    private int findMaxIndex(FloatBuffer buffer, int startIndex, int length) {
        if (startIndex < 0 || length <= 0 || startIndex + length > buffer.limit()) {
             Log.e(TAG, "Invalid range for findMaxIndex. Start: "+startIndex+", Length: "+length+", Limit: "+buffer.limit());
             return -1; // Invalid arguments or range
        }
        float maxVal = -Float.MAX_VALUE;
        int maxIdx = -1; // This index is relative to the start of the vocab list (0 to vocabSize-1)
        for (int i = 0; i < length; i++) {
            float currentVal = buffer.get(startIndex + i);
            if (currentVal > maxVal) {
                maxVal = currentVal;
                maxIdx = i; // The index within the vocab corresponds to the Token ID
            }
        }
        // The index 'maxIdx' corresponds to the token ID
        return maxIdx;
    }


     // Helper to create a dummy buffer - REPLACE with actual audio preprocessing
     private ByteBuffer createDummyInputBuffer(int[] shape, DataType dataType) {
         // --- THIS IS A PLACEHOLDER - DO NOT USE FOR REAL TRANSCRIPTION ---
         if (shape == null || shape.length == 0) {
             Log.e(TAG, "Invalid shape for dummy buffer creation.");
             return null;
         }
         try {
             long numElements = 1;
             for (int dim : shape) {
                 // Models often have dynamic dimensions (e.g., time, marked as -1 or 0).
                 // Use a reasonable fixed size for dummy data. Whisper tiny expects 3000 frames (30s).
                 // Shape might be [1, 80, 3000] (batch, mel_bins, frames)
                 int actualDim = dim;
                  if (dim <= 0) {
                       if (shape.length == 3 && Arrays.asList(shape).indexOf(dim) == 2) { // If it's likely the time dimension
                            actualDim = 3000; // Use standard whisper frame count for dummy data
                            Log.w(TAG, "Shape dimension is non-positive (" + dim + "), using fixed size " + actualDim + " for dummy buffer size calculation (assuming time dimension).");
                       } else {
                            actualDim = 1; // Default to 1 for other non-positive dims
                            Log.w(TAG, "Shape dimension is non-positive (" + dim + "), using 1 for dummy buffer size calculation.");
                       }
                  }
                  // Prevent overflow during multiplication
                  if (Long.MAX_VALUE / actualDim < numElements) {
                       throw new IllegalArgumentException("Dummy buffer size calculation overflowed.");
                  }
                 numElements *= actualDim;
             }
             if (numElements == 0) {
                 Log.e(TAG, "Calculated zero elements for shape: " + Arrays.toString(shape));
                 return null;
             }


             int elementSize = getDataTypeSizeBytes(dataType);
             if (elementSize <= 0) {
                  Log.e(TAG, "Unsupported data type for dummy buffer: " + dataType);
                  return null;
             }

             long totalBytesLong = numElements * elementSize;
             if (totalBytesLong > Integer.MAX_VALUE) {
                  Log.e(TAG, "Required buffer size exceeds Integer.MAX_VALUE: " + totalBytesLong);
                  // Consider if model actually requires smaller input or if dummy size is too large
                  return null; // ByteBuffer uses int size
             }
             int totalBytes = (int) totalBytesLong;


             ByteBuffer buffer = ByteBuffer.allocateDirect(totalBytes); // Use Direct buffer for TFLite
             buffer.order(ByteOrder.nativeOrder());

             // Fill with zeros - models are sometimes tolerant of zero input for testing
             // No need to fill explicitly, allocateDirect often zeros memory, but being explicit is safer.
             byte[] zeros = new byte[totalBytes]; // Allocate temporary zero array
             buffer.put(zeros);
             buffer.rewind(); // Rewind after putting data


             Log.d(TAG, "Created DUMMY input buffer of size " + totalBytes + " bytes for shape " + Arrays.toString(shape) + ", Type: " + dataType);
             return buffer;

         } catch (IllegalArgumentException e) {
              Log.e(TAG, "Error creating dummy input buffer: " + e.getMessage());
              return null;
         } catch (OutOfMemoryError e) {
             Log.e(TAG, "OutOfMemoryError creating dummy input buffer. Model input might be too large.");
             return null;
         }
     }

    // Helper function to get size of TFLite DataType in bytes
     private int getDataTypeSizeBytes(DataType dataType) {
         switch (dataType) {
            case FLOAT32: return 4;
            case INT32:   return 4;
            case UINT8:   return 1; // Not common for Whisper input/output but possible
            case INT64:   return 8;
            case BOOL:    return 1;
            case INT16:   return 2; // Sometimes used for audio input
            case INT8:    return 1; // Often used for quantized models
            // case FLOAT16: return 2; // Requires specific handling if supported by device/delegate
            default:
                Log.w(TAG, "Unsupported or unknown data type size calculation for: " + dataType);
                return -1; // Indicate error or unsupported type
         }
     }

    // Helper method to update UI components from any thread
    private void updateUI(final String result, final String status) {
         mainHandler.post(() -> {
             if (result != null) {
                 textViewResult.setText(result);
             }
             if (status != null) {
                 textViewStatus.setText(status);
             }
             // Determine button states based on status
             boolean modelReady = whisperHelper != null && idToTokenMap != null && !idToTokenMap.isEmpty();
             switch (status != null ? status : "") {
                 case "Status: Recording...":
                     buttonStartRecord.setEnabled(false);
                     buttonStopRecord.setEnabled(true);
                     textViewResult.setText(""); // Clear previous result on start
                     break;
                 case "Status: Stopping and Processing...":
                     buttonStartRecord.setEnabled(false); // Keep disabled
                     buttonStopRecord.setEnabled(false); // Disable stop during processing
                     break;
                 case "Status: Idle":
                 case "Status: Error":
                 default: // Includes initial state and error states
                     buttonStartRecord.setEnabled(modelReady); // Enable only if model/vocab loaded
                     buttonStopRecord.setEnabled(false);
                      // Ensure status reflects idle state if it wasn't explicitly set to error
                      if (status != null && !status.startsWith("Status: Error") && textViewStatus.getText().toString().startsWith("Status:")) {
                         textViewStatus.setText("Status: Idle");
                      }
                     break;
             }
         });
    }

    // Resets button states, typically called from updateUI on Main thread (now integrated into updateUI)
    // private void resetRecordingStateUI() { ... } // Removed as logic is in updateUI switch

    // Only resets internal state variables, called from appropriate thread
     private synchronized void resetRecordingState() {
          isRecording = false; // Ensure flag is false
          if (recordingThread != null) {
              if (recordingThread.isAlive()) {
                   Log.w(TAG,"Interrupting recording thread during reset.");
                   recordingThread.interrupt();
              }
              recordingThread = null;
          }
          releaseAudioRecord(); // Ensure recorder is released if still held
          // Clear buffer reference if it wasn't already
          if (recordingBuffer != null) {
                try { recordingBuffer.close(); } catch (IOException e) { /* Ignored */ }
                recordingBuffer = null;
          }
          Log.d(TAG,"Internal recording state reset.");
          // UI update should happen separately via updateUI/mainHandler
          updateUI(null, "Status: Idle"); // Force UI reset if called internally
     }


    // Synchronized method to safely release the AudioRecord instance
    private synchronized void releaseAudioRecord() {
        if (audioRecord != null) {
            Log.d(TAG, "Attempting to release AudioRecord. Current state: " + audioRecord.getState() + ", Recording state: " + audioRecord.getRecordingState());
            try {
                 // Check state before stopping/releasing
                 if (audioRecord.getState() == AudioRecord.STATE_INITIALIZED) {
                     // Only stop if it was actually recording
                     if (audioRecord.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) {
                         try {
                            audioRecord.stop();
                            Log.d(TAG, "AudioRecord stopped in releaseAudioRecord.");
                         } catch (IllegalStateException e) {
                            // This can happen if stop() was already called or called in wrong state
                            Log.e(TAG, "IllegalStateException while stopping AudioRecord (may be expected if already stopped): " + e.getMessage());
                         }
                     }
                     // Release the recorder object
                     audioRecord.release();
                     Log.i(TAG, "AudioRecord released.");
                 } else {
                      Log.w(TAG, "AudioRecord not in initialized state, skipping stop/release. State: " + audioRecord.getState());
                      // If state is UNINITIALIZED but object exists, try releasing anyway
                       if (audioRecord.getState() == AudioRecord.STATE_UNINITIALIZED) {
                            Log.w(TAG, "AudioRecord state is UNINITIALIZED, attempting release anyway.");
                            audioRecord.release();
                            Log.i(TAG, "AudioRecord released from UNINITIALIZED state.");
                       }
                 }
            } catch (Exception e) { // Catch any other potential issues during release
                 Log.e(TAG, "Exception releasing AudioRecord: " + e.getMessage(), e);
            } finally {
                 audioRecord = null; // Ensure it's null after attempting release
            }
        } else {
             // Log.d(TAG, "AudioRecord instance was already null during release attempt."); // Optional log
        }
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permission Granted! Click Start again.", Toast.LENGTH_SHORT).show();
                 // Re-check if model/vocab are ready before enabling
                 boolean modelReady = whisperHelper != null && idToTokenMap != null && !idToTokenMap.isEmpty();
                 buttonStartRecord.setEnabled(modelReady);
            } else {
                Toast.makeText(this, "Permission Denied. Cannot record audio.", Toast.LENGTH_SHORT).show();
                buttonStartRecord.setEnabled(false); // Keep disabled if permission denied
            }
        }
        // Add other permission results handling here if needed
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i(TAG,"onDestroy called.");

        // Stop recording and release resources if activity is destroyed while recording
        if (isRecording) {
            Log.w(TAG, "Activity destroyed while recording. Forcing stop and release.");
            resetRecordingState(); // This handles stopping thread, releasing record, resetting flags
        }

        // Shutdown executor service gracefully
        inferenceExecutorService.shutdown();
        try {
            // Wait a reasonable time for existing tasks to complete
            if (!inferenceExecutorService.awaitTermination(500, java.util.concurrent.TimeUnit.MILLISECONDS)) {
                Log.w(TAG, "Inference executor did not terminate gracefully, forcing shutdown.");
                inferenceExecutorService.shutdownNow(); // Cancel currently executing tasks
            }
        } catch (InterruptedException e) {
            Log.e(TAG,"Interrupted while waiting for executor shutdown.");
            inferenceExecutorService.shutdownNow();
            Thread.currentThread().interrupt();
        }


        if (whisperHelper != null) {
            whisperHelper.close(); // Close the TFLite interpreter
            whisperHelper = null;
        }

         // Remove callbacks from handler to prevent memory leaks
         mainHandler.removeCallbacksAndMessages(null);
         Log.i(TAG,"Activity destroyed, resources released.");
    }
} // End of MainActivity class