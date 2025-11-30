# Chirp Signal Generation and Reception Plan

## Part 1: Enhanced gen-chirp Command

### Requirements
- Allow creating N channel output
- Allow selecting channel of the signal (chirp)
- Allow selecting ref channel (for reference sine wave)
- Allow selecting ref amplitude (in dB)
- Allow selecting signal amplitude (in dB)
- Allow selecting delay between ref signal and chirp signal
- Allow selecting ref signal duration
- Allow selecting ref signal frequency (default 4400 Hz)
- Allow selecting signal duration, start, end frequency
- Allow selecting silence duration before ref and after chirp
- Should create whole buffer at once
- Should generate sinewave as ref in selected channel offset by padding
- Should generate chirp offset relative to sinewave by delay

### Implementation Steps

1. **Add new command-line flags:**
   - `--ref-channel` (int): Channel index for reference signal (default: 0)
   - `--signal-channel` (int): Channel index for chirp signal (default: 0)
   - `--ref-amplitude` (string): Reference signal amplitude in dB (default: "-6")
   - `--ref-duration` (duration): Reference signal duration (default: 100ms)
   - `--ref-frequency` (float64): Reference signal frequency in Hz (default: 4400)
   - `--delay` (duration): Delay between ref signal end and chirp signal start (default: 0)
   - `--silence-before` (duration): Silence duration before ref signal (default: 0)
   - `--silence-after` (duration): Silence duration after chirp signal (default: 0)

2. **Buffer structure calculation:**
   ```
   Total samples = silence_before + ref_duration + delay + chirp_duration + silence_after
   ```

3. **Generation algorithm:**
   - Allocate full buffer (all channels, all samples) initialized to zero
   - Generate reference sine wave in ref_channel starting at silence_before offset
   - Generate chirp signal in signal_channel starting at (silence_before + ref_duration + delay) offset
   - All other channels remain zero (silence)
   - Write entire buffer to WAV file

4. **Code structure:**
   - Parse all new flags
   - Calculate total buffer size
   - Create multi-channel tensor
   - Generate ref sine wave using `dsp.NewSineGenerator` in ref_channel
   - Generate chirp using `dsp.NewChirpGenerator` (or variants) in signal_channel
   - Write tensor to WAV file

## Part 2: recv-chirp Subcommand

### Requirements
- Allow selecting delay between ref signal and chirp signal
- Allow selecting ref signal duration
- Allow selecting ref signal frequency (default 4400 Hz)
- Allow selecting signal duration, start, end frequency
- Should record audio or read from WAV file into small tensor
- Should use DSP operations (Goertzel) to look for ref frequency
- Add this tensor into list of tensors and add next
- Add as many tensors as needed to cover the ref signal, delay, and the chirp plus configurable padding
- Try to find exact position of the ref signal by amplitude (use DSP tools to measure/track)
- Cut out everything before the ref signal
- Use chirp generator to calculate convolution
- Write the calculated convolution to output WAV

### Implementation Steps

1. **Add command-line flags:**
   - `--input` (string): Input WAV file path (alternative to device capture)
   - `--device-id` (string): Capture device ID (optional)
   - `--output` (string): Output WAV file path (required)
   - `--sample-rate` (int): Sample rate in Hz (default: 48000)
   - `--channels` (int): Number of channels (default: 1)
   - `--ref-frequency` (float64): Reference signal frequency in Hz (default: 4400)
   - `--ref-duration` (duration): Reference signal duration (default: 100ms)
   - `--delay` (duration): Delay between ref signal end and chirp signal start (default: 0)
   - `--chirp-duration` (duration): Chirp signal duration
   - `--chirp-freq-start` (float64): Chirp start frequency in Hz
   - `--chirp-freq-end` (float64): Chirp end frequency in Hz
   - `--chirp-type` (string): Chirp type: linear, log, or pink-log (default: linear)
   - `--padding` (duration): Padding after chirp signal (default: 0)
   - `--ref-channel` (int): Channel index to search for reference signal (default: 0)
   - `--signal-channel` (int): Channel index containing chirp signal (default: 0)

2. **Reference signal detection algorithm:**
   - Read audio in small chunks (tensors)
   - For each chunk, extract ref_channel
   - Use `dsp.Goertzel` to measure amplitude at ref_frequency
   - Track amplitude over time to find ref signal start
   - Detection criteria: amplitude exceeds threshold (e.g., 3x noise floor or configurable)
   - Once detected, continue collecting chunks until we have:
     - Ref signal duration
     - Delay duration
     - Chirp duration
     - Padding duration

3. **Signal processing:**
   - Concatenate all collected tensors
   - Extract signal_channel from concatenated data
   - Find exact start of ref signal (refine position using peak detection)
   - Cut everything before ref signal start
   - Extract chirp region: from (ref_end + delay) to (ref_end + delay + chirp_duration)

4. **Convolution calculation:**
   - Create chirp generator matching the expected chirp parameters
   - Use `ChirpGenerator.Correlate` (or equivalent) to compute convolution
   - The correlation will produce a peak at the matched position

5. **Output:**
   - Write convolution result to output WAV file
   - Optionally include metadata about detected positions

### Code Structure

```go
func newRecvChirpCmd() *cobra.Command {
    // Define flags
    // Return command with RunE handler
}

func runRecvChirp(...) error {
    // 1. Parse parameters
    // 2. Create unmarshaller (file or device)
    // 3. Read audio in chunks
    // 4. Detect ref signal using Goertzel
    // 5. Collect required duration
    // 6. Extract and process signal
    // 7. Perform convolution
    // 8. Write output
}
```

### Key Functions Needed

1. **Reference signal detection:**
   - `detectRefSignal(chunks []tensor, refFreq, sampleRate, threshold) (startIndex, endIndex)`
   - Uses sliding window with Goertzel
   - Tracks amplitude over time
   - Returns sample indices

2. **Signal extraction:**
   - `extractSignalChannel(tensor, channelIndex) []float32`
   - `extractChirpRegion(signal, refEnd, delay, chirpDuration, sampleRate) []float32`

3. **Convolution:**
   - Use existing `ChirpGenerator.Correlate` method
   - Or implement custom correlation if needed

### Edge Cases to Handle

- Ref signal not found: return error with helpful message
- Insufficient audio data: return error
- Multiple ref signals: use first detected
- Ref signal amplitude too low: warn user
- Chirp signal corrupted: handle gracefully

