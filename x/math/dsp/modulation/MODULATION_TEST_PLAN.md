# Modulation Encoder/Decoder Roundtrip Test Plan

## Overview
Test PSK, FSK, and QAM encoders/decoders with roundtrip tests that:
1. Encode data using protocol marshaller
2. Add noise to simulate real-world conditions
3. Decode and verify data integrity
4. Test various modulation orders (bits per symbol)

## Test Strategy

### Phase 1: FSK Tests
- **2-FSK (1 bit/symbol)**: Binary FSK with 2 frequencies
- **4-FSK (2 bits/symbol)**: 4 frequencies
- **8-FSK (3 bits/symbol)**: 8 frequencies
- **16-FSK (4 bits/symbol)**: 16 frequencies

### Phase 2: PSK Tests
- **BPSK (1 bit/symbol)**: Binary Phase Shift Keying
- **QPSK (2 bits/symbol)**: Quadrature Phase Shift Keying
- **8-PSK (3 bits/symbol)**: 8 phase states
- **16-PSK (4 bits/symbol)**: 16 phase states

### Phase 3: QAM Tests
- **4-QAM (2 bits/symbol)**: QPSK equivalent
- **16-QAM (4 bits/symbol)**: 16 constellation points
- **64-QAM (6 bits/symbol)**: 64 constellation points (if supported)
- **256-QAM (8 bits/symbol)**: 256 constellation points (if supported)

## Test Parameters

### Signal Generation
- **Sample Rate**: 48000 Hz
- **Carrier Frequency**: 1000 Hz (for PSK/QAM)
- **Symbol Rate**: 1000 baud
- **Padding**: Non-aligned padding (e.g., 0.123s) to test phase synchronization
- **Noise**: -20 dB white noise added to signal

### Protocol
- Use simple protocol marshaller (header + payload + CRC)
- Test with various payload sizes: 10, 50, 100, 500 bytes

### Decoder Requirements
- **Sync Frames**: May need preamble/sync pattern for phase/frequency lock
- **Filtering**: Low-pass filter with cutoff = symbolRate/2
- **Timing Recovery**: Handle non-aligned padding

## Test Structure

### Test Functions
1. `TestFSKRoundtrip_2FSK` - Binary FSK
2. `TestFSKRoundtrip_4FSK` - 4-FSK
3. `TestFSKRoundtrip_8FSK` - 8-FSK
4. `TestFSKRoundtrip_16FSK` - 16-FSK
5. `TestPSKRoundtrip_BPSK` - Binary PSK
6. `TestPSKRoundtrip_QPSK` - Quadrature PSK
7. `TestPSKRoundtrip_8PSK` - 8-PSK
8. `TestPSKRoundtrip_16PSK` - 16-PSK
9. `TestQAMRoundtrip_16QAM` - 16-QAM
10. `TestQAMRoundtrip_64QAM` - 64-QAM (if supported)
11. `TestQAMRoundtrip_256QAM` - 256-QAM (if supported)

### Test Helper Functions
- `generateTestSignal()` - Generate modulated signal with padding and noise
- `addNoise()` - Add white/pink/brown noise to signal
- `addPadding()` - Add non-aligned padding
- `protocolEncode()` - Wrap data with protocol header
- `protocolDecode()` - Extract data from protocol packet
- `verifyRoundtrip()` - Compare original and decoded data

## Implementation Notes

### Sync Frames
- Add preamble before actual data (e.g., alternating 0/1 pattern)
- Helps decoder lock onto carrier phase/frequency
- Preamble length: ~10-20 symbols

### Padding Alignment
- Use fractional padding durations (e.g., 0.123s, 0.456s)
- Ensures decoder handles non-symbol-aligned boundaries
- Tests robustness of timing recovery

### Noise Handling
- Start with -20 dB white noise
- May need to test with different noise levels (-10, -15, -20, -25 dB)
- Test with different noise types (white, pink, brown)

### Decoder Improvements Needed
1. **Phase Lock**: For PSK, need to recover carrier phase
2. **Frequency Lock**: For FSK, need to identify frequency bins
3. **Timing Recovery**: Handle non-aligned symbol boundaries
4. **Frame Sync**: Detect packet start using preamble or protocol header

## Success Criteria
- All roundtrip tests pass with -20 dB noise
- Data integrity: 100% bit accuracy for clean signals
- Acceptable error rate with noise (target: <1% bit error rate at -20 dB)
- Protocol header correctly detected and parsed
- Padding doesn't affect decoding accuracy

## Current Status

### FSK Implementation Status
✅ **Completed:**
1. **Frequency Mapping**: Fixed - bit 0 maps to freqLow (index 0), bit 1 maps to freqHigh (index 1)
2. **Preamble Detection**: Implemented - decoder detects 16-symbol preamble (highest frequency) and only starts decoding after detection
3. **START/END Markers**: Added - encoder generates START marker (4 symbols, lowest frequency) and END marker (4 symbols, lowest frequency) for packet boundaries
4. **Noise Handling**: Decoder does not produce bytes when there is no transmission (only noise/padding)
5. **Preamble Handling**: Preamble can be included in decoded data (higher level code handles packet boundaries)
6. **Low-Pass Filtering**: Integrated low-pass filter for noise reduction before frequency detection
7. **Signal Detection**: Energy-based signal detection with AGC
8. **Symbol Synchronization**: Symbol synchronizer handles symbol boundaries correctly
9. **Padding**: Tests now include 0.5-1s padding after signal to ensure decoder has enough samples
10. **Multi-bit FSK Tests**: Created tests for 4-FSK, 8-FSK, and 16-FSK

🔄 **In Progress:**
- Noise robustness: Decoder needs better handling of noise in preamble detection
- All roundtrip tests currently failing due to preamble detection issues with noise

📋 **Issues to Fix:**
- Preamble detection fails when noise is present (-20dB)
- Decoder produces zeros instead of decoded data when noise interferes
- Need to make preamble detection more robust to noise (lower threshold, more lenient detection)

### FSK Decoder Issues (Resolved)
1. ~~**No Filter Support**~~: ✅ Low-pass filter integrated in decoder initialization
2. ~~**Symbol Timing**~~: ✅ SymbolSynchronizer handles symbol boundaries
3. ~~**Frequency Correlation**~~: ✅ Goertzel algorithm used for robust frequency detection
4. ~~**No Sync Frames**~~: ✅ Preamble detection implemented (16 symbols, requires 12 for lock)

### PSK Decoder Issues
1. **Phase Lock**: No carrier phase recovery - assumes phase starts at 0
2. **Symbol Timing**: Fixed sampling at symbol rate, doesn't handle timing offsets
3. **Padding Handling**: Doesn't skip padding before demodulation
4. **No Sync Frames**: No preamble to establish phase reference

### QAM Decoder Issues
1. **Symbol Timing**: Similar to PSK, needs timing recovery
2. **Constellation Lock**: Needs to establish I/Q reference
3. **Padding Handling**: Doesn't skip padding

## Required Improvements

### 1. Sync Frames / Preamble
- Add preamble generation to encoders (e.g., alternating 0/1 pattern)
- Add preamble detection to decoders
- Use preamble to establish:
  - Phase reference (for PSK/QAM)
  - Frequency reference (for FSK)
  - Symbol timing

### 2. Timing Recovery
- Implement symbol timing recovery to handle non-aligned boundaries
- Use early-late gate or similar algorithm
- Handle fractional symbol offsets

### 3. Phase Recovery (PSK/QAM)
- Implement carrier phase recovery
- Use Costas loop or similar for phase lock
- Handle phase ambiguity (BPSK has 180° ambiguity)

### 4. Filter Support
- Add `SetFilter()` to FSKDecoder
- Ensure all decoders properly filter before symbol decision

### 5. Padding/Signal Detection
- Implement energy-based signal detection
- Skip padding automatically
- Find optimal signal start point

