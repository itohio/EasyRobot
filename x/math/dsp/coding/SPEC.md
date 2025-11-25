# Error Correction Code (ECC) Module Specification

## Overview

The ECC module provides error correction capabilities for digital communication systems. It implements byte-level encoders and decoders that can be composed with other modulation schemes to improve reliability in noisy channels.

## Design Goals

1. **Composability**: Implement `ByteEncoder` and `ByteDecoder` interfaces for seamless integration with existing modulation schemes
2. **Performance**: Optimize hot paths for real-time communication scenarios
3. **Flexibility**: Support multiple ECC schemes with configurable parameters
4. **Reliability**: Provide robust error correction for lossy channels

## Architecture

### Components

1. **Reed-Solomon (RS) Codec**
   - Configurable code parameters (n, k, t)
   - Byte-level encoding/decoding
   - Error detection and correction
   - Systematic encoding (data + parity)

2. **CRC (Cyclic Redundancy Check)**
   - Multiple polynomial support (CRC-8, CRC-16, CRC-32)
   - Byte-level encoding/decoding
   - Error detection (no correction)

3. **Hamming Code**
   - Hamming(7,4) encoding
   - Single-bit error correction
   - Double-bit error detection

4. **Gray Code**
   - Noise-resilient binary encoding
   - Adjacent values differ by only 1 bit
   - Configurable bit width (1-8 bits)
   - 1:1 mapping (no size change)

5. **NRZI (Non-Return-to-Zero Inverted)**
   - Differential encoding: transitions encode data
   - Robust to polarity inversion
   - Simple clock recovery
   - 1:1 mapping (no size change)

6. **4b/5b Encoding**
   - Maps 4-bit nibbles to 5-bit symbols
   - DC balance and clock recovery
   - Used in Fast Ethernet (100BASE-TX)
   - Output size: input_size * 2 (2 bytes per input byte)

7. **8b/10b Encoding**
   - Maps 8-bit bytes to 10-bit symbols
   - DC balance and bounded run length
   - Used in PCIe, SATA, DisplayPort, Gigabit Ethernet
   - Output size: input_size * 2 (2 bytes per input byte)
   - Note: Simplified implementation without running disparity (RD) tracking

8. **Scrambling (LFSR-based)**
   - Pseudo-randomizes data using linear feedback shift registers
   - Reduces long runs, spreads spectrum, reduces EMI
   - Configurable polynomials (7-bit, 15-bit, 23-bit)
   - 1:1 mapping (no size change)

9. **Run-Length Limited (RLL) Codes**
   - Limits consecutive zeros/ones
   - Common variants: RLL(1,7), RLL(2,7)
   - Used in magnetic storage
   - Variable size expansion (~1.5x for RLL(1,7))
   - Note: Simplified implementation

10. **Differential Manchester (Biphase-M)**
    - Combines differential and Manchester encoding
    - Transition at start, '0' = transition in middle, '1' = no transition in middle
    - Clock recovery and polarity-independent
    - Output size: input_size * 2 (2 bytes per input byte)

11. **Miller Encoding (Delay Encoding)**
    - '1' = transition in middle, '0' = no transition (or transition at start if previous was '0')
    - Better spectral efficiency than Manchester
    - Used in RFID (ISO 18000-6)
    - Output size: input_size * 2 (2 bytes per input byte)

12. **AMI (Alternate Mark Inversion)**
    - '0' = no signal, '1' = alternating polarity
    - DC balance and error detection
    - Used in T1/E1 lines
    - 1:1 mapping (no size change)
    - Note: Simplified byte-level implementation

13. **HDB3 (High Density Bipolar 3)**
    - AMI variant that limits consecutive zeros to maximum of 3
    - Used in E1 lines (European telecom)
    - 1:1 mapping (no size change)
    - Note: Simplified implementation

14. **B8ZS (Bipolar 8-Zero Substitution)**
    - AMI variant for T1 lines
    - Substitutes 8 consecutive zeros with special pattern
    - Used in T1 lines (North American telecom)
    - 1:1 mapping (no size change)
    - Note: Simplified implementation

15. **Biphase-L (Biphase Level)**
    - Similar to Manchester with different transition rules
    - '0' = transition at start and middle, '1' = transition only at start
    - Clock recovery and DC balance
    - Output size: input_size * 2 (2 bytes per input byte)

16. **PPM (Pulse Position Modulation)**
    - Encodes data in pulse timing/position
    - Common variants: 2-PPM, 4-PPM, 8-PPM
    - Power efficient, good for optical/RF
    - Used in IR communication, some RF protocols
    - Note: Simplified byte-level implementation

3. **Utility Functions**
   - RS parameter calculation helpers
   - Code rate and overhead calculations
   - Error correction capability estimation

## Interfaces

### ByteEncoder Interface

```go
type ByteEncoder interface {
    Encode(dst, src []byte) int
    Reset()
}
```

### ByteDecoder Interface

```go
type ByteDecoder interface {
    Decode(dst, src []byte) int
    Reset()
}
```

## Reed-Solomon Implementation

### Parameters

- **n**: Total codeword length (data + parity symbols)
- **k**: Data symbols per codeword
- **t**: Maximum correctable errors (t = (n-k)/2)
- **m**: Symbol size in bits (typically 8 for byte-level)

### Encoding

- Input: k data bytes
- Output: n bytes (k data + (n-k) parity)
- Systematic: Original data preserved, parity appended

### Decoding

- Input: n bytes (potentially corrupted)
- Output: k data bytes (corrected if errors ≤ t)
- Error detection: Identifies if errors are present
- Error correction: Corrects up to t errors

### Utility Functions

- `CalculateRSParams(dataLen, maxErrors int) (n, k, t int)`: Calculate optimal RS parameters
- `CalculateCodeRate(n, k int) float32`: Calculate code rate (k/n)
- `CalculateOverhead(n, k int) float32`: Calculate overhead percentage
- `MaxCorrectableErrors(n, k int) int`: Maximum correctable errors

## CRC Implementation

### Parameters

- **Polynomial**: CRC polynomial (e.g., CRC-8, CRC-16, CRC-32)
- **Initial Value**: Starting CRC value
- **XOR Out**: Final XOR value

### Encoding

- Input: Data bytes
- Output: Data bytes + CRC bytes
- Systematic: Original data preserved, CRC appended

### Decoding

- Input: Data bytes + CRC bytes
- Output: Data bytes (if CRC valid)
- Error detection: Validates CRC, returns error if invalid

## Usage Patterns

### Composition with Modulation

```go
// Encode: data -> RS encoder -> Manchester encoder -> FSK modulator
rsEncoder := ecc.NewRSEncoder(n, k)
manchesterEncoder := manchester.NewEncoder()
fskModulator := mfsk.NewModulator(...)

// Decode: FSK demodulator -> Manchester decoder -> RS decoder -> data
fskDemodulator := mfsk.NewDemodulator(...)
manchesterDecoder := manchester.NewDecoder()
rsDecoder := ecc.NewRSDecoder(n, k)
```

### Standalone Usage

```go
// Encode
data := []byte{0x12, 0x34, 0x56}
encoded := make([]byte, rsEncoder.Encode(nil, data))
rsEncoder.Encode(encoded, data)

// Decode (with potential errors)
decoded := make([]byte, len(data))
bytesDecoded := rsDecoder.Decode(decoded, encoded)
```

## Testing Requirements

1. **Unit Tests**
   - Encoder/decoder basic functionality
   - Parameter validation
   - Edge cases (empty input, single byte, etc.)

2. **Roundtrip Tests**
   - Perfect channel (no errors)
   - Verify data integrity
   - Multiple data sizes

3. **Lossy Roundtrip Tests**
   - Introduce random byte errors
   - Test error correction capability
   - Measure correction success rate
   - Test with various error rates

## Performance Considerations

- **Hot Path Optimization**: Avoid type casts in encoding/decoding loops
- **Memory Allocation**: Pre-allocate buffers where possible
- **Lookup Tables**: Use tables for Galois field operations (RS)
- **Batch Processing**: Process multiple codewords efficiently

## Error Handling

- **Invalid Parameters**: Panic on construction with invalid parameters
- **Decode Errors**: Return 0 bytes written if correction fails
- **Buffer Size**: Return 0 if destination buffer too small
- **State Management**: Reset() clears internal state

## Bit-Level Encoding Schemes

### Gray Code

**Purpose**: Noise resilience - adjacent values differ by only 1 bit, reducing multi-bit errors.

**Usage**:
```go
encoder := ecc.NewGrayEncoder() // 8-bit default
encoder := ecc.NewGrayEncoderWithBits(4) // 4-bit (nibble-level)
decoder := ecc.NewGrayDecoder()
decoder := ecc.NewGrayDecoderWithBits(4)
```

**Known Quirks**:
- 1:1 mapping (no size change)
- Configurable bit width (1-8 bits)
- Most effective when used with modulation schemes where symbol errors are likely to be adjacent

### NRZI (Non-Return-to-Zero Inverted)

**Purpose**: Robust to polarity inversion, simple clock recovery.

**Usage**:
```go
encoder := ecc.NewNRZIEncoder()
decoder := ecc.NewNRZIDecoder()
```

**Known Quirks**:
- 1:1 mapping (no size change)
- Stateful: maintains last output state
- Long runs of zeros can cause clock recovery issues (consider combining with scrambling)

### 4b/5b Encoding

**Purpose**: DC balance, clock recovery, bounded run length.

**Usage**:
```go
encoder := ecc.NewFourB5BEncoder()
decoder := ecc.NewFourB5BDecoder()
```

**Known Quirks**:
- Output size: input_size * 2 (2 bytes per input byte)
- Each 4-bit nibble becomes 5-bit symbol (stored in byte)
- Ensures no more than 3 consecutive zeros
- Used in Fast Ethernet (100BASE-TX), FDDI

### 8b/10b Encoding

**Purpose**: DC balance, bounded run length, embedded clock.

**Usage**:
```go
encoder := ecc.NewEightB10BEncoder()
decoder := ecc.NewEightB10BDecoder()
```

**Known Quirks**:
- Output size: input_size * 2 (2 bytes per input byte)
- Each 8-bit byte becomes 10-bit symbol
- **Simplified implementation**: Does not track running disparity (RD)
- Full DC balance requires RD tracking to alternate between positive/negative disparity symbols
- Used in PCIe, SATA, DisplayPort, Gigabit Ethernet
- For production use, consider full RD-tracking implementation

### Scrambling (LFSR-based)

**Purpose**: Reduces long runs, spreads spectrum, reduces EMI.

**Usage**:
```go
encoder := ecc.NewScramblerEncoder() // Default 7-bit
encoder := ecc.NewScramblerEncoderWithPolynomial(poly, width) // Custom
decoder := ecc.NewScramblerDecoder() // Must match encoder polynomial
```

**Known Quirks**:
- 1:1 mapping (no size change)
- Encoder and decoder must use same polynomial and initial state
- Common polynomials:
  - 7-bit: x^7 + x^4 + 1 (polynomial 0x91)
  - 15-bit: x^15 + x^14 + 1 (polynomial 0xC001)
  - 23-bit: x^23 + x^18 + 1 (polynomial 0x40041)
- Stateful: maintains LFSR state
- Widely used in Ethernet, Wi-Fi, and many digital standards

### Run-Length Limited (RLL) Codes

**Purpose**: Limits consecutive zeros/ones for clock recovery.

**Usage**:
```go
encoder := ecc.NewRLLEncoder() // RLL(1,7) default
encoder := ecc.NewRLLEncoderWithParams(d, k) // Custom (d,k)
decoder := ecc.NewRLLDecoder() // Must match encoder parameters
```

**Known Quirks**:
- Variable size expansion (~1.5x for RLL(1,7))
- Common variants: RLL(1,7), RLL(2,7)
- **Simplified implementation**: Full RLL requires complex state machines and lookup tables
- Used in magnetic storage (hard drives, floppy disks)
- For production use, consider full RLL implementation with proper encoding tables

### Differential Manchester (Biphase-M)

**Purpose**: Clock recovery, polarity-independent, combines differential and Manchester.

**Usage**:
```go
encoder := ecc.NewDiffManchesterEncoder()
decoder := ecc.NewDiffManchesterDecoder()
```

**Known Quirks**:
- Output size: input_size * 2 (2 bytes per input byte)
- Stateful: maintains last output state
- Used in Token Ring, some industrial protocols
- More complex than standard Manchester

### Miller Encoding (Delay Encoding)

**Purpose**: Better spectral efficiency than Manchester while maintaining clock recovery.

**Usage**:
```go
encoder := ecc.NewMillerEncoder()
decoder := ecc.NewMillerDecoder()
```

**Known Quirks**:
- Output size: input_size * 2 (2 bytes per input byte)
- Stateful: maintains last state and last bit
- Used in RFID (ISO 18000-6), some telemetry
- '0' encoding depends on previous bit value

### AMI (Alternate Mark Inversion)

**Purpose**: DC balance, error detection (violations indicate errors).

**Usage**:
```go
encoder := ecc.NewAMIEncoder()
decoder := ecc.NewAMIDecoder()
```

**Known Quirks**:
- 1:1 mapping (no size change)
- 3-level encoding: 0, +1, -1 (simplified to binary in byte-level implementation)
- Stateful: maintains polarity for alternating marks
- Used in T1/E1 lines, some legacy telecom
- **Simplified implementation**: Full AMI requires 3-level signal detection
- Long runs of zeros can cause clock recovery issues

### HDB3 (High Density Bipolar 3)

**Purpose**: AMI variant that limits consecutive zeros to maximum of 3.

**Usage**:
```go
encoder := ecc.NewHDB3Encoder()
decoder := ecc.NewHDB3Decoder()
```

**Known Quirks**:
- 1:1 mapping (no size change)
- Stateful: maintains polarity and zero count
- Used in E1 lines (European telecom)
- Inserts violations (B00V or 000V patterns) when 4 consecutive zeros detected
- **Simplified implementation**: Full HDB3 requires proper violation pattern detection

### B8ZS (Bipolar 8-Zero Substitution)

**Purpose**: AMI variant for T1 lines, substitutes 8 consecutive zeros.

**Usage**:
```go
encoder := ecc.NewB8ZSEncoder()
decoder := ecc.NewB8ZSDecoder()
```

**Known Quirks**:
- 1:1 mapping (no size change)
- Stateful: maintains polarity and zero count
- Used in T1 lines (North American telecom)
- Substitutes 8 consecutive zeros with 000VB0VB pattern
- **Simplified implementation**: Full B8ZS requires proper violation pattern detection

### Biphase-L (Biphase Level)

**Purpose**: Clock recovery, DC balance, similar to Manchester.

**Usage**:
```go
encoder := ecc.NewBiphaseLEncoder()
decoder := ecc.NewBiphaseLDecoder()
```

**Known Quirks**:
- Output size: input_size * 2 (2 bytes per input byte)
- Stateful: maintains last output state
- Used in some industrial protocols
- Different transition rules than standard Manchester

### PPM (Pulse Position Modulation)

**Purpose**: Power efficient, good for optical/RF communication.

**Usage**:
```go
encoder := ecc.NewPPMEncoder() // 4-PPM default
encoder := ecc.NewPPMEncoderWithPositions(positions) // 2, 4, or 8 positions
decoder := ecc.NewPPMDecoder() // Must match encoder positions
```

**Known Quirks**:
- 1:1 mapping in simplified implementation (full PPM would expand)
- Configurable positions: 2-PPM, 4-PPM, 8-PPM
- Used in IR communication, some RF protocols
- **Simplified implementation**: Full PPM requires timing information and would expand data size
- For production use, consider full PPM implementation with proper timing

## Usage Recommendations

### For Noise Resilience
- **Gray Code**: Best for reducing multi-bit errors when symbol misdetection is likely
- **Scrambling**: Best for reducing long runs and spreading spectrum

### For Clock Recovery
- **Manchester/Differential Manchester**: Excellent clock recovery, 2x size expansion
- **4b/5b/8b/10b**: Good clock recovery with DC balance, 1.25x-2x size expansion
- **RLL**: Limits run lengths, variable expansion

### For DC Balance
- **4b/5b**: Good DC balance, moderate expansion
- **8b/10b**: Excellent DC balance (with RD tracking), 2x expansion
- **AMI/HDB3/B8ZS**: DC balance with 3-level encoding

### For Polarity Robustness
- **NRZI**: Robust to polarity inversion
- **Differential Manchester**: Polarity-independent with clock recovery

### For Spectral Efficiency
- **Miller**: Better than Manchester, maintains clock recovery
- **Scrambling**: Spreads spectrum, reduces EMI

## Composition Examples

```go
// Example 1: Noise-resilient communication
grayEncoder := ecc.NewGrayEncoder()
scrambler := ecc.NewScramblerEncoder()
// Chain: data -> Gray -> Scrambler -> Modulator

// Example 2: Clock recovery with DC balance
fourB5BEncoder := ecc.NewFourB5BEncoder()
// Chain: data -> 4b/5b -> Modulator

// Example 3: Polarity-robust with clock recovery
nrziEncoder := ecc.NewNRZIEncoder()
// Chain: data -> NRZI -> Modulator

// Example 4: Full stack with error correction
grayEncoder := ecc.NewGrayEncoder()
rsEncoder := ecc.NewRSEncoder(n, k)
// Chain: data -> Gray -> RS -> Modulator
```

## Future Enhancements

1. Full 8b/10b with running disparity (RD) tracking for true DC balance
2. Full RLL implementation with proper encoding tables
3. Full PPM implementation with timing information
4. Additional ECC schemes (BCH, LDPC)
5. Adaptive error correction based on channel conditions
6. Interleaving support for burst error correction
7. Soft-decision decoding for improved performance
8. Full 3-level detection for AMI/HDB3/B8ZS

