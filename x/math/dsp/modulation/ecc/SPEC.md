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
   - Multiple polynomial support
   - Byte-level encoding/decoding
   - Error detection (no correction)

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

## Future Enhancements

1. Additional ECC schemes (BCH, LDPC)
2. Adaptive error correction based on channel conditions
3. Interleaving support for burst error correction
4. Soft-decision decoding for improved performance

