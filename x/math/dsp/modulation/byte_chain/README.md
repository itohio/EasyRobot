# Byte Chain Package

The `byte_chain` package provides composable chaining of `ByteEncoder` and `ByteDecoder` interfaces, allowing you to combine multiple encoding/decoding layers easily.

## Overview

This package allows you to chain multiple byte-level encoders/decoders together in a pipeline, similar to how the compose functionality in the `modulation` package chains `ByteEncoder` with `Modulator`. The key difference is that `byte_chain` works purely at the byte level, allowing multiple `ByteEncoder`/`ByteDecoder` implementations to be composed.

## Usage Example

### Chaining Multiple Encoders

```go
import (
    "github.com/itohio/EasyRobot/x/math/dsp/modulation/byte_chain"
    "github.com/itohio/EasyRobot/x/math/dsp/modulation/ecc"
    "github.com/itohio/EasyRobot/x/math/dsp/modulation/manchester"
)

// Create individual encoders
encryptionEncoder := // ... your encryption encoder (implements ByteEncoder)
rsEncoder := ecc.NewRSEncoder(255, 223)  // Reed-Solomon error correction
manchesterEncoder := manchester.NewEncoder()  // Manchester line coding

// Chain them together (order matters!)
// Flow: data -> encryption -> reed solomon -> manchester -> output
chainEncoder := byte_chain.NewChainEncoder(
    encryptionEncoder,
    rsEncoder,
    manchesterEncoder,
)

// Encode data
data := []byte{0x12, 0x34, 0x56, 0x78}
expectedSize := chainEncoder.Encode(nil, data)
output := make([]byte, expectedSize)
bytesWritten := chainEncoder.Encode(output, data)
```

### Chaining Multiple Decoders (Reverse Order)

```go
// Create decoders in REVERSE order of encoders
manchesterDecoder := manchester.NewDecoder()
rsDecoder := ecc.NewRSDecoder(255, 223)
decryptionDecoder := // ... your decryption decoder (implements ByteDecoder)

// Chain them together (reverse order!)
// Flow: data -> manchester -> reed solomon -> decryption -> output
chainDecoder := byte_chain.NewChainDecoder(
    manchesterDecoder,
    rsDecoder,
    decryptionDecoder,
)

// Decode data
decoded := make([]byte, expectedSize)
bytesDecoded := chainDecoder.Decode(decoded, output)
```

### Integration with Compose Package

The `byte_chain` can be used together with the `compose` package:

```go
import (
    "github.com/itohio/EasyRobot/x/math/dsp/modulation"
    "github.com/itohio/EasyRobot/x/math/dsp/modulation/byte_chain"
    "github.com/itohio/EasyRobot/x/math/dsp/modulation/mfsk"
    // ... other imports
)

// Create byte encoder chain: encryption -> reed solomon -> manchester
byteChain := byte_chain.NewChainEncoder(
    encryptionEncoder,
    rsEncoder,
    manchesterEncoder,
)

// Create signal modulator: BFSK
bfskModulator := mfsk.NewModulator(sampleRate, freqLow, freqHigh, symbolRate)

// Compose: byte chain -> signal modulator
// Full flow: data -> encryption -> RS -> manchester -> BFSK -> signal
fullEncoder := modulation.NewComposeEncoder(byteChain, bfskModulator)
```

## Design Principles

1. **Abstract and Interface-Based**: Works only with `modulation.ByteEncoder` and `modulation.ByteDecoder` interfaces, not concrete types.

2. **Order Matters**: 
   - Encoders are applied in the order provided
   - Decoders must be provided in reverse order of encoders

3. **Composable**: Can be composed with other encoders/decoders or used with the `compose` package.

4. **Efficient**: Uses intermediate buffers that are reused across calls to minimize allocations.

## Interface Compliance

Both `ChainEncoder` and `ChainDecoder` implement the `ByteEncoder` and `ByteDecoder` interfaces respectively, making them first-class composable components:

- `ChainEncoder` implements `modulation.ByteEncoder`
- `ChainDecoder` implements `modulation.ByteDecoder`

This means you can use a `ChainEncoder` anywhere a `ByteEncoder` is expected, including with the compose functionality in the `modulation` package.

## Common Patterns

### Error Correction + Line Coding
```go
rsEncoder := ecc.NewRSEncoder(255, 223)
manchesterEncoder := manchester.NewEncoder()
chain := byte_chain.NewChainEncoder(rsEncoder, manchesterEncoder)
```

### Encryption + Error Correction + Line Coding
```go
encryptionEncoder := // Your encryption ByteEncoder
rsEncoder := ecc.NewRSEncoder(255, 223)
manchesterEncoder := manchester.NewEncoder()
chain := byte_chain.NewChainEncoder(encryptionEncoder, rsEncoder, manchesterEncoder)
```

### CRC + Manchester
```go
crcEncoder := ecc.NewCRCEncoder()
manchesterEncoder := manchester.NewEncoder()
chain := byte_chain.NewChainEncoder(crcEncoder, manchesterEncoder)
```

## See Also

- `modulation.NewComposeEncoder` / `modulation.NewComposeDecoder` - Composes `ByteEncoder` with `Modulator`
- `modulation/interfaces.go` - Defines `ByteEncoder` and `ByteDecoder` interfaces

