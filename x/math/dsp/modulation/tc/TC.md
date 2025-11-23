# Spec: SMPTE TimeCode Modulator / Demodulator

# 0. Overview

SMPTE TimeCode = industry-standard timecode format used in audio/video production for frame-accurate synchronization.

The SMPTE TimeCode system:
- Encodes time in HH:MM:SS:FF format (Hours:Minutes:Seconds:Frames)
- Uses BCD (Binary Coded Decimal) encoding - 4 bits per digit
- Supports multiple frame rates: 24, 25, 29.97, 30 fps
- Supports drop-frame and non-drop-frame formats
- Includes user bits for additional metadata
- For LTC (Linear Timecode): uses bi-phase mark encoding for audio transmission

This spec implements SMPTE 12M standard for timecode encoding/decoding.

---

# 1. SMPTE TimeCode Frame Format

## 1.1 TimeCode Structure

SMPTE timecode represents time as:
- **HH**: Hours (00-23, BCD encoded, 2 digits = 8 bits)
- **MM**: Minutes (00-59, BCD encoded, 2 digits = 8 bits)
- **SS**: Seconds (00-59, BCD encoded, 2 digits = 8 bits)
- **FF**: Frames (00-29/23/24 depending on frame rate, BCD encoded, 2 digits = 8 bits)

## 1.2 Frame Rates

Supported frame rates:
- **24 fps**: Cinema standard
- **25 fps**: PAL standard (Europe)
- **29.97 fps**: NTSC standard (drop-frame)
- **30 fps**: Non-drop-frame standard

## 1.3 Drop Frame vs Non-Drop Frame

- **Non-drop-frame**: Each frame is counted sequentially
- **Drop-frame**: Frames 00 and 01 are dropped every minute except minutes divisible by 10
  - Used for 29.97 fps to maintain sync with wall-clock time
  - Format indicator: bit flag in timecode

## 1.4 LTC (Linear Timecode) Bit Structure

LTC frame consists of 80 bits transmitted over 1 frame period:

**Bit Layout (LSB to MSB, transmitted left to right):**

```
Bits 0-3:   Frame units (0-9, BCD)
Bit 4:      Drop frame flag
Bit 5:      Color frame flag
Bits 6-9:   Frame tens (0-2, BCD)
Bits 10-13: Second units (0-9, BCD)
Bit 14:     Unassigned
Bits 15-18: Second tens (0-5, BCD)
Bits 19-22: Minute units (0-9, BCD)
Bit 23:     Unassigned
Bits 24-27: Minute tens (0-5, BCD)
Bits 28-31: Hour units (0-9, BCD)
Bit 32:     Unassigned
Bits 33-34: Hour tens (0-2, BCD)
Bits 35-58: User bits (24 bits, optional)
Bits 59-78: Sync word (fixed pattern: 0011 1111 1111 1101)
Bit 79:     Unassigned
```

**Sync Word Pattern:**
- Binary: `0011 1111 1111 1101`
- Hex: `0x3FFD`
- Used for frame synchronization

## 1.5 BCD Encoding

Each decimal digit (0-9) is encoded as 4-bit binary:
- 0 = 0000
- 1 = 0001
- 2 = 0010
- 3 = 0011
- 4 = 0100
- 5 = 0101
- 6 = 0110
- 7 = 0111
- 8 = 1000
- 9 = 1001

Invalid BCD values (1010-1111) indicate corruption.

## 1.6 User Bits

24 bits available for user-defined data:
- Can store device ID, scene/take numbers, etc.
- Optional - can be set to zero if not used

---

# 2. LTC Encoding (Bi-Phase Mark)

For Linear Timecode (LTC), the 80-bit frame is encoded using bi-phase mark encoding:

## 2.1 Bi-Phase Mark Rules

- **Bit 0 (low)**: Transition at start of bit period, no transition at middle
- **Bit 1 (high)**: Transition at start AND middle of bit period
- **Clock recovery**: Transitions provide clock information

## 2.2 Audio Signal

LTC is typically transmitted as an audio signal:
- **Frequency**: 2400 Hz for 30 fps (scales with frame rate)
- **Bandwidth**: ~2.4 kHz
- **Amplitude**: Typically -10 to -20 dBV
- **Waveform**: Square wave with bi-phase mark encoding

## 2.3 Sample Rate Considerations

For audio transmission:
- Standard sample rates: 44.1 kHz, 48 kHz, 96 kHz
- One LTC frame = 80 bits × 2 samples/bit (bi-phase) = 160 samples minimum
- Frame period = 1 / frame_rate seconds

---

# 3. TimeCode Data Structure

## 3.1 SMPTETimeCode Struct

```go
type SMPTETimeCode struct {
    Hours   int    // 0-23
    Minutes int    // 0-59
    Seconds int    // 0-59
    Frames  int    // 0-29 (depends on frame rate)
    
    FrameRate   FrameRate  // 24, 25, 29.97, 30
    DropFrame   bool       // Drop-frame flag
    ColorFrame  bool       // Color frame flag
    UserBits    [3]byte    // 24 bits user data
}
```

## 3.2 Frame Rate Type

```go
type FrameRate float32

const (
    FrameRate24   FrameRate = 24.0
    FrameRate25   FrameRate = 25.0
    FrameRate2997 FrameRate = 29.97
    FrameRate30   FrameRate = 30.0
)
```

---

# 4. Encoding Functions

## 4.1 BCD Encoding

- `EncodeBCD(digit int) byte`: Encode single digit (0-9) to 4-bit BCD
- `DecodeBCD(bcd byte) (int, error)`: Decode 4-bit BCD to digit, validate

## 4.2 TimeCode to Bits

- `MarshalLTC(tc *SMPTETimeCode) []byte`: Convert timecode to 80-bit LTC frame
- Bit ordering: LSB first (as transmitted in LTC)

## 4.3 Bits to TimeCode

- `UnmarshalLTC(data []byte) (*SMPTETimeCode, error)`: Parse 80-bit LTC frame
- Validates sync word, BCD values, frame ranges

---

# 5. Bi-Phase Mark Encoding

## 5.1 Bi-Phase Encoder

- `EncodeBiPhase(bits []byte, sampleRate float32, frameRate FrameRate) []float32`
- Converts 80-bit frame to audio samples using bi-phase mark encoding
- Returns samples for one frame period

## 5.2 Bi-Phase Decoder

- `DecodeBiPhase(samples []float32, sampleRate float32, frameRate FrameRate) ([]byte, error)`
- Recovers bits from bi-phase mark encoded audio samples
- Performs clock recovery and bit decision

---

# 6. Time Utilities

## 6.1 Time Conversion

- `FromTime(t time.Time, frameRate FrameRate, dropFrame bool) *SMPTETimeCode`
- `ToTime(tc *SMPTETimeCode) time.Time`
- `AddFrames(tc *SMPTETimeCode, frames int) *SMPTETimeCode`
- `SubtractFrames(tc *SMPTETimeCode, frames int) *SMPTETimeCode`

## 6.2 String Formatting

- `String() string`: Format as "HH:MM:SS:FF"
- `Parse(s string, frameRate FrameRate) (*SMPTETimeCode, error)`

## 6.3 Drop Frame Calculation

- `CalculateDropFrames(hours, minutes int, frameRate FrameRate) int`
- Calculates number of frames to drop for drop-frame timecode

---

# 7. Modulator/Demodulator Integration

## 7.1 LTC Modulator

- Uses bi-phase mark encoder
- Generates audio samples at specified sample rate
- Continuous frame generation for streaming

## 7.2 LTC Demodulator

- Detects sync word for frame alignment
- Decodes bi-phase mark to recover bits
- Validates and parses timecode frames
- Handles clock recovery and jitter

---

# 8. Implementation Notes

- BCD validation: Reject invalid BCD values (0xA-0xF)
- Sync word detection: Use correlation for robust detection
- Frame rate handling: Support all standard frame rates
- Drop frame: Correctly handle frame dropping logic
- User bits: Optional, can be zero
- Bi-phase encoding: Ensure proper transitions for clock recovery
- Sample rate: Must be integer multiple of frame rate for clean encoding

---

# 9. References

- SMPTE 12M: Time and Control Code
- SMPTE 259M: SDI (for VITC)
- EBU Tech 3097: Timecode for Television
- Wikipedia: SMPTE timecode, Linear timecode

---

### End of spec
