package tc

import (
	"errors"
	"fmt"
	"time"
)

// FrameRate represents SMPTE frame rates
type FrameRate float32

const (
	// FrameRate24 is 24 fps (cinema standard)
	FrameRate24 FrameRate = 24.0
	// FrameRate25 is 25 fps (PAL standard)
	FrameRate25 FrameRate = 25.0
	// FrameRate2997 is 29.97 fps (NTSC drop-frame)
	FrameRate2997 FrameRate = 29.97
	// FrameRate30 is 30 fps (non-drop-frame)
	FrameRate30 FrameRate = 30.0
)

// SMPTETimeCode represents SMPTE timecode in HH:MM:SS:FF format
type SMPTETimeCode struct {
	// Time components (BCD encoded in bit representation)
	Hours   int // 0-23
	Minutes int // 0-59
	Seconds int // 0-59
	Frames  int // 0-29 (depends on frame rate)

	// Frame rate
	FrameRate FrameRate

	// Flags
	DropFrame  bool // Drop-frame flag (for 29.97 fps)
	ColorFrame bool // Color frame flag

	// User bits (24 bits = 3 bytes)
	UserBits [3]byte
}

// LTCFrameSize is the number of bits in an LTC frame
const LTCFrameSize = 80

// LTCSyncWord is the sync word pattern for LTC frames
// Binary: 0011 1111 1111 1101 (MSB first, transmitted LSB first)
// When stored LSB first in bytes: 1011 1111 1111 1100 = 0xBFFC
// But we need to reverse bits for transmission, so we store as 0x3FFD
const LTCSyncWord = uint32(0x3FFD)

// EncodeBCD encodes a decimal digit (0-9) to 4-bit BCD
func EncodeBCD(digit int) (byte, error) {
	if digit < 0 || digit > 9 {
		return 0, fmt.Errorf("digit %d out of range [0-9]", digit)
	}
	return byte(digit), nil
}

// DecodeBCD decodes a 4-bit BCD value to decimal digit
// Returns error if BCD value is invalid (0xA-0xF)
func DecodeBCD(bcd byte) (int, error) {
	digit := int(bcd & 0x0F)
	if digit > 9 {
		return 0, fmt.Errorf("invalid BCD value: 0x%X", bcd)
	}
	return digit, nil
}

// setBit sets a bit at the given position (0-79) in the frame
// Bits are stored LSB first in bytes
func setBit(frame []byte, bitPos int, value bool) {
	byteIdx := bitPos / 8
	bitIdx := bitPos % 8
	if value {
		frame[byteIdx] |= 1 << bitIdx
	} else {
		frame[byteIdx] &^= 1 << bitIdx
	}
}

// getBit gets a bit at the given position (0-79) from the frame
func getBit(frame []byte, bitPos int) bool {
	byteIdx := bitPos / 8
	bitIdx := bitPos % 8
	return (frame[byteIdx] & (1 << bitIdx)) != 0
}

// setBCD sets a 4-bit BCD value at the given bit position
func setBCD(frame []byte, bitPos int, value int) error {
	bcd, err := EncodeBCD(value)
	if err != nil {
		return err
	}
	for i := 0; i < 4; i++ {
		setBit(frame, bitPos+i, (bcd&(1<<i)) != 0)
	}
	return nil
}

// getBCD gets a 4-bit BCD value at the given bit position
func getBCD(frame []byte, bitPos int) (int, error) {
	var bcd byte
	for i := 0; i < 4; i++ {
		if getBit(frame, bitPos+i) {
			bcd |= 1 << i
		}
	}
	return DecodeBCD(bcd)
}

// MarshalLTC converts SMPTE timecode to 80-bit LTC frame
// Bits are transmitted LSB first (bit 0 first, bit 79 last)
// Returns 10 bytes (80 bits)
func MarshalLTC(tc *SMPTETimeCode) ([]byte, error) {
	// Validate timecode values
	if err := tc.Validate(); err != nil {
		return nil, err
	}

	frame := make([]byte, 10) // 80 bits = 10 bytes

	// Frame units (bits 0-3)
	if err := setBCD(frame, 0, tc.Frames%10); err != nil {
		return nil, err
	}

	// Drop frame flag (bit 4)
	setBit(frame, 4, tc.DropFrame)

	// Color frame flag (bit 5)
	setBit(frame, 5, tc.ColorFrame)

	// Frame tens (bits 6-9)
	if err := setBCD(frame, 6, tc.Frames/10); err != nil {
		return nil, err
	}

	// Second units (bits 10-13)
	if err := setBCD(frame, 10, tc.Seconds%10); err != nil {
		return nil, err
	}

	// Bit 14: Unassigned (set to 0)
	setBit(frame, 14, false)

	// Second tens (bits 15-18)
	if err := setBCD(frame, 15, tc.Seconds/10); err != nil {
		return nil, err
	}

	// Minute units (bits 19-22)
	if err := setBCD(frame, 19, tc.Minutes%10); err != nil {
		return nil, err
	}

	// Bit 23: Unassigned (set to 0)
	setBit(frame, 23, false)

	// Minute tens (bits 24-27)
	if err := setBCD(frame, 24, tc.Minutes/10); err != nil {
		return nil, err
	}

	// Hour units (bits 28-31)
	if err := setBCD(frame, 28, tc.Hours%10); err != nil {
		return nil, err
	}

	// Bit 32: Unassigned (set to 0)
	setBit(frame, 32, false)

	// Hour tens (bits 33-34) - only 2 bits needed (0-2)
	hourTens := tc.Hours / 10
	setBit(frame, 33, (hourTens&1) != 0)
	setBit(frame, 34, (hourTens&2) != 0)

	// User bits (bits 35-58, 24 bits)
	for i := 0; i < 24; i++ {
		byteIdx := i / 8
		bitIdx := i % 8
		setBit(frame, 35+i, (tc.UserBits[byteIdx]&(1<<bitIdx)) != 0)
	}

	// Sync word (bits 59-78): 0011 1111 1111 1101
	// Stored LSB first: bit 59 = LSB of sync word
	syncWord := LTCSyncWord
	for i := 0; i < 20; i++ {
		setBit(frame, 59+i, (syncWord&(1<<i)) != 0)
	}

	// Bit 79: Unassigned (set to 0)
	setBit(frame, 79, false)

	return frame, nil
}

// UnmarshalLTC parses 80-bit LTC frame to SMPTE timecode
func UnmarshalLTC(data []byte) (*SMPTETimeCode, error) {
	if len(data) < 10 {
		return nil, errors.New("LTC frame must be at least 10 bytes (80 bits)")
	}

	tc := &SMPTETimeCode{}

	// Frame units (bits 0-3)
	frameUnits, err := getBCD(data, 0)
	if err != nil {
		return nil, fmt.Errorf("invalid frame units BCD: %w", err)
	}

	// Drop frame flag (bit 4)
	tc.DropFrame = getBit(data, 4)

	// Color frame flag (bit 5)
	tc.ColorFrame = getBit(data, 5)

	// Frame tens (bits 6-9)
	frameTens, err := getBCD(data, 6)
	if err != nil {
		return nil, fmt.Errorf("invalid frame tens BCD: %w", err)
	}
	tc.Frames = frameTens*10 + frameUnits

	// Second units (bits 10-13)
	secondUnits, err := getBCD(data, 10)
	if err != nil {
		return nil, fmt.Errorf("invalid second units BCD: %w", err)
	}

	// Second tens (bits 15-18)
	secondTens, err := getBCD(data, 15)
	if err != nil {
		return nil, fmt.Errorf("invalid second tens BCD: %w", err)
	}
	tc.Seconds = secondTens*10 + secondUnits

	// Minute units (bits 19-22)
	minuteUnits, err := getBCD(data, 19)
	if err != nil {
		return nil, fmt.Errorf("invalid minute units BCD: %w", err)
	}

	// Minute tens (bits 24-27)
	minuteTens, err := getBCD(data, 24)
	if err != nil {
		return nil, fmt.Errorf("invalid minute tens BCD: %w", err)
	}
	tc.Minutes = minuteTens*10 + minuteUnits

	// Hour units (bits 28-31)
	hourUnits, err := getBCD(data, 28)
	if err != nil {
		return nil, fmt.Errorf("invalid hour units BCD: %w", err)
	}

	// Hour tens (bits 33-34) - only 2 bits
	hourTens := 0
	if getBit(data, 33) {
		hourTens |= 1
	}
	if getBit(data, 34) {
		hourTens |= 2
	}
	tc.Hours = hourTens*10 + hourUnits

	// User bits (bits 35-58, 24 bits)
	for i := 0; i < 24; i++ {
		byteIdx := i / 8
		bitIdx := i % 8
		if getBit(data, 35+i) {
			tc.UserBits[byteIdx] |= 1 << bitIdx
		}
	}

	// Verify sync word (bits 59-78)
	receivedSync := uint32(0)
	for i := 0; i < 20; i++ {
		if getBit(data, 59+i) {
			receivedSync |= 1 << i
		}
	}
	if receivedSync != LTCSyncWord {
		return nil, fmt.Errorf("invalid sync word: got 0x%X, expected 0x%X", receivedSync, LTCSyncWord)
	}

	// Note: FrameRate is not encoded in LTC frame and must be set by caller
	// Validate basic ranges (hours, minutes, seconds) but not frames (requires frame rate)
	if tc.Hours < 0 || tc.Hours > 23 {
		return nil, fmt.Errorf("hours out of range: %d (must be 0-23)", tc.Hours)
	}
	if tc.Minutes < 0 || tc.Minutes > 59 {
		return nil, fmt.Errorf("minutes out of range: %d (must be 0-59)", tc.Minutes)
	}
	if tc.Seconds < 0 || tc.Seconds > 59 {
		return nil, fmt.Errorf("seconds out of range: %d (must be 0-59)", tc.Seconds)
	}
	// Frames validation requires frame rate, so we only check it's not obviously wrong
	if tc.Frames < 0 || tc.Frames > 30 {
		return nil, fmt.Errorf("frames out of range: %d (must be 0-30)", tc.Frames)
	}

	return tc, nil
}

// Validate validates SMPTE timecode values
func (tc *SMPTETimeCode) Validate() error {
	if tc.Hours < 0 || tc.Hours > 23 {
		return fmt.Errorf("hours out of range: %d (must be 0-23)", tc.Hours)
	}
	if tc.Minutes < 0 || tc.Minutes > 59 {
		return fmt.Errorf("minutes out of range: %d (must be 0-59)", tc.Minutes)
	}
	if tc.Seconds < 0 || tc.Seconds > 59 {
		return fmt.Errorf("seconds out of range: %d (must be 0-59)", tc.Seconds)
	}

	maxFrames := int(tc.FrameRate)
	if tc.FrameRate == FrameRate2997 {
		maxFrames = 30 // 29.97 uses 0-29 frame numbers
	}
	if tc.Frames < 0 || tc.Frames >= maxFrames {
		return fmt.Errorf("frames out of range: %d (must be 0-%d for %.2f fps)", tc.Frames, maxFrames-1, tc.FrameRate)
	}

	return nil
}

// String returns timecode as "HH:MM:SS:FF" or "HH:MM:SS;FF" (drop-frame) string
func (tc *SMPTETimeCode) String() string {
	separator := ":"
	if tc.DropFrame {
		separator = ";"
	}
	return fmt.Sprintf("%02d:%02d:%02d%s%02d", tc.Hours, tc.Minutes, tc.Seconds, separator, tc.Frames)
}

// Parse parses timecode string "HH:MM:SS:FF" or "HH:MM:SS;FF" (drop-frame)
func Parse(s string, frameRate FrameRate) (*SMPTETimeCode, error) {
	tc := &SMPTETimeCode{FrameRate: frameRate}

	var h, m, sec, f int
	dropFrame := false

	// Check for drop-frame separator (;)
	if n, err := fmt.Sscanf(s, "%d:%d:%d;%d", &h, &m, &sec, &f); err == nil && n == 4 {
		dropFrame = true
	} else if n, err := fmt.Sscanf(s, "%d:%d:%d:%d", &h, &m, &sec, &f); err != nil || n != 4 {
		return nil, fmt.Errorf("invalid timecode format: %s (expected HH:MM:SS:FF or HH:MM:SS;FF)", s)
	}

	tc.Hours = h
	tc.Minutes = m
	tc.Seconds = sec
	tc.Frames = f
	tc.DropFrame = dropFrame

	if err := tc.Validate(); err != nil {
		return nil, err
	}

	return tc, nil
}

// FromTime converts time.Time to SMPTE timecode
func FromTime(t time.Time, frameRate FrameRate, dropFrame bool) *SMPTETimeCode {
	// Calculate total seconds since midnight
	midnight := time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, t.Location())
	elapsed := t.Sub(midnight)
	totalSeconds := int(elapsed.Seconds())

	hours := totalSeconds / 3600
	minutes := (totalSeconds % 3600) / 60
	seconds := totalSeconds % 60

	// Calculate frames from nanoseconds
	nanos := elapsed.Nanoseconds() % 1e9
	frameDuration := 1.0 / float64(frameRate)
	frames := int(float64(nanos) / (frameDuration * 1e9))

	// Apply drop-frame logic if needed
	if dropFrame && frameRate == FrameRate2997 {
		// Drop frames 00 and 01 every minute except minutes divisible by 10
		dropFrames := CalculateDropFrames(hours, minutes, frameRate)
		if frames < dropFrames {
			// Adjust for dropped frames
			if minutes%10 != 0 {
				frames += 2
			}
		}
	}

	tc := &SMPTETimeCode{
		Hours:     hours,
		Minutes:   minutes,
		Seconds:   seconds,
		Frames:    frames,
		FrameRate: frameRate,
		DropFrame: dropFrame,
	}

	return tc
}

// ToTime converts SMPTE timecode to time.Time
func (tc *SMPTETimeCode) ToTime(baseDate time.Time) time.Time {
	// Calculate total seconds
	totalSeconds := tc.Hours*3600 + tc.Minutes*60 + tc.Seconds

	// Calculate frame duration
	frameDuration := 1.0 / float64(tc.FrameRate)

	// Add frame offset
	nanos := int64(float64(tc.Frames) * frameDuration * 1e9)

	// Adjust for drop-frame if needed
	if tc.DropFrame && tc.FrameRate == FrameRate2997 {
		dropFrames := CalculateDropFrames(tc.Hours, tc.Minutes, tc.FrameRate)
		nanos -= int64(float64(dropFrames) * frameDuration * 1e9)
	}

	// Get midnight of base date
	midnight := time.Date(baseDate.Year(), baseDate.Month(), baseDate.Day(), 0, 0, 0, 0, baseDate.Location())

	// Add time components
	return midnight.Add(time.Duration(totalSeconds)*time.Second + time.Duration(nanos))
}

// CalculateDropFrames calculates number of frames to drop for drop-frame timecode
func CalculateDropFrames(hours, minutes int, frameRate FrameRate) int {
	if frameRate != FrameRate2997 {
		return 0
	}

	// Drop 2 frames every minute except minutes divisible by 10
	// Total drops = (minutes - minutes/10) * 2
	droppedMinutes := minutes - (minutes / 10)
	return droppedMinutes * 2
}

// AddFrames adds frames to timecode, handling rollover
func (tc *SMPTETimeCode) AddFrames(frames int) *SMPTETimeCode {
	result := *tc // Copy

	maxFrames := int(tc.FrameRate)
	if tc.FrameRate == FrameRate2997 {
		maxFrames = 30
	}

	result.Frames += frames

	// Handle frame rollover
	for result.Frames >= maxFrames {
		result.Frames -= maxFrames
		result.Seconds++
	}
	for result.Seconds >= 60 {
		result.Seconds -= 60
		result.Minutes++
	}
	for result.Minutes >= 60 {
		result.Minutes -= 60
		result.Hours++
	}
	if result.Hours >= 24 {
		result.Hours -= 24
	}

	// Handle underflow
	for result.Frames < 0 {
		result.Frames += maxFrames
		result.Seconds--
	}
	for result.Seconds < 0 {
		result.Seconds += 60
		result.Minutes--
	}
	for result.Minutes < 0 {
		result.Minutes += 60
		result.Hours--
	}
	if result.Hours < 0 {
		result.Hours += 24
	}

	return &result
}

// SubtractFrames subtracts frames from timecode
func (tc *SMPTETimeCode) SubtractFrames(frames int) *SMPTETimeCode {
	return tc.AddFrames(-frames)
}
