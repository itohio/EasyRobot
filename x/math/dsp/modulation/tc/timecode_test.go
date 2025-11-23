package tc

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEncodeBCD_DecodeBCD(t *testing.T) {
	tests := []struct {
		name     string
		digit    int
		wantBCD  byte
		wantErr  bool
	}{
		{"zero", 0, 0x0, false},
		{"one", 1, 0x1, false},
		{"five", 5, 0x5, false},
		{"nine", 9, 0x9, false},
		{"ten", 10, 0, true},
		{"negative", -1, 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bcd, err := EncodeBCD(tt.digit)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.wantBCD, bcd)

				// Roundtrip test
				decoded, err := DecodeBCD(bcd)
				require.NoError(t, err)
				assert.Equal(t, tt.digit, decoded)
			}
		})
	}
}

func TestDecodeBCD_Invalid(t *testing.T) {
	invalidBCD := []byte{0xA, 0xB, 0xC, 0xD, 0xE, 0xF}
	for _, bcd := range invalidBCD {
		_, err := DecodeBCD(bcd)
		assert.Error(t, err, "BCD value 0x%X should be invalid", bcd)
	}
}

func TestSMPTETimeCode_Marshal_Unmarshal(t *testing.T) {
	tests := []struct {
		name string
		tc   *SMPTETimeCode
	}{
		{
			name: "basic 30fps",
			tc: &SMPTETimeCode{
				Hours:     12,
				Minutes:   34,
				Seconds:   56,
				Frames:    7,
				FrameRate: FrameRate30,
				DropFrame: false,
			},
		},
		{
			name: "drop-frame 29.97fps",
			tc: &SMPTETimeCode{
				Hours:     1,
				Minutes:   23,
				Seconds:   45,
				Frames:    12,
				FrameRate: FrameRate2997,
				DropFrame: true,
			},
		},
		{
			name: "25fps PAL",
			tc: &SMPTETimeCode{
				Hours:     23,
				Minutes:   59,
				Seconds:   59,
				Frames:    24,
				FrameRate: FrameRate25,
				DropFrame: false,
			},
		},
		{
			name: "24fps cinema",
			tc: &SMPTETimeCode{
				Hours:     0,
				Minutes:   0,
				Seconds:   0,
				Frames:    0,
				FrameRate: FrameRate24,
				DropFrame: false,
			},
		},
		{
			name: "with user bits",
			tc: &SMPTETimeCode{
				Hours:     10,
				Minutes:   20,
				Seconds:   30,
				Frames:    15,
				FrameRate: FrameRate30,
				UserBits:  [3]byte{0x12, 0x34, 0x56},
			},
		},
		{
			name: "with color frame",
			tc: &SMPTETimeCode{
				Hours:      5,
				Minutes:    15,
				Seconds:    25,
				Frames:     8,
				FrameRate:  FrameRate30,
				ColorFrame: true,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal
			data, err := MarshalLTC(tt.tc)
			require.NoError(t, err)
			require.Equal(t, 10, len(data), "LTC frame must be 10 bytes (80 bits)")

			// Unmarshal
			tc2, err := UnmarshalLTC(data)
			require.NoError(t, err)

			// Set frame rate (not encoded in LTC frame, must be known from context)
			tc2.FrameRate = tt.tc.FrameRate

			// Verify all fields
			assert.Equal(t, tt.tc.Hours, tc2.Hours)
			assert.Equal(t, tt.tc.Minutes, tc2.Minutes)
			assert.Equal(t, tt.tc.Seconds, tc2.Seconds)
			assert.Equal(t, tt.tc.Frames, tc2.Frames)
			assert.Equal(t, tt.tc.FrameRate, tc2.FrameRate)
			assert.Equal(t, tt.tc.DropFrame, tc2.DropFrame)
			assert.Equal(t, tt.tc.ColorFrame, tc2.ColorFrame)
			assert.Equal(t, tt.tc.UserBits, tc2.UserBits)
		})
	}
}

func TestSMPTETimeCode_Validate(t *testing.T) {
	tests := []struct {
		name    string
		tc      *SMPTETimeCode
		wantErr bool
	}{
		{
			name: "valid",
			tc: &SMPTETimeCode{
				Hours:     12,
				Minutes:   34,
				Seconds:   56,
				Frames:    7,
				FrameRate: FrameRate30,
			},
			wantErr: false,
		},
		{
			name: "invalid hours",
			tc: &SMPTETimeCode{
				Hours:     24,
				Minutes:   0,
				Seconds:   0,
				Frames:    0,
				FrameRate: FrameRate30,
			},
			wantErr: true,
		},
		{
			name: "invalid minutes",
			tc: &SMPTETimeCode{
				Hours:     12,
				Minutes:   60,
				Seconds:   0,
				Frames:    0,
				FrameRate: FrameRate30,
			},
			wantErr: true,
		},
		{
			name: "invalid seconds",
			tc: &SMPTETimeCode{
				Hours:     12,
				Minutes:   34,
				Seconds:   60,
				Frames:    0,
				FrameRate: FrameRate30,
			},
			wantErr: true,
		},
		{
			name: "invalid frames 30fps",
			tc: &SMPTETimeCode{
				Hours:     12,
				Minutes:   34,
				Seconds:   56,
				Frames:    30,
				FrameRate: FrameRate30,
			},
			wantErr: true,
		},
		{
			name: "invalid frames 25fps",
			tc: &SMPTETimeCode{
				Hours:     12,
				Minutes:   34,
				Seconds:   56,
				Frames:    25,
				FrameRate: FrameRate25,
			},
			wantErr: true,
		},
		{
			name: "valid 29.97fps",
			tc: &SMPTETimeCode{
				Hours:     12,
				Minutes:   34,
				Seconds:   56,
				Frames:    29,
				FrameRate: FrameRate2997,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.tc.Validate()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestSMPTETimeCode_String(t *testing.T) {
	tests := []struct {
		name string
		tc   *SMPTETimeCode
		want string
	}{
		{
			name: "non-drop-frame",
			tc: &SMPTETimeCode{
				Hours:     12,
				Minutes:   34,
				Seconds:   56,
				Frames:    7,
				DropFrame: false,
			},
			want: "12:34:56:07",
		},
		{
			name: "drop-frame",
			tc: &SMPTETimeCode{
				Hours:     1,
				Minutes:   23,
				Seconds:   45,
				Frames:    12,
				DropFrame: true,
			},
			want: "01:23:45;12",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.tc.String())
		})
	}
}

func TestParse(t *testing.T) {
	tests := []struct {
		name      string
		s         string
		frameRate FrameRate
		want      *SMPTETimeCode
		wantErr   bool
	}{
		{
			name:      "non-drop-frame",
			s:         "12:34:56:07",
			frameRate: FrameRate30,
			want: &SMPTETimeCode{
				Hours:     12,
				Minutes:   34,
				Seconds:   56,
				Frames:    7,
				FrameRate: FrameRate30,
				DropFrame: false,
			},
			wantErr: false,
		},
		{
			name:      "drop-frame",
			s:         "01:23:45;12",
			frameRate: FrameRate2997,
			want: &SMPTETimeCode{
				Hours:     1,
				Minutes:   23,
				Seconds:   45,
				Frames:    12,
				FrameRate: FrameRate2997,
				DropFrame: true,
			},
			wantErr: false,
		},
		{
			name:      "invalid format",
			s:         "12:34:56",
			frameRate: FrameRate30,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc, err := Parse(tt.s, tt.frameRate)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.want.Hours, tc.Hours)
				assert.Equal(t, tt.want.Minutes, tc.Minutes)
				assert.Equal(t, tt.want.Seconds, tc.Seconds)
				assert.Equal(t, tt.want.Frames, tc.Frames)
				assert.Equal(t, tt.want.DropFrame, tc.DropFrame)
			}
		})
	}
}

func TestSMPTETimeCode_AddFrames(t *testing.T) {
	tests := []struct {
		name     string
		tc       *SMPTETimeCode
		add      int
		want     *SMPTETimeCode
	}{
		{
			name: "add frames within second",
			tc: &SMPTETimeCode{
				Hours:     10,
				Minutes:   20,
				Seconds:   30,
				Frames:    5,
				FrameRate: FrameRate30,
			},
			add: 10,
			want: &SMPTETimeCode{
				Hours:     10,
				Minutes:   20,
				Seconds:   30,
				Frames:    15,
				FrameRate: FrameRate30,
			},
		},
		{
			name: "rollover frames to seconds",
			tc: &SMPTETimeCode{
				Hours:     10,
				Minutes:   20,
				Seconds:   30,
				Frames:    28,
				FrameRate: FrameRate30,
			},
			add: 5,
			want: &SMPTETimeCode{
				Hours:     10,
				Minutes:   20,
				Seconds:   31,
				Frames:    3,
				FrameRate: FrameRate30,
			},
		},
		{
			name: "rollover seconds to minutes",
			tc: &SMPTETimeCode{
				Hours:     10,
				Minutes:   20,
				Seconds:   59,
				Frames:    28,
				FrameRate: FrameRate30,
			},
			add: 5,
			want: &SMPTETimeCode{
				Hours:     10,
				Minutes:   21,
				Seconds:   0,
				Frames:    3,
				FrameRate: FrameRate30,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.tc.AddFrames(tt.add)
			assert.Equal(t, tt.want.Hours, result.Hours)
			assert.Equal(t, tt.want.Minutes, result.Minutes)
			assert.Equal(t, tt.want.Seconds, result.Seconds)
			assert.Equal(t, tt.want.Frames, result.Frames)
		})
	}
}

func TestSMPTETimeCode_FromTime_ToTime(t *testing.T) {
	baseDate := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	testTime := time.Date(2024, 1, 1, 12, 34, 56, 789000000, time.UTC)

	tc := FromTime(testTime, FrameRate30, false)
	assert.Equal(t, 12, tc.Hours)
	assert.Equal(t, 34, tc.Minutes)
	assert.Equal(t, 56, tc.Seconds)
	// Frames will depend on nanoseconds, should be around 23-24 for 30fps

	// Convert back to time
	recoveredTime := tc.ToTime(baseDate)
	// Allow some error due to frame quantization
	diff := recoveredTime.Sub(testTime)
	if diff < 0 {
		diff = -diff
	}
	// Should be within one frame period (30 fps = 33.33ms per frame)
	framePeriodNanos := int64(float64(time.Second.Nanoseconds()) / float64(FrameRate30))
	framePeriod := time.Duration(framePeriodNanos)
	assert.Less(t, diff, framePeriod*2, "time conversion should be accurate within frame period")
}

func TestCalculateDropFrames(t *testing.T) {
	tests := []struct {
		name      string
		hours     int
		minutes   int
		frameRate FrameRate
		want      int
	}{
		{"minute 0", 0, 0, FrameRate2997, 0},
		{"minute 1", 0, 1, FrameRate2997, 2},
		{"minute 9", 0, 9, FrameRate2997, 18},
		{"minute 10", 0, 10, FrameRate2997, 18}, // No drop at minute 10
		{"minute 11", 0, 11, FrameRate2997, 20},
		{"minute 59", 0, 59, FrameRate2997, 108}, // (59 - 5) * 2
		{"non-drop-frame", 0, 30, FrameRate30, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CalculateDropFrames(tt.hours, tt.minutes, tt.frameRate)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestUnmarshalLTC_InvalidSyncWord(t *testing.T) {
	// Create valid frame
	tc := &SMPTETimeCode{
		Hours:     12,
		Minutes:   34,
		Seconds:   56,
		Frames:    7,
		FrameRate: FrameRate30,
	}
	data, err := MarshalLTC(tc)
	require.NoError(t, err)

	// Corrupt sync word
	data[7] ^= 0xFF

	// Try to unmarshal
	_, err = UnmarshalLTC(data)
	assert.Error(t, err, "should fail with invalid sync word")
}

func TestUnmarshalLTC_TooShort(t *testing.T) {
	data := make([]byte, 9) // Too short
	_, err := UnmarshalLTC(data)
	assert.Error(t, err)
}
