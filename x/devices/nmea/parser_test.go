package nmea

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseSentence(t *testing.T) {
	tests := []struct {
		name    string
		line    string
		wantErr bool
		check   func(*testing.T, *Sentence)
	}{
		{
			name:    "valid GPGGA",
			line:    "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n",
			wantErr: false,
			check: func(t *testing.T, s *Sentence) {
				assert.Equal(t, "GPGGA", s.Type)
				assert.Greater(t, len(s.Fields), 0)
			},
		},
		{
			name:    "valid GPRMC",
			line:    "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A\r\n",
			wantErr: false,
			check: func(t *testing.T, s *Sentence) {
				assert.Equal(t, "GPRMC", s.Type)
				assert.Greater(t, len(s.Fields), 0)
			},
		},
		{
			name:    "invalid checksum",
			line:    "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*00\r\n",
			wantErr: true,
		},
		{
			name:    "missing dollar sign",
			line:    "GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := ParseSentence(tt.line)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, s)
			} else {
				require.NoError(t, err)
				require.NotNil(t, s)
				if tt.check != nil {
					tt.check(t, s)
				}
			}
		})
	}
}

func TestParseGGA(t *testing.T) {
	line := "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n"
	s, err := ParseSentence(line)
	require.NoError(t, err)

	data, err := ParseGGA(s)
	require.NoError(t, err)

	assert.True(t, data.TimeValid)
	assert.True(t, data.PosValid)
	assert.InDelta(t, 48.1173, float64(data.Latitude), 0.01)
	assert.InDelta(t, 11.5167, float64(data.Longitude), 0.01)
	assert.Equal(t, 1, data.FixQuality)
	assert.Equal(t, 8, data.NumSatellites)
	assert.InDelta(t, 0.9, float64(data.HDOP), 0.1)
	assert.InDelta(t, 545.4, float64(data.Altitude), 1.0)
}

func TestParseRMC(t *testing.T) {
	line := "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A\r\n"
	s, err := ParseSentence(line)
	require.NoError(t, err)

	data, err := ParseRMC(s)
	require.NoError(t, err)

	assert.True(t, data.TimeValid)
	assert.True(t, data.PosValid)
	assert.True(t, data.NavValid)
	assert.InDelta(t, 48.1173, float64(data.Latitude), 0.01)
	assert.InDelta(t, 11.5167, float64(data.Longitude), 0.01)
	assert.InDelta(t, 11.5, float64(data.Speed), 1.0) // 22.4 knots ≈ 11.5 m/s
	assert.InDelta(t, 84.4, float64(data.Course), 0.1)
}

func TestParser_ParseBytes(t *testing.T) {
	parser := NewParser()

	// Test with multiple sentences
	data := []byte("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A\r\n")

	sentences, consumed, err := parser.ParseBytes(data)
	require.NoError(t, err)
	assert.Equal(t, 2, len(sentences))
	assert.Equal(t, len(data), consumed)
	assert.Equal(t, "GPGGA", sentences[0].Type)
	assert.Equal(t, "GPRMC", sentences[1].Type)
}

func TestParser_ParseBytes_Incomplete(t *testing.T) {
	parser := NewParser()

	// Test with incomplete sentence
	data := []byte("$GPGGA,123519,4807.038,N,")
	sentences, consumed, err := parser.ParseBytes(data)
	require.NoError(t, err)
	assert.Equal(t, 0, len(sentences))
	assert.Equal(t, 0, consumed)

	// Complete the sentence
	data2 := []byte("01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n")
	sentences, consumed, err = parser.ParseBytes(data2)
	require.NoError(t, err)
	assert.Equal(t, 1, len(sentences))
}

func TestMergeData(t *testing.T) {
	dst := &Data{
		LastUpdate: time.Now(),
	}
	src := &Data{
		Time:       time.Now().Add(time.Hour),
		TimeValid:  true,
		Latitude:   48.1173,
		Longitude:  11.5167,
		PosValid:   true,
		FixQuality: 1,
		NumSatellites: 8,
		Speed:     11.5,
		Course:    84.4,
		NavValid:  true,
		LastUpdate: time.Now().Add(time.Hour),
	}

	MergeData(dst, src)

	assert.True(t, dst.TimeValid)
	assert.True(t, dst.PosValid)
	assert.True(t, dst.NavValid)
	assert.Equal(t, src.Latitude, dst.Latitude)
	assert.Equal(t, src.Longitude, dst.Longitude)
	assert.Equal(t, src.Speed, dst.Speed)
	assert.Equal(t, src.Course, dst.Course)
}

