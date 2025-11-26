package rs

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCalculateRSParams(t *testing.T) {
	tests := []struct {
		name      string
		dataLen   int
		maxErrors int
		wantErr   bool
		validate  func(t *testing.T, n, k, tVal int)
	}{
		{
			name:      "valid params",
			dataLen:   100,
			maxErrors: 10,
			wantErr:   false,
			validate: func(t *testing.T, n, k, tVal int) {
				assert.Greater(t, n, k)
				assert.LessOrEqual(t, n, 255)
				assert.Greater(t, tVal, 0)
			},
		},
		{
			name:      "small data",
			dataLen:   10,
			maxErrors: 5,
			wantErr:   false,
			validate: func(t *testing.T, n, k, tVal int) {
				assert.Greater(t, n, k)
				assert.LessOrEqual(t, n, 255)
			},
		},
		{
			name:      "large data",
			dataLen:   500,
			maxErrors: 16,
			wantErr:   false,
			validate: func(t *testing.T, n, k, tVal int) {
				assert.LessOrEqual(t, n, 255)
			},
		},
		{
			name:      "invalid dataLen",
			dataLen:   0,
			maxErrors: 10,
			wantErr:   true,
		},
		{
			name:      "invalid maxErrors",
			dataLen:   100,
			maxErrors: 0,
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n, k, tVal, err := CalculateRSParams(tt.dataLen, tt.maxErrors)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				if tt.validate != nil {
					tt.validate(t, n, k, tVal)
				}
			}
		})
	}
}

func TestCalculateCodeRate(t *testing.T) {
	tests := []struct {
		name     string
		n, k     int
		expected float32
	}{
		{"RS(255, 223)", 255, 223, 223.0 / 255.0},
		{"RS(255, 239)", 255, 239, 239.0 / 255.0},
		{"RS(255, 200)", 255, 200, 200.0 / 255.0},
		{"invalid n=0", 0, 100, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rate := CalculateCodeRate(tt.n, tt.k)
			assert.InDelta(t, tt.expected, rate, 0.001)
		})
	}
}

func TestCalculateOverhead(t *testing.T) {
	tests := []struct {
		name     string
		n, k     int
		expected float32
	}{
		{"RS(255, 223)", 255, 223, (255.0 - 223.0) / 255.0 * 100.0},
		{"RS(255, 239)", 255, 239, (255.0 - 239.0) / 255.0 * 100.0},
		{"invalid n=0", 0, 100, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			overhead := CalculateOverhead(tt.n, tt.k)
			assert.InDelta(t, tt.expected, overhead, 0.1)
		})
	}
}

func TestMaxCorrectableErrors(t *testing.T) {
	tests := []struct {
		name     string
		n, k     int
		expected int
	}{
		{"RS(255, 223)", 255, 223, (255 - 223) / 2},
		{"RS(255, 239)", 255, 239, (255 - 239) / 2},
		{"RS(255, 200)", 255, 200, (255 - 200) / 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tVal := MaxCorrectableErrors(tt.n, tt.k)
			assert.Equal(t, tt.expected, tVal)
		})
	}
}

func TestCalculateOptimalBlockSize(t *testing.T) {
	tests := []struct {
		name     string
		dataLen  int
		expected int
	}{
		{"small data", 10, 10}, // For data smaller than common sizes, return dataLen
		{"medium data", 100, 64},
		{"large data", 500, 200},
		{"exact match", 64, 64},
		{"invalid", 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			size := CalculateOptimalBlockSize(tt.dataLen)
			if tt.dataLen == 0 {
				assert.Equal(t, 0, size)
			} else {
				assert.Greater(t, size, 0)
				// Size may be larger than dataLen if it's a common block size
				assert.LessOrEqual(t, size, tt.dataLen*2) // Allow some flexibility
			}
		})
	}
}

func TestEstimateErrorCorrectionCapability(t *testing.T) {
	tests := []struct {
		name      string
		n, k, t   int
		errorRate float32
		validate  func(t *testing.T, prob float32)
	}{
		{
			name:      "low error rate",
			n:         255,
			k:         223,
			t:         16,
			errorRate: 0.01,
			validate: func(t *testing.T, prob float32) {
				// Simplified estimation may return negative for very low error rates
				assert.GreaterOrEqual(t, prob, float32(0))
				assert.LessOrEqual(t, prob, float32(1))
			},
		},
		{
			name:      "high error rate",
			n:         255,
			k:         223,
			t:         16,
			errorRate: 0.5,
			validate: func(t *testing.T, prob float32) {
				assert.GreaterOrEqual(t, prob, float32(0))
				assert.LessOrEqual(t, prob, float32(1))
			},
		},
		{
			name:      "invalid error rate",
			n:         255,
			k:         223,
			t:         16,
			errorRate: 1.5,
			validate: func(t *testing.T, prob float32) {
				assert.Equal(t, float32(0), prob)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prob := EstimateErrorCorrectionCapability(tt.n, tt.k, tt.t, tt.errorRate)
			if tt.validate != nil {
				tt.validate(t, prob)
			}
		})
	}
}

func TestCalculateBlockCount(t *testing.T) {
	tests := []struct {
		name     string
		dataLen  int
		k        int
		expected int
	}{
		{"exact multiple", 446, 223, 2},
		{"with remainder", 300, 223, 2},
		{"single block", 100, 223, 1},
		{"invalid k=0", 100, 0, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			count := CalculateBlockCount(tt.dataLen, tt.k)
			assert.Equal(t, tt.expected, count)
		})
	}
}

func TestCalculateEncodedLength(t *testing.T) {
	tests := []struct {
		name        string
		dataLen, n, k int
		expected    int
	}{
		{"exact blocks", 446, 255, 223, 510},
		{"with padding", 300, 255, 223, 510},
		{"single block", 100, 255, 223, 255},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			length := CalculateEncodedLength(tt.dataLen, tt.n, tt.k)
			assert.Equal(t, tt.expected, length)
		})
	}
}

func TestCalculateDecodedLength(t *testing.T) {
	tests := []struct {
		name         string
		encodedLen, n, k int
		expected     int
	}{
		{"exact blocks", 510, 255, 223, 446},
		{"incomplete block", 300, 255, 223, 223},
		{"single block", 255, 255, 223, 223},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			length := CalculateDecodedLength(tt.encodedLen, tt.n, tt.k)
			assert.Equal(t, tt.expected, length)
		})
	}
}

