package tc

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/dsp/modulation/mpsk"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
)

func TestModulator_Encode(t *testing.T) {
	sampleRate := float32(44100)
	carrierFreq := float32(1000)
	symbolRate := float32(1000)
	modulationOrder := 2 // BPSK

	pskModulator := mpsk.NewModulator(sampleRate, carrierFreq, symbolRate, modulationOrder)
	frameRate := FrameRate30
	tcModulator := NewModulator(pskModulator, frameRate)

	tc := &SMPTETimeCode{
		Hours:     12,
		Minutes:   34,
		Seconds:   56,
		Frames:    7,
		FrameRate: frameRate,
	}

	// Calculate expected samples
	expectedSamples := tcModulator.Encode(nil, tc)
	assert.Greater(t, expectedSamples, 0)

	// Encode to samples
	amplitudes := vec.New(expectedSamples)
	samplesWritten := tcModulator.Encode(amplitudes, tc)
	assert.Equal(t, expectedSamples, samplesWritten)
	assert.Greater(t, samplesWritten, 0)
}

func TestModulator_EncodeMultiple(t *testing.T) {
	sampleRate := float32(44100)
	carrierFreq := float32(1000)
	symbolRate := float32(1000)
	modulationOrder := 2 // BPSK

	pskModulator := mpsk.NewModulator(sampleRate, carrierFreq, symbolRate, modulationOrder)
	frameRate := FrameRate30
	tcModulator := NewModulator(pskModulator, frameRate)

	timecodes := []*SMPTETimeCode{
		{Hours: 1, Minutes: 0, Seconds: 0, Frames: 0, FrameRate: frameRate},
		{Hours: 1, Minutes: 0, Seconds: 0, Frames: 1, FrameRate: frameRate},
		{Hours: 1, Minutes: 0, Seconds: 0, Frames: 2, FrameRate: frameRate},
	}

	// Calculate expected samples
	expectedSamples := tcModulator.EncodeMultiple(nil, timecodes)
	assert.Greater(t, expectedSamples, 0)

	// Encode to samples
	amplitudes := vec.New(expectedSamples)
	samplesWritten := tcModulator.EncodeMultiple(amplitudes, timecodes)
	assert.Equal(t, expectedSamples, samplesWritten)
	assert.Greater(t, samplesWritten, 0)
}

func TestModulator_Reset(t *testing.T) {
	sampleRate := float32(44100)
	carrierFreq := float32(1000)
	symbolRate := float32(1000)
	modulationOrder := 2

	pskModulator := mpsk.NewModulator(sampleRate, carrierFreq, symbolRate, modulationOrder)
	frameRate := FrameRate30
	tcModulator := NewModulator(pskModulator, frameRate)

	tc := &SMPTETimeCode{
		Hours:     12,
		Minutes:   34,
		Seconds:   56,
		Frames:    7,
		FrameRate: frameRate,
	}

	// Encode once
	amplitudes1 := vec.New(10000)
	tcModulator.Encode(amplitudes1, tc)

	// Reset
	tcModulator.Reset()

	// Encode again - should work the same
	amplitudes2 := vec.New(10000)
	samplesWritten := tcModulator.Encode(amplitudes2, tc)
	assert.Greater(t, samplesWritten, 0)
}

func TestModulator_FrameRate(t *testing.T) {
	sampleRate := float32(44100)
	carrierFreq := float32(1000)
	symbolRate := float32(1000)
	modulationOrder := 2

	pskModulator := mpsk.NewModulator(sampleRate, carrierFreq, symbolRate, modulationOrder)
	frameRate := FrameRate30
	tcModulator := NewModulator(pskModulator, frameRate)

	assert.Equal(t, frameRate, tcModulator.FrameRate())
}

func TestLTCModulator_Encode(t *testing.T) {
	sampleRate := float32(44100)
	frameRate := FrameRate30
	ltcModulator := NewLTCModulator(sampleRate, frameRate)

	tc := &SMPTETimeCode{
		Hours:     12,
		Minutes:   34,
		Seconds:   56,
		Frames:    7,
		FrameRate: frameRate,
	}

	// Calculate expected samples
	expectedSamples := ltcModulator.Encode(nil, tc)
	assert.Greater(t, expectedSamples, 0)

	// Encode to samples
	amplitudes := vec.New(expectedSamples)
	samplesWritten := ltcModulator.Encode(amplitudes, tc)
	assert.Equal(t, expectedSamples, samplesWritten)
	assert.Greater(t, samplesWritten, 0)
}
