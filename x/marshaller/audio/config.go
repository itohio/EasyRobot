package audio

import (
	"context"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// wavConfig holds WAV file format settings.
type wavConfig struct {
	paths      []string // Multiple WAV file paths
	sampleRate int      // Target sample rate (0 = use source)
	channels   int      // Target channels (0 = use source)
	bitDepth   int      // Target bit depth (16, 24, 32, or 0 = use source/auto)
}

func defaultWavConfig() wavConfig {
	return wavConfig{
		paths:      nil,
		sampleRate: 0,
		channels:   0,
		bitDepth:   0, // 0 = auto (use source or default to 16)
	}
}

// deviceConfig holds audio device settings.
type deviceConfig struct {
	deviceIDs  []string // Multiple device IDs (base64-encoded hashes from malgo)
	deviceType string   // "capture" or "playback"
	sampleRate int      // Target sample rate
	channels   int      // Target channels
	format     string   // Format: "F32", "S32", "S24", "S16", "U8" (empty = use device default)
}

func defaultDeviceConfig() deviceConfig {
	return deviceConfig{
		deviceIDs:  nil,
		deviceType: "",
		sampleRate: 48000,
		channels:   2,
		format:     "",
	}
}

// streamConfig holds stream synchronization settings.
type streamConfig struct {
	frameSize    int  // Frames per read/write operation
	bufferSize   int  // Buffer size in frames (0 = auto)
	allowBestEffort bool // Allow best-effort synchronization (drop frames if needed)
}

func defaultStreamConfig() streamConfig {
	return streamConfig{
		frameSize:    0, // 0 = auto (use device default)
		bufferSize:   0, // 0 = auto (3 seconds worth)
		allowBestEffort: true,
	}
}

// tensorConfig holds tensor conversion options.
type tensorConfig struct {
	destinationType types.DataType // Target data type for conversion (DT_UNKNOWN = preserve source)
}

func defaultTensorConfig() tensorConfig {
	return tensorConfig{
		destinationType: types.DT_UNKNOWN, // Preserve source format
	}
}

// config holds all audio marshaller configuration.
type config struct {
	ctx          context.Context
	wav          wavConfig
	device       deviceConfig
	stream       streamConfig
	tensor       tensorConfig
	ReleaseAfterProcessing bool
}

func defaultConfig() config {
	return config{
		ctx:                    context.Background(),
		wav:                    defaultWavConfig(),
		device:                 defaultDeviceConfig(),
		stream:                 defaultStreamConfig(),
		tensor:                 defaultTensorConfig(),
		ReleaseAfterProcessing: false,
	}
}

