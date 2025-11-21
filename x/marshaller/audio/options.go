package audio

import (
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

type configOption interface {
	types.Option
	applyConfig(*config)
}

type optionFunc struct {
	onTypes  func(*types.Options)
	onConfig func(*config)
}

func (o optionFunc) Apply(opts *types.Options) {
	if o.onTypes != nil {
		o.onTypes(opts)
	}
}

func (o optionFunc) applyConfig(cfg *config) {
	if o.onConfig != nil {
		o.onConfig(cfg)
	}
}

func newOption(onCfg func(*config)) configOption {
	return optionFunc{onConfig: onCfg}
}

// WithWAVPath registers a WAV file path for reading or writing.
// Can be called multiple times to specify multiple files.
func WithWAVPath(path string) types.Option {
	path = strings.TrimSpace(path)
	return newOption(func(cfg *config) {
		if path == "" {
			return
		}
		cfg.wav.paths = append(cfg.wav.paths, path)
	})
}

// WithWAVPaths registers multiple WAV file paths at once.
func WithWAVPaths(paths ...string) types.Option {
	return newOption(func(cfg *config) {
		for _, path := range paths {
			path = strings.TrimSpace(path)
			if path != "" {
				cfg.wav.paths = append(cfg.wav.paths, path)
			}
		}
	})
}

// WithSampleRate sets the target sample rate (Hz).
// For unmarshaller: used when source doesn't specify rate or for validation.
// For marshaller: used when tensor doesn't have rate info.
func WithSampleRate(rate int) types.Option {
	return newOption(func(cfg *config) {
		if rate > 0 {
			cfg.wav.sampleRate = rate
			cfg.device.sampleRate = rate
		}
	})
}

// WithChannels sets the target number of channels.
// For unmarshaller: used when source doesn't specify channels or for validation.
// For marshaller: used when tensor doesn't have channel info.
func WithChannels(channels int) types.Option {
	return newOption(func(cfg *config) {
		if channels > 0 {
			cfg.wav.channels = channels
			cfg.device.channels = channels
		}
	})
}

// WithFormat sets the audio format.
// Supported formats: "F32", "S32", "S24", "S16", "U8"
// For unmarshaller: preferred format (device will use if supported).
// For marshaller: target format for conversion if needed.
func WithFormat(format string) types.Option {
	format = strings.ToUpper(strings.TrimSpace(format))
	return newOption(func(cfg *config) {
		cfg.device.format = format
	})
}

// WithAudioDevice registers an audio device by ID.
// Can be called multiple times to specify multiple devices.
// For unmarshaller: capture devices (input).
// For marshaller: playback devices (output).
func WithAudioDevice(deviceID string) types.Option {
	deviceID = strings.TrimSpace(deviceID)
	return newOption(func(cfg *config) {
		if deviceID != "" {
			cfg.device.deviceIDs = append(cfg.device.deviceIDs, deviceID)
		}
	})
}

// WithCaptureDevice registers a capture (input) device by ID.
// Can be called multiple times for multiple devices.
func WithCaptureDevice(deviceID string) types.Option {
	deviceID = strings.TrimSpace(deviceID)
	return newOption(func(cfg *config) {
		if deviceID != "" {
			cfg.device.deviceType = "capture"
			cfg.device.deviceIDs = append(cfg.device.deviceIDs, deviceID)
		}
	})
}

// WithPlaybackDevice registers a playback (output) device by ID.
// Can be called multiple times for multiple devices.
func WithPlaybackDevice(deviceID string) types.Option {
	deviceID = strings.TrimSpace(deviceID)
	return newOption(func(cfg *config) {
		if deviceID != "" {
			cfg.device.deviceType = "playback"
			cfg.device.deviceIDs = append(cfg.device.deviceIDs, deviceID)
		}
	})
}

// WithFrameSize sets the number of frames per read/write operation.
// 0 = auto (use device default or optimal size).
func WithFrameSize(frames int) types.Option {
	return newOption(func(cfg *config) {
		if frames > 0 {
			cfg.stream.frameSize = frames
		}
	})
}

// WithBufferSize sets the buffer size in frames.
// 0 = auto (3 seconds worth of audio).
func WithBufferSize(size int) types.Option {
	return newOption(func(cfg *config) {
		if size > 0 {
			cfg.stream.bufferSize = size
		}
	})
}

// WithBestEffort enables best-effort synchronization for multi-device operations.
// When enabled, frames may be dropped if synchronization cannot be maintained.
func WithBestEffort(enable bool) types.Option {
	return newOption(func(cfg *config) {
		cfg.stream.allowBestEffort = enable
	})
}

func applyOptions(base types.Options, cfg config, opts []types.Option) (types.Options, config) {
	local := base
	localCfg := cfg
	// Copy focused configs
	if len(cfg.wav.paths) > 0 {
		localCfg.wav.paths = append([]string(nil), cfg.wav.paths...)
	}
	if len(cfg.device.deviceIDs) > 0 {
		localCfg.device.deviceIDs = append([]string(nil), cfg.device.deviceIDs...)
	}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		opt.Apply(&local)
		if cfgOpt, ok := opt.(configOption); ok {
			cfgOpt.applyConfig(&localCfg)
		}
	}
	if local.Context != nil {
		localCfg.ctx = local.Context
	}
	localCfg.ReleaseAfterProcessing = local.ReleaseAfterProcessing
	return local, localCfg
}
