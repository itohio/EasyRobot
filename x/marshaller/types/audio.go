package types

// AudioDeviceInfo contains information about an audio device
type AudioDeviceInfo struct {
	// ID is the device identifier (implementation-specific, typically a base64-encoded hash)
	ID string

	// Name is the human-readable device name
	Name string

	// IsDefault indicates whether this is the default device for its type
	IsDefault bool

	// MinChannels is the minimum number of channels supported
	MinChannels int

	// MaxChannels is the maximum number of channels supported
	MaxChannels int

	// MinSampleRate is the minimum sample rate supported (Hz)
	MinSampleRate int

	// MaxSampleRate is the maximum sample rate supported (Hz)
	MaxSampleRate int

	// Formats lists supported audio formats (e.g., "F32", "S32", "S24", "S16", "U8")
	Formats []string

	// DeviceType indicates the device type: "capture" (input) or "playback" (output)
	DeviceType string

	// Metadata contains additional implementation-specific information
	Metadata map[string]any
}
