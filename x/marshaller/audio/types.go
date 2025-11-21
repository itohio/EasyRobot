package audio

// This file contains internal types used by the audio marshaller package.
// Public types are defined in x/marshaller/types package.

// sourceKind represents the type of audio source.
type sourceKind int

const (
	sourceKindUnknown sourceKind = iota
	sourceKindWAVFile
	sourceKindAudioDevice
	sourceKindWAVFileList // Multiple WAV files
)

// sourceSpec describes an audio source (WAV file or device).
type sourceSpec struct {
	Kind   sourceKind
	Path   string   // WAV file path (for WAV sources)
	Device *deviceSpec // Device specification (for device sources)
	Files  []string // Multiple WAV file paths (for file list)
}

// deviceSpec describes an audio device configuration.
type deviceSpec struct {
	ID         string // Device ID (base64-encoded hash from malgo)
	DeviceType string // "capture" or "playback"
	SampleRate int
	Channels   int
	Format     string // "F32", "S32", "S24", "S16", "U8"
}

