package audio

import (
	"context"
)

// audioDevice is an alias for deviceInterface - actual implementation is in device_malgo.go
type audioDevice = deviceInterface

// newAudioDevice creates a new audio device wrapper.
func newAudioDevice(id string, deviceType string, cfg deviceConfig, streamCfg streamConfig, ctx context.Context) (deviceInterface, error) {
	// Use malgo implementation
	return newMalgoDevice(id, deviceType, cfg, streamCfg, ctx)
}
