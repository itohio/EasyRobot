package cameras

import (
	"context"

	"github.com/itohio/EasyRobot/cmd/display/source"
)

// Run lists all available cameras with their capabilities.
// Reuses source.ListCameras() for camera enumeration.
func Run(ctx context.Context) error {
	return source.ListCameras()
}

