package destination

import (
	"context"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// Destination is the interface for frame destinations.
type Destination interface {
	// RegisterFlags registers command-line flags for this destination.
	RegisterFlags()

	// Start initializes the destination.
	Start(ctx context.Context) error

	// AddFrame adds a frame to the destination.
	AddFrame(frame types.Frame) error

	// Close closes the destination and cleans up resources.
	Close() error
}

