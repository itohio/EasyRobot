package source

import (
	"context"
	"flag"
	"fmt"
)

var interestRoutes FlagArray

// interestSource implements Source for DNDM interest channels.
// This is a placeholder implementation - full DNDM integration TBD.
type interestSource struct {
	baseSource
	routes []string
}

// NewInterest creates a new DNDM interest source.
func NewInterest() Source {
	return &interestSource{}
}

func (s *interestSource) RegisterFlags() {
	// Flags are registered by RegisterInterestFlags() separately
	// This method exists for interface compliance but does nothing
}

func (s *interestSource) Start(ctx context.Context) error {
	if len(s.routes) == 0 {
		return fmt.Errorf("no interest routes specified")
	}

	// TODO: Implement DNDM interest consumer
	// 1. Connect to DNDM network
	// 2. Create Interest for each route
	// 3. Listen for incoming proto messages
	// 4. Convert proto Frame → types.Frame
	// 5. Create FrameStream from received frames

	return fmt.Errorf("DNDM interest source not yet implemented")
}

// RegisterInterestFlags registers flags for DNDM interest source.
// Call this before flag.Parse() if DNDM support is enabled.
func RegisterInterestFlags() {
	flag.Var(&interestRoutes, "interest", "DNDM interest route (can repeat)")
}

// NewInterestFromFlags creates an interest source from flags.
// Must be called after flag.Parse().
// Returns (nil, nil) if no interest routes are specified (not an error).
func NewInterestFromFlags() (Source, error) {
	if len(interestRoutes) == 0 {
		return nil, nil // Not an error, just not specified
	}
	src := NewInterest().(*interestSource)
	src.routes = []string(interestRoutes)
	return src, nil
}

