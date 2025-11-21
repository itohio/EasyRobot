package source

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/spf13/cobra"
)

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
	// Flags are registered by AddInterestFlags() separately
	// This method exists for interface compliance but does nothing
}

func (s *interestSource) Start(ctx context.Context) error {
	if len(s.routes) == 0 {
		slog.Error("No interest routes specified")
		return fmt.Errorf("no interest routes specified")
	}

	slog.Info("Starting interest source", "routes", s.routes, "count", len(s.routes))

	// TODO: Implement DNDM interest consumer
	// 1. Connect to DNDM network
	// 2. Create Interest for each route
	// 3. Listen for incoming proto messages
	// 4. Convert proto Frame → types.Frame
	// 5. Create FrameStream from received frames

	slog.Error("DNDM interest source not yet implemented")
	return fmt.Errorf("DNDM interest source not yet implemented")
}

// AddInterestFlags adds interest flags to a cobra command.
// Call this on any command that needs interest flags (if DNDM support is enabled).
func AddInterestFlags(cmd *cobra.Command) {
	cmd.Flags().StringArray("interest", nil, "DNDM interest route (can repeat)")
}

// NewInterestFromCobra creates an interest source from flags in a cobra command.
// Returns (nil, nil) if no interest routes are specified (not an error).
func NewInterestFromCobra(cmd *cobra.Command) (Source, error) {
	interestRoutes, _ := cmd.Flags().GetStringArray("interest")
	if len(interestRoutes) == 0 {
		slog.Debug("No interest routes specified")
		return nil, nil // Not an error, just not specified
	}
	slog.Debug("Creating interest source from cobra command", "routes", interestRoutes)
	src := NewInterest().(*interestSource)
	src.routes = interestRoutes
	return src, nil
}

