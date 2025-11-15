package destination

import (
	"context"
	"flag"
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// FlagArray is a custom flag type for repeatable flags (used by DNDM).
type FlagArray []string

// String returns a string representation of the flag array.
func (f *FlagArray) String() string {
	if f == nil || len(*f) == 0 {
		return ""
	}
	result := ""
	for i, v := range *f {
		if i > 0 {
			result += ","
		}
		result += v
	}
	return result
}

// Set adds a value to the flag array.
func (f *FlagArray) Set(value string) error {
	*f = append(*f, value)
	return nil
}

var intentRoutes FlagArray

// intentDestination implements Destination for DNDM intent channels.
// This is a placeholder implementation - full DNDM integration TBD.
type intentDestination struct {
	ctx     context.Context
	routes  []string
	started bool
}

// NewIntent creates a new DNDM intent destination.
func NewIntent() Destination {
	return &intentDestination{}
}

func (i *intentDestination) RegisterFlags() {
	// Flags are registered by RegisterIntentFlags() separately
	// This method exists for interface compliance but does nothing
}

func (i *intentDestination) Start(ctx context.Context) error {
	if i.started {
		return fmt.Errorf("intent destination already started")
	}
	if len(i.routes) == 0 {
		return nil // Intent is disabled
	}

	i.ctx = ctx
	i.started = true

	// TODO: Implement DNDM intent producer
	// 1. Connect to DNDM network
	// 2. Create Intent for each route
	// 3. Wait for Interest matches
	// 4. Convert types.Frame → proto Frame
	// 5. Send via intent.Send()

	return nil
}

func (i *intentDestination) AddFrame(frame types.Frame) error {
	if !i.started || len(i.routes) == 0 {
		return nil // Intent is disabled or not started
	}

	// TODO: Implement frame publishing to DNDM
	// 1. Convert types.Frame → proto Frame
	// 2. Check if Interest matches are available
	// 3. Send frame via intent.Send() for each route

	_ = frame // Suppress unused for now
	return fmt.Errorf("DNDM intent destination not yet implemented")
}

func (i *intentDestination) Close() error {
	i.started = false
	return nil
}

// RegisterIntentFlags registers flags for DNDM intent destination.
// Call this before flag.Parse() if DNDM support is enabled.
func RegisterIntentFlags() {
	flag.Var(&intentRoutes, "intent", "DNDM intent route (can repeat)")
}

// NewIntentFromFlags creates an intent destination from flags.
// Must be called after flag.Parse().
func NewIntentFromFlags() Destination {
	if len(intentRoutes) == 0 {
		return nil
	}
	dest := NewIntent().(*intentDestination)
	dest.routes = []string(intentRoutes)
	return dest
}

