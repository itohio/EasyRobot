package destination

import (
	"context"
	"flag"
	"fmt"

	pbdevices "github.com/itohio/EasyRobot/types/devices"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/dndm"
	"github.com/itohio/dndm/x/bus"
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

// IntentDestination implements Destination for DNDM intent channels.
type IntentDestination struct {
	ctx       context.Context
	router    *dndm.Router
	routes    []string
	producers map[string]*bus.Producer[*pbdevices.LIDARReading]
	started   bool
}

// NewIntent creates a new DNDM intent destination.
// If router is nil, intent publishing will be disabled.
func NewIntent(router *dndm.Router) Destination {
	return &IntentDestination{
		router:    router,
		producers: make(map[string]*bus.Producer[*pbdevices.LIDARReading]),
	}
}

// NewIntentWithRouter creates a new DNDM intent destination with a router.
// This is a convenience function that sets the router.
func NewIntentWithRouter(router *dndm.Router) Destination {
	dest := NewIntent(router).(*IntentDestination)
	dest.router = router
	return dest
}

func (i *IntentDestination) RegisterFlags() {
	// Flags are registered by RegisterIntentFlags() separately
	// This method exists for interface compliance but does nothing
}

func (i *IntentDestination) Start(ctx context.Context) error {
	if i.started {
		return fmt.Errorf("intent destination already started")
	}
	if len(i.routes) == 0 {
		return nil // Intent is disabled
	}
	if i.router == nil {
		return nil // Router not provided, intent disabled
	}

	i.ctx = ctx
	i.started = true

	// Create producers for each route
	for _, route := range i.routes {
		producer, err := bus.NewProducer[*pbdevices.LIDARReading](ctx, i.router, route)
		if err != nil {
			// Clean up already created producers
			for _, p := range i.producers {
				p.Close()
			}
			i.producers = make(map[string]*bus.Producer[*pbdevices.LIDARReading])
			return fmt.Errorf("failed to create producer for route %s: %w", route, err)
		}
		i.producers[route] = producer
	}

	return nil
}

func (i *IntentDestination) AddFrame(frame types.Frame) error {
	if !i.started || len(i.routes) == 0 || i.router == nil {
		return nil // Intent is disabled or not started
	}

	// Extract LIDARReading from frame metadata if available
	// For now, we'll need to handle this differently since frames don't directly contain LIDARReading
	// This will be handled by the caller converting LIDARReading to Frame
	// For intent destination, we expect the frame to have LIDARReading in metadata
	// or we need a different approach - see SetLIDARReading method below

	return nil
}

// SetLIDARReading publishes a LIDARReading to all registered intent routes.
// This is a convenience method for publishing LIDAR readings directly.
func (i *IntentDestination) SetLIDARReading(reading *pbdevices.LIDARReading) error {
	if !i.started || len(i.producers) == 0 {
		return nil // Intent is disabled or not started
	}

	var lastErr error
	for route, producer := range i.producers {
		if err := producer.Send(i.ctx, reading); err != nil {
			lastErr = fmt.Errorf("failed to send to route %s: %w", route, err)
			// Continue to other routes
		}
	}

	return lastErr
}

func (i *IntentDestination) Close() error {
	for _, producer := range i.producers {
		producer.Close()
	}
	i.producers = make(map[string]*bus.Producer[*pbdevices.LIDARReading])
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
// If router is nil, intent publishing will be disabled.
func NewIntentFromFlags(router *dndm.Router) Destination {
	if len(intentRoutes) == 0 {
		return nil
	}
	dest := NewIntentWithRouter(router).(*IntentDestination)
	dest.routes = []string(intentRoutes)
	return dest
}
