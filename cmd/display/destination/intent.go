package destination

import (
	"context"
	"fmt"
	"log/slog"

	pbdevices "github.com/itohio/EasyRobot/types/devices"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/dndm"
	"github.com/itohio/dndm/x/bus"
	"github.com/spf13/cobra"
)

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
	// Flags are registered by AddIntentFlags() separately
	// This method exists for interface compliance but does nothing
}

func (i *IntentDestination) Start(ctx context.Context) error {
	if i.started {
		slog.Warn("Intent destination already started")
		return fmt.Errorf("intent destination already started")
	}
	if len(i.routes) == 0 {
		slog.Info("Intent destination disabled (no routes)")
		return nil // Intent is disabled
	}
	if i.router == nil {
		slog.Info("Intent destination disabled (no router)")
		return nil // Router not provided, intent disabled
	}

	slog.Info("Starting intent destination", "routes", i.routes, "count", len(i.routes))
	i.ctx = ctx
	i.started = true

	// Create producers for each route
	for _, route := range i.routes {
		slog.Debug("Creating producer for route", "route", route)
		producer, err := bus.NewProducer[*pbdevices.LIDARReading](ctx, i.router, route)
		if err != nil {
			slog.Error("Failed to create producer for route", "route", route, "err", err)
			// Clean up already created producers
			for _, p := range i.producers {
				p.Close()
			}
			i.producers = make(map[string]*bus.Producer[*pbdevices.LIDARReading])
			return fmt.Errorf("failed to create producer for route %s: %w", route, err)
		}
		i.producers[route] = producer
		slog.Info("Producer created for route", "route", route)
	}

	slog.Info("Intent destination started successfully", "producers", len(i.producers))
	return nil
}

func (i *IntentDestination) AddFrame(frame types.Frame) error {
	if !i.started || len(i.routes) == 0 || i.router == nil {
		return nil // Intent is disabled or not started
	}

	slog.Debug("Intent destination received frame", "frame_index", frame.Index, "routes", len(i.routes))
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

	slog.Debug("Publishing LIDARReading to intent routes", "routes", len(i.producers))
	var lastErr error
	for route, producer := range i.producers {
		slog.Debug("Sending LIDARReading to route", "route", route)
		if err := producer.Send(i.ctx, reading); err != nil {
			slog.Error("Failed to send LIDARReading to route", "route", route, "err", err)
			lastErr = fmt.Errorf("failed to send to route %s: %w", route, err)
			// Continue to other routes
		} else {
			slog.Debug("LIDARReading sent successfully to route", "route", route)
		}
	}

	return lastErr
}

func (i *IntentDestination) Close() error {
	slog.Info("Closing intent destination", "producers", len(i.producers))
	for route, producer := range i.producers {
		slog.Debug("Closing producer for route", "route", route)
		producer.Close()
	}
	i.producers = make(map[string]*bus.Producer[*pbdevices.LIDARReading])
	i.started = false
	slog.Info("Intent destination closed")
	return nil
}

// AddIntentFlags adds intent flags to a cobra command.
// Call this on any command that needs intent flags (if DNDM support is enabled).
func AddIntentFlags(cmd *cobra.Command) {
	cmd.Flags().StringArray("intent", nil, "DNDM intent route (can repeat)")
}

// NewIntentFromCobra creates an intent destination from flags in a cobra command.
// If router is nil, intent publishing will be disabled.
func NewIntentFromCobra(cmd *cobra.Command, router *dndm.Router) Destination {
	intentRoutes, _ := cmd.Flags().GetStringArray("intent")
	if len(intentRoutes) == 0 {
		slog.Debug("No intent routes specified")
		return nil
	}
	slog.Info("Creating intent destination from cobra command", "routes", intentRoutes, "router_provided", router != nil)
	dest := NewIntentWithRouter(router).(*IntentDestination)
	dest.routes = intentRoutes
	return dest
}
