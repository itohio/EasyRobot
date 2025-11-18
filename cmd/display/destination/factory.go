package destination

import (
	"flag"
	"log/slog"
)

// RegisterAllFlags registers flags for all destination types.
// Call this before flag.Parse().
func RegisterAllFlags() {
	flag.BoolVar(&noDisplay, "no-display", false, "Omit display window")
	flag.StringVar(&title, "title", "Display", "Display window title")
	flag.IntVar(&width, "window-width", 0, "Display window width (0 = auto)")
	flag.IntVar(&height, "window-height", 0, "Display window height (0 = auto)")
	flag.StringVar(&outputPath, "output", "", "Output video file path (e.g., output.mp4)")
	// Note: Intent destination flags will be registered separately if DNDM is enabled
}

// NewAllDestinations creates all destination types.
// Returns destinations that should be enabled based on flags.
// Must be called after flag.Parse().
func NewAllDestinations() []Destination {
	slog.Info("Creating destinations from flags",
		"no_display", noDisplay,
		"title", title,
		"width", width,
		"height", height,
		"output_path", outputPath,
	)

	var dests []Destination

	// Display destination (always enabled unless --no-display)
	if !noDisplay {
		slog.Info("Adding display destination", "title", title, "width", width, "height", height)
		dests = append(dests, NewDisplay())
	} else {
		slog.Info("Display destination disabled (--no-display)")
	}

	// Video file destination (enabled if --output is set)
	if outputPath != "" {
		slog.Info("Adding video file destination", "path", outputPath)
		dests = append(dests, NewVideo())
	}

	// DNDM intent destination is not created here - it requires a router
	// Callers should use NewIntentFromFlags(router) separately

	slog.Info("Destinations created", "count", len(dests))
	return dests
}
