package destination

import (
	"flag"
)

// RegisterAllFlags registers flags for all destination types.
// Call this before flag.Parse().
func RegisterAllFlags() {
	flag.BoolVar(&noDisplay, "no-display", false, "Omit display window")
	flag.StringVar(&title, "title", "Display", "Display window title")
	flag.IntVar(&width, "width", 0, "Display window width (0 = auto)")
	flag.IntVar(&height, "height", 0, "Display window height (0 = auto)")
	flag.StringVar(&outputPath, "output", "", "Output video file path (e.g., output.mp4)")
	// Note: Intent destination flags will be registered separately if DNDM is enabled
}

// NewAllDestinations creates all destination types.
// Returns destinations that should be enabled based on flags.
// Must be called after flag.Parse().
func NewAllDestinations() []Destination {
	var dests []Destination

	// Display destination (always enabled unless --no-display)
	dests = append(dests, NewDisplay())

	// Video file destination (enabled if --output is set)
	if outputPath != "" {
		dests = append(dests, NewVideo())
	}

	// DNDM intent destination (enabled if --intent routes are set)
	if intentDest := NewIntentFromFlags(); intentDest != nil {
		dests = append(dests, intentDest)
	}

	return dests
}

