package destination

import (
	"log/slog"

	"github.com/spf13/cobra"
)

// AddDestinationFlags adds destination flags to a cobra command.
// Call this on any command that needs destination flags.
func AddDestinationFlags(cmd *cobra.Command) {
	cmd.Flags().Bool("no-display", false, "Omit display window")
	cmd.Flags().String("title", "Display", "Display window title")
	cmd.Flags().Int("window-width", 0, "Display window width (0 = auto)")
	cmd.Flags().Int("window-height", 0, "Display window height (0 = auto)")
	cmd.Flags().String("output", "", "Output video file path (e.g., output.mp4)")
	// Note: Intent destination flags will be registered separately if DNDM is enabled
}

// NewFromCobra creates all destination types based on flags from a cobra command.
// Returns destinations that should be enabled based on flags.
func NewFromCobra(cmd *cobra.Command) []Destination {
	// Read flags from cobra command
	noDisplay, _ := cmd.Flags().GetBool("no-display")
	title, _ := cmd.Flags().GetString("title")
	width, _ := cmd.Flags().GetInt("window-width")
	height, _ := cmd.Flags().GetInt("window-height")
	outputPath, _ := cmd.Flags().GetString("output")

	slog.Info("Creating destinations from cobra command",
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
		disp := NewDisplay().(*displayDestination)
		// Set destination-specific values from flags
		setDisplayParams(disp, title, width, height)
		dests = append(dests, disp)
	} else {
		slog.Info("Display destination disabled (--no-display)")
	}

	// Video file destination (enabled if --output is set)
	if outputPath != "" {
		slog.Info("Adding video file destination", "path", outputPath)
		vid := NewVideo().(*videoDestination)
		setVideoParams(vid, outputPath)
		dests = append(dests, vid)
	}

	// DNDM intent destination is not created here - it requires a router
	// Callers should use NewIntentFromCobra(cmd, router) separately

	slog.Info("Destinations created", "count", len(dests))
	return dests
}
