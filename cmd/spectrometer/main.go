package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"log/slog"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/itohio/EasyRobot/cmd/spectrometer/cameras"
	"github.com/itohio/EasyRobot/cmd/spectrometer/internal/calibrate"
	"github.com/itohio/EasyRobot/cmd/spectrometer/measure"
	"github.com/itohio/EasyRobot/cmd/spectrometer/ports"
	"github.com/spf13/cobra"
)

var (
	verbose int
	vv      bool
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "spectrometer",
		Short: "Spectrometer application",
		Long:  "Command-line tool for spectrometer operations: listing cameras and ports, calibrating cameras, measuring spectra, and running live displays.",
	}

	// Add PersistentFlags for verbose flags (available to all subcommands)
	rootCmd.PersistentFlags().IntVar(&verbose, "v", 0, "Set log verbosity level (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG, 4=TRACE)")
	rootCmd.PersistentFlags().BoolVar(&vv, "vv", false, "Shortcut for -v=4 (TRACE level, maximum verbosity)")

	// Setup logging in PersistentPreRun (runs before any subcommand)
	rootCmd.PersistentPreRun = func(cmd *cobra.Command, args []string) {
		logLevel := verbose
		if vv {
			logLevel = 4
		}
		setupLogging(logLevel)
	}

	// Add subcommands
	rootCmd.AddCommand(newCamerasCmd())
	rootCmd.AddCommand(newPortsCmd())
	rootCmd.AddCommand(newCalibrateCmd())
	rootCmd.AddCommand(newMeasureCmd())
	rootCmd.AddCommand(newMeasureEmissivityCmd())
	rootCmd.AddCommand(newMeasureTransmissionCmd())
	rootCmd.AddCommand(newMeasureReflectanceCmd())
	rootCmd.AddCommand(newMeasureRamanCmd())
	rootCmd.AddCommand(newMeasureFluorescenceCmd())
	rootCmd.AddCommand(newFreerunCmd())

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func setupLogging(level int) {
	var logLevel slog.Level
	switch level {
	case 0:
		logLevel = slog.LevelError
	case 1:
		logLevel = slog.LevelWarn
	case 2:
		logLevel = slog.LevelInfo
	case 3:
		logLevel = slog.LevelDebug
	case 4:
		logLevel = slog.LevelDebug // TRACE not available, use DEBUG
	default:
		logLevel = slog.LevelInfo
	}

	opts := &slog.HandlerOptions{
		Level: logLevel,
	}
	handler := slog.NewTextHandler(os.Stderr, opts)
	slog.SetDefault(slog.New(handler))
}

// Command constructors for cobra subcommands
func newCamerasCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "cameras",
		Short: "List available cameras",
		Long:  "List all available cameras with their parameters and capabilities.",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			if ctx == nil {
				ctx, _ = signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
			}
			return cameras.Run(ctx)
		},
	}
}

func newPortsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "ports",
		Short: "List available serial ports",
		Long:  "List all available serial ports (COM ports) on the system.",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			if ctx == nil {
				ctx, _ = signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
			}
			return ports.Run(ctx)
		},
	}
}

func newCalibrateCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "calibrate",
		Short: "Calibrate camera spectrometer",
		Long:  "Calibration commands for camera spectrometers.",
	}

	cameraCmd := &cobra.Command{
		Use:   "camera",
		Short: "Calibrate camera spectrometer",
		Long:  "Calibrate a camera spectrometer using a calibration target. Captures image, displays it, and detects spectral strip bounding box.",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			if ctx == nil {
				ctx, _ = signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
			}
			return calibrate.RunCameraCalibration(ctx, cmd)
		},
	}

	// Register source and destination flags for camera calibration
	source.AddSourceFlags(cameraCmd)
	destination.AddDestinationFlags(cameraCmd)

	cmd.AddCommand(cameraCmd)
	return cmd
}

func newMeasureCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "measure",
		Short: "Measure spectrum (CR30 device)",
		Long:  "Measure spectrum using CR30 device. Connects to device via serial port and performs measurement.",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			if ctx == nil {
				ctx, _ = signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
			}
			return measure.Run(ctx, cmd)
		},
	}

	// Add measure-specific flags
	cmd.Flags().String("device", "cr30", "Device type (cr30)")
	cmd.Flags().String("port", "", "Serial port (e.g., COM3, /dev/ttyUSB0)")
	cmd.Flags().Int("baud", 19200, "Serial port baud rate")
	cmd.Flags().Int("readings", 1, "Number of readings to average (1 = single measurement)")
	cmd.Flags().String("output", "", "Output file path (format auto-detected from extension)")
	cmd.Flags().String("patches", "", "Path to patch file (YAML/JSON) for patch-based measurement")
	cmd.Flags().Bool("button", false, "Use button-initiated measurements (wait for device button press)")

	return cmd
}

func newMeasureEmissivityCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "measure-emissivity",
		Short: "Measure emissivity spectrum",
		Long:  "Measure emissivity spectrum (not yet implemented).",
		RunE: func(cmd *cobra.Command, args []string) error {
			// TODO: Implement in measure/emissivity.go
			return fmt.Errorf("not implemented yet")
		},
	}
}

func newMeasureTransmissionCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "measure-transmission",
		Short: "Measure transmission spectrum",
		Long:  "Measure transmission spectrum (not yet implemented).",
		RunE: func(cmd *cobra.Command, args []string) error {
			// TODO: Implement in measure/transmission.go
			return fmt.Errorf("not implemented yet")
		},
	}
}

func newMeasureReflectanceCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "measure-reflectance",
		Short: "Measure reflectance spectrum",
		Long:  "Measure reflectance spectrum (not yet implemented).",
		RunE: func(cmd *cobra.Command, args []string) error {
			// TODO: Implement in measure/reflectance.go
			return fmt.Errorf("not implemented yet")
		},
	}
}

func newMeasureRamanCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "measure-raman",
		Short: "Measure Raman spectrum",
		Long:  "Measure Raman spectrum (not yet implemented).",
		RunE: func(cmd *cobra.Command, args []string) error {
			// TODO: Implement in measure/raman.go
			return fmt.Errorf("not implemented yet")
		},
	}
}

func newMeasureFluorescenceCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "measure-fluorescence",
		Short: "Measure fluorescence spectrum",
		Long:  "Measure fluorescence spectrum (not yet implemented).",
		RunE: func(cmd *cobra.Command, args []string) error {
			// TODO: Implement in measure/fluorescence.go
			return fmt.Errorf("not implemented yet")
		},
	}
}

func newFreerunCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "freerun",
		Short: "Run live spectrum display",
		Long:  "Run live spectrum display (not yet implemented).",
		RunE: func(cmd *cobra.Command, args []string) error {
			// TODO: Implement in freerun/freerun.go
			return fmt.Errorf("not implemented yet")
		},
	}
}

// detectFormat detects file format from extension
func detectFormat(filename string) string {
	filename = strings.ToLower(filename)
	if strings.HasSuffix(filename, ".pb") || strings.HasSuffix(filename, ".proto") {
		return "pb"
	}
	if strings.HasSuffix(filename, ".json") {
		return "json"
	}
	if strings.HasSuffix(filename, ".yaml") || strings.HasSuffix(filename, ".yml") {
		return "yaml"
	}
	if strings.HasSuffix(filename, ".csv") {
		return "csv"
	}
	return "yaml" // default
}
