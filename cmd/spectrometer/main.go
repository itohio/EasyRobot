package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"log/slog"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/itohio/EasyRobot/cmd/spectrometer/cameras"
	"github.com/itohio/EasyRobot/cmd/spectrometer/ports"
)

var (
	verbose = flag.Int("v", 0, "Set log verbosity level (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG, 4=TRACE)")
	vv      = flag.Bool("vv", false, "Shortcut for -v=4 (TRACE level, maximum verbosity)")
)

func main() {
	// Count -v flags before flag.Parse() consumes them
	verboseCount := 0
	hasVV := false
	for _, arg := range os.Args {
		if arg == "-v" {
			verboseCount++
		} else if arg == "-vv" {
			hasVV = true
			verboseCount = 2
			break
		}
	}

	// Register source and destination flags (they will be parsed)
	source.RegisterAllFlags()
	destination.RegisterAllFlags()

	flag.Parse()

	// Setup logging based on verbose level
	logLevel := *verbose
	if hasVV {
		logLevel = 4
	} else if *verbose == 0 && verboseCount > 0 {
		logLevel = verboseCount
	}

	setupLogging(logLevel)

	// Get command from args
	args := flag.Args()
	if len(args) == 0 {
		printUsage()
		os.Exit(1)
	}

	command := args[0]
	commandArgs := args[1:]

	// Create context with cancellation
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// Route to command handler
	var err error
	switch command {
	case "cameras":
		err = runCameras(ctx, commandArgs)
	case "ports":
		err = runPorts(ctx, commandArgs)
	case "calibrate":
		if len(commandArgs) > 0 && commandArgs[0] == "camera" {
			err = runCalibrateCamera(ctx, commandArgs[1:])
		} else {
			fmt.Fprintf(os.Stderr, "Error: 'calibrate' requires 'camera' subcommand\n")
			printUsage()
			os.Exit(1)
		}
	case "measure":
		err = runMeasure(ctx, commandArgs)
	case "measure-emissivity":
		err = runMeasureEmissivity(ctx, commandArgs)
	case "measure-transmission":
		err = runMeasureTransmission(ctx, commandArgs)
	case "measure-reflectance":
		err = runMeasureReflectance(ctx, commandArgs)
	case "measure-raman":
		err = runMeasureRaman(ctx, commandArgs)
	case "measure-fluorescence":
		err = runMeasureFluorescence(ctx, commandArgs)
	case "freerun":
		err = runFreerun(ctx, commandArgs)
	case "help", "-h", "--help":
		printUsage()
		os.Exit(0)
	default:
		fmt.Fprintf(os.Stderr, "Error: unknown command '%s'\n", command)
		printUsage()
		os.Exit(1)
	}

	if err != nil {
		slog.Error("Command failed", "command", command, "error", err)
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

func printUsage() {
	fmt.Fprintf(os.Stderr, `Usage: spectrometer <command> [options]

Commands:
  cameras                    List available cameras
  ports                      List available serial ports (COM ports)
  calibrate camera           Calibrate camera spectrometer
  measure                    Measure spectrum (CR30 device)
  measure-emissivity         Measure emissivity spectrum
  measure-transmission       Measure transmission spectrum
  measure-reflectance        Measure reflectance spectrum
  measure-raman              Measure Raman spectrum (future)
  measure-fluorescence       Measure fluorescence spectrum (future)
  freerun                    Run live spectrum display

Common flags:
  -v=N                       Set log verbosity (0=ERROR, 1=WARN, 2=INFO, 3=DEBUG, 4=TRACE)
  -vv                        Shortcut for -v=4 (maximum verbosity)
  -h, --help, help           Show this help message

Use 'spectrometer <command> -h' for command-specific help.
`)
}

// Command handlers (to be implemented in respective command files)
func runCameras(ctx context.Context, args []string) error {
	return cameras.Run(ctx)
}

func runPorts(ctx context.Context, args []string) error {
	return ports.Run(ctx)
}

func runCalibrateCamera(ctx context.Context, args []string) error {
	// TODO: Implement in calibrate/calibrate.go
	return fmt.Errorf("not implemented yet")
}

func runMeasure(ctx context.Context, args []string) error {
	// TODO: Implement in measure/measure.go
	return fmt.Errorf("not implemented yet")
}

func runMeasureEmissivity(ctx context.Context, args []string) error {
	// TODO: Implement in measure/emissivity.go
	return fmt.Errorf("not implemented yet")
}

func runMeasureTransmission(ctx context.Context, args []string) error {
	// TODO: Implement in measure/transmission.go
	return fmt.Errorf("not implemented yet")
}

func runMeasureReflectance(ctx context.Context, args []string) error {
	// TODO: Implement in measure/reflectance.go
	return fmt.Errorf("not implemented yet")
}

func runMeasureRaman(ctx context.Context, args []string) error {
	// TODO: Implement in measure/raman.go
	return fmt.Errorf("not implemented yet")
}

func runMeasureFluorescence(ctx context.Context, args []string) error {
	// TODO: Implement in measure/fluorescence.go
	return fmt.Errorf("not implemented yet")
}

func runFreerun(ctx context.Context, args []string) error {
	// TODO: Implement in freerun/freerun.go
	return fmt.Errorf("not implemented yet")
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

