package measure

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"log/slog"

	"github.com/itohio/EasyRobot/cmd/spectrometer/internal/obtainer"
	"github.com/itohio/EasyRobot/types/math"
	"github.com/itohio/EasyRobot/types/spectrometer"
	jsonMarshaller "github.com/itohio/EasyRobot/x/marshaller/json"
	marshallerProto "github.com/itohio/EasyRobot/x/marshaller/proto"
	"github.com/itohio/EasyRobot/x/math/colorscience"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Run executes the measure command
func Run(ctx context.Context, cmd *cobra.Command) error {
	// Read flags from cobra command
	device, _ := cmd.Flags().GetString("device")
	port, _ := cmd.Flags().GetString("port")
	baud, _ := cmd.Flags().GetInt("baud")
	readings, _ := cmd.Flags().GetInt("readings")
	output, _ := cmd.Flags().GetString("output")
	patches, _ := cmd.Flags().GetString("patches")
	buttonMode, _ := cmd.Flags().GetBool("button")

	// Validate device type
	if device != "cr30" {
		return fmt.Errorf("unsupported device type: %q (only 'cr30' is supported)", device)
	}

	// Validate port
	if port == "" {
		return fmt.Errorf("--port is required for CR30 device")
	}

	// Validate readings
	if readings < 1 {
		return fmt.Errorf("--readings must be >= 1, got %d", readings)
	}

	// Create obtainer configuration
	config := map[string]interface{}{
		"port": port,
		"baud": baud,
	}

	// Create obtainer
	obt, err := obtainer.NewObtainer(ctx, device, config)
	if err != nil {
		return fmt.Errorf("failed to create obtainer: %w", err)
	}
	defer obt.Disconnect()

	// Connect to device
	if err := obt.Connect(ctx); err != nil {
		return fmt.Errorf("failed to connect to device: %w", err)
	}

	// Get device info
	info := obt.DeviceInfo()
	slog.Info("Connected to device",
		"name", info.Name,
		"model", info.Model,
		"serial", info.Serial,
		"firmware", info.Firmware,
	)

	// Perform measurement
	var spd colorscience.SPD
	if buttonMode {
		// Button-initiated measurement using Start() callback pattern
		spd, err = measureWithButton(ctx, obt, readings)
		if err != nil {
			return fmt.Errorf("button-initiated measurement failed: %w", err)
		}
	} else if readings == 1 {
		// Single PC-initiated measurement
		spd, err = measureSingle(ctx, obt)
		if err != nil {
			return fmt.Errorf("single measurement failed: %w", err)
		}
	} else {
		// Averaged PC-initiated measurement
		spd, err = measureAveraged(ctx, obt, readings)
		if err != nil {
			return fmt.Errorf("averaged measurement failed: %w", err)
		}
	}

	// Export if output specified
	if output != "" {
		if err := exportMeasurement(ctx, spd, output); err != nil {
			return fmt.Errorf("export failed: %w", err)
		}
		slog.Info("Measurement exported", "path", output)
	} else {
		// Print measurement summary
		printMeasurementSummary(spd)
	}

	// Note: patches flag is not yet used, but kept for future implementation
	_ = patches

	return nil
}

// measureSingle performs a single measurement
func measureSingle(ctx context.Context, obt obtainer.Obtainer) (colorscience.SPD, error) {
	numWl := obt.NumWavelengths()
	dst := mat.New(2, numWl)

	if err := obt.Measure(ctx, dst); err != nil {
		return colorscience.SPD{}, fmt.Errorf("measure failed: %w", err)
	}

	// Extract wavelengths and values
	wavelengths := dst.Row(0).(vec.Vector)
	values := dst.Row(1).(vec.Vector)

	// Create SPD
	spd := colorscience.NewSPD(wavelengths, values)
	return spd, nil
}

// measureWithButton performs button-initiated measurements using Start() callback pattern
func measureWithButton(ctx context.Context, obt obtainer.Obtainer, numReadings int) (colorscience.SPD, error) {
	if numReadings < 1 {
		numReadings = 1
	}

	numWl := obt.NumWavelengths()
	readings := make([]colorscience.SPD, 0, numReadings)
	readingsChan := make(chan colorscience.SPD, numReadings)
	errChan := make(chan error, 1)

	// Callback to collect measurements
	callback := func(spd colorscience.SPD, err error) error {
		if err != nil {
			select {
			case errChan <- err:
			default:
			}
			return err // Stop listening on error
		}

		select {
		case readingsChan <- spd:
			slog.Info("Button measurement received", "count", len(readings)+1, "target", numReadings)
		default:
			// Channel full, shouldn't happen
		}

		// Stop when we have enough readings
		if len(readingsChan) >= numReadings {
			return fmt.Errorf("done") // Signal to stop
		}

		return nil // Continue listening
	}

	// Start listener
	slog.Info("Waiting for button-initiated measurements", "count", numReadings, "press device button to measure")
	if err := obt.Start(ctx, callback); err != nil {
		return colorscience.SPD{}, fmt.Errorf("failed to start measurement listener: %w", err)
	}

	// Collect measurements
	for len(readings) < numReadings {
		select {
		case spd := <-readingsChan:
			readings = append(readings, spd)
		case err := <-errChan:
			_ = obt.Stop(ctx)
			if err.Error() == "done" {
				break // Done collecting
			}
			return colorscience.SPD{}, err
		case <-ctx.Done():
			_ = obt.Stop(ctx)
			return colorscience.SPD{}, ctx.Err()
		}
	}

	// Stop listener
	if err := obt.Stop(ctx); err != nil {
		return colorscience.SPD{}, fmt.Errorf("failed to stop measurement listener: %w", err)
	}

	// Average measurements if multiple
	if len(readings) == 1 {
		return readings[0], nil
	}

	// Average multiple readings
	wavelengths := readings[0].Wavelengths()
	averagedValues := vec.New(numWl)

	for _, reading := range readings {
		values := reading.Values()
		for j := 0; j < numWl; j++ {
			averagedValues[j] += values[j]
		}
	}

	// Divide by count
	for j := 0; j < numWl; j++ {
		averagedValues[j] /= float32(len(readings))
	}

	return colorscience.NewSPD(wavelengths, averagedValues), nil
}

// measureAveraged performs multiple measurements and averages them
func measureAveraged(ctx context.Context, obt obtainer.Obtainer, numReadings int) (colorscience.SPD, error) {
	numWl := obt.NumWavelengths()

	// Preallocate matrices for all readings
	readings := make([]mat.Matrix, numReadings)
	for i := range readings {
		readings[i] = mat.New(2, numWl)
	}

	// Perform all measurements
	slog.Info("Performing measurements", "count", numReadings)
	for i := 0; i < numReadings; i++ {
		if err := obt.Measure(ctx, readings[i]); err != nil {
			return colorscience.SPD{}, fmt.Errorf("measurement %d/%d failed: %w", i+1, numReadings, err)
		}
		slog.Debug("Measurement completed", "index", i+1, "total", numReadings)
	}

	// Average the measurements
	wavelengths := readings[0].Row(0).(vec.Vector)
	averagedValues := vec.New(numWl)

	// Sum all values
	for i := 0; i < numReadings; i++ {
		values := readings[i].Row(1).(vec.Vector)
		for j := 0; j < numWl; j++ {
			averagedValues[j] += values[j]
		}
	}

	// Divide by count to get average
	for j := 0; j < numWl; j++ {
		averagedValues[j] /= float32(numReadings)
	}

	// Create averaged SPD
	spd := colorscience.NewSPD(wavelengths, averagedValues)
	return spd, nil
}

// exportMeasurement exports the measurement to a file
// Format is auto-detected from file extension: .csv, .json, .pb/.proto
func exportMeasurement(ctx context.Context, spd colorscience.SPD, path string) error {
	format := detectFormat(path)
	slog.Debug("Exporting measurement", "path", path, "format", format)

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer file.Close()

	switch format {
	case "csv":
		return exportCSV(ctx, file, spd)
	case "json":
		return exportJSON(ctx, file, spd)
	case "pb", "proto":
		return exportProto(ctx, file, spd)
	default:
		return fmt.Errorf("unsupported export format: %s (supported: csv, json, pb)", format)
	}
}

// detectFormat detects file format from extension
func detectFormat(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".csv":
		return "csv"
	case ".json":
		return "json"
	case ".pb", ".proto":
		return "pb"
	default:
		return "json" // default to JSON
	}
}

// exportCSV exports SPD to CSV format (wavelength, intensity columns)
func exportCSV(ctx context.Context, w *os.File, spd colorscience.SPD) error {
	wavelengths := spd.Wavelengths()
	values := spd.Values()

	if wavelengths == nil || values == nil || wavelengths.Len() == 0 {
		return fmt.Errorf("SPD is empty")
	}

	writer := csv.NewWriter(w)
	defer writer.Flush()

	// Write header
	if err := writer.Write([]string{"wavelength", "intensity"}); err != nil {
		return fmt.Errorf("failed to write CSV header: %w", err)
	}

	// Write data rows
	for i := 0; i < wavelengths.Len(); i++ {
		row := []string{
			fmt.Sprintf("%.3f", wavelengths[i]),
			fmt.Sprintf("%.6f", values[i]),
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("failed to write CSV row %d: %w", i, err)
		}
	}

	return nil
}

// exportJSON exports SPD to JSON format using SpectrumMeasurement proto message
func exportJSON(ctx context.Context, w *os.File, spd colorscience.SPD) error {
	measurement := spdToMeasurement(spd)

	marshaller := jsonMarshaller.NewMarshaller()
	if err := marshaller.Marshal(w, measurement); err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	return nil
}

// exportProto exports SPD to protobuf binary format
func exportProto(ctx context.Context, w *os.File, spd colorscience.SPD) error {
	measurement := spdToMeasurement(spd)

	marshaller := marshallerProto.NewMarshaller()
	if err := marshaller.Marshal(w, measurement); err != nil {
		return fmt.Errorf("failed to marshal protobuf: %w", err)
	}

	return nil
}

// spdToMeasurement converts SPD to SpectrumMeasurement proto message
func spdToMeasurement(spd colorscience.SPD) *spectrometer.SpectrumMeasurement {
	wavelengths := spd.Wavelengths()
	values := spd.Values()

	if wavelengths == nil || values == nil || wavelengths.Len() == 0 {
		return nil
	}

	// Convert SPD matrix to proto Matrix (row-major flattened)
	// SPD is 2 rows: row 0 = wavelengths, row 1 = values
	protoMatrix := &math.Matrix{
		Rows: 2,
		Cols: int32(wavelengths.Len()),
		Data: make([]float32, 2*wavelengths.Len()),
	}

	// Flatten row-major: [wavelengths..., values...]
	for i := 0; i < wavelengths.Len(); i++ {
		protoMatrix.Data[i] = wavelengths[i]              // Row 0
		protoMatrix.Data[wavelengths.Len()+i] = values[i] // Row 1
	}

	// Create measurement metadata
	metadata := &spectrometer.MeasurementMetadata{
		Timestamp:  timestamppb.Now(),
		SourceType: "sensor",
		Mode:       spectrometer.MeasurementMode_MODE_SINGLE,
	}

	// Create measurement
	measurement := &spectrometer.SpectrumMeasurement{
		Metadata: metadata,
		Spectrum: protoMatrix,
	}

	return measurement
}

// printMeasurementSummary prints a summary of the measurement
func printMeasurementSummary(spd colorscience.SPD) {
	wavelengths := spd.Wavelengths()
	values := spd.Values()

	if wavelengths == nil || values == nil || wavelengths.Len() == 0 {
		fmt.Println("No measurement data")
		return
	}

	fmt.Printf("\nMeasurement Summary:\n")
	fmt.Printf("  Wavelengths: %d points\n", wavelengths.Len())
	fmt.Printf("  Range: %.1f - %.1f nm\n", wavelengths[0], wavelengths[wavelengths.Len()-1])

	// Find min/max intensity
	minVal := values[0]
	maxVal := values[0]
	for i := 1; i < values.Len(); i++ {
		if values[i] < minVal {
			minVal = values[i]
		}
		if values[i] > maxVal {
			maxVal = values[i]
		}
	}
	fmt.Printf("  Intensity range: %.3f - %.3f\n", minVal, maxVal)
	fmt.Printf("\nFirst 5 points:\n")
	fmt.Printf("  Wavelength (nm)  Intensity\n")
	for i := 0; i < 5 && i < wavelengths.Len(); i++ {
		fmt.Printf("  %12.1f  %10.3f\n", wavelengths[i], values[i])
	}
	if wavelengths.Len() > 5 {
		fmt.Printf("  ... (%d more points)\n", wavelengths.Len()-5)
	}
}
