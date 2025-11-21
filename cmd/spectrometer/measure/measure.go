package measure

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"log/slog"

	"github.com/itohio/EasyRobot/cmd/spectrometer/internal/acquirer"
	"github.com/itohio/EasyRobot/cmd/spectrometer/internal/render"
	"github.com/itohio/EasyRobot/types/math"
	"github.com/itohio/EasyRobot/types/spectrometer"
	jsonMarshaller "github.com/itohio/EasyRobot/x/marshaller/json"
	marshallerProto "github.com/itohio/EasyRobot/x/marshaller/proto"
	mathx "github.com/itohio/EasyRobot/x/math"
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

	// Validate device type (check if available)
	availableDevices := acquirer.AvailableDevices()
	deviceSupported := false
	for _, d := range availableDevices {
		if d == device {
			deviceSupported = true
			break
		}
	}
	if !deviceSupported {
		return fmt.Errorf("unsupported device type: %q (available: %v)", device, availableDevices)
	}

	// Validate readings
	if readings < 1 {
		return fmt.Errorf("--readings must be >= 1, got %d", readings)
	}

	// Create acquirer configuration
	config := map[string]interface{}{
		"port": port,
		"baud": baud,
	}

	// Create acquirer
	acq, err := acquirer.NewAcquirer(ctx, device, config)
	if err != nil {
		return fmt.Errorf("failed to create acquirer: %w", err)
	}
	defer acq.Disconnect()

	// Connect to device
	if err := acq.Connect(ctx); err != nil {
		return fmt.Errorf("failed to connect to device: %w", err)
	}

	// Get device info
	info := acq.DeviceInfo()
	slog.Info("Connected to device",
		"name", info.Name,
		"model", info.Model,
		"serial", info.Serial,
		"firmware", info.Firmware,
	)

	// Handle patch-based measurement if patches flag is provided
	if patches != "" {
		return measurePatches(ctx, acq, patches, readings, buttonMode, output)
	}

	// Perform single measurement
	var spd colorscience.SPD
	if buttonMode {
		// Button-initiated measurement using Start() callback pattern
		spd, err = measureWithButton(ctx, acq, readings)
		if err != nil {
			return fmt.Errorf("button-initiated measurement failed: %w", err)
		}
	} else if readings == 1 {
		// Single PC-initiated measurement
		spd, err = measureSingle(ctx, acq)
		if err != nil {
			return fmt.Errorf("single measurement failed: %w", err)
		}
	} else {
		// Averaged PC-initiated measurement
		spd, err = measureAveraged(ctx, acq, readings)
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

	return nil
}

// measureSingle performs a single measurement
func measureSingle(ctx context.Context, acq acquirer.Acquirer) (colorscience.SPD, error) {
	numWl := acq.NumWavelengths()
	dst := mat.New(2, numWl)

	if err := acq.Measure(ctx, dst); err != nil {
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
func measureWithButton(ctx context.Context, acq acquirer.Acquirer, numReadings int) (colorscience.SPD, error) {
	if numReadings < 1 {
		numReadings = 1
	}

	numWl := acq.NumWavelengths()
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
	slog.Info("Waiting for button-initiated measurements", "count", numReadings, "instruction", "press device button to measure")
	if err := acq.Start(ctx, callback); err != nil {
		return colorscience.SPD{}, fmt.Errorf("failed to start measurement listener: %w", err)
	}

	// Collect measurements
	for len(readings) < numReadings {
		select {
		case spd := <-readingsChan:
			readings = append(readings, spd)
		case err := <-errChan:
			_ = acq.Stop(ctx)
			if err.Error() == "done" {
				break // Done collecting
			}
			return colorscience.SPD{}, err
		case <-ctx.Done():
			_ = acq.Stop(ctx)
			return colorscience.SPD{}, ctx.Err()
		}
	}

	// Stop listener
	if err := acq.Stop(ctx); err != nil {
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
func measureAveraged(ctx context.Context, acq acquirer.Acquirer, numReadings int) (colorscience.SPD, error) {
	numWl := acq.NumWavelengths()

	// Preallocate matrices for all readings
	readings := make([]mat.Matrix, numReadings)
	for i := range readings {
		readings[i] = mat.New(2, numWl)
	}

	// Perform all measurements
	slog.Info("Performing measurements", "count", numReadings)
	for i := 0; i < numReadings; i++ {
		if err := acq.Measure(ctx, readings[i]); err != nil {
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

// measurePatches performs patch-based measurements.
// Loads patches from file, measures each patch, calculates colorimetry and Delta E.
// Works with any acquirer that implements the Acquirer interface.
func measurePatches(ctx context.Context, acq acquirer.Acquirer, patchesPath string, readings int, buttonMode bool, output string) error {
	// Load patches
	patches, err := LoadPatches(ctx, patchesPath)
	if err != nil {
		return fmt.Errorf("failed to load patches: %w", err)
	}

	if len(patches.GetPatches()) == 0 {
		return fmt.Errorf("no patches found in file")
	}

	slog.Info("Loaded patches", "count", len(patches.GetPatches()))

	// Default white point (D65)
	whitePoint := colorscience.WhitePointD65_10

	// Measure each patch
	results := make([]render.Patch, 0, len(patches.GetPatches()))
	for i, protoPatch := range patches.GetPatches() {
		slog.Info("Measuring patch", "index", i+1, "total", len(patches.GetPatches()), "name", protoPatch.GetName())

		// Convert proto patch to render patch
		refPatch, err := ConvertProtoPatchToRenderPatch(protoPatch)
		if err != nil {
			return fmt.Errorf("failed to convert patch %q: %w", protoPatch.GetName(), err)
		}

		// Measure spectrum
		var measuredSPD colorscience.SPD
		if buttonMode {
			measuredSPD, err = measureWithButton(ctx, acq, readings)
			if err != nil {
				return fmt.Errorf("button-initiated measurement failed for patch %q: %w", protoPatch.GetName(), err)
			}
		} else if readings == 1 {
			measuredSPD, err = measureSingle(ctx, acq)
			if err != nil {
				return fmt.Errorf("single measurement failed for patch %q: %w", protoPatch.GetName(), err)
			}
		} else {
			measuredSPD, err = measureAveraged(ctx, acq, readings)
			if err != nil {
				return fmt.Errorf("averaged measurement failed for patch %q: %w", protoPatch.GetName(), err)
			}
		}

		// Convert measured SPD to render patch with colorimetry
		measuredPatch, err := ConvertMeasuredSPDToRenderPatch(measuredSPD, refPatch, whitePoint)
		if err != nil {
			return fmt.Errorf("failed to compute colorimetry for patch %q: %w", protoPatch.GetName(), err)
		}

		results = append(results, measuredPatch)

		// Print patch result
		printPatchResult(measuredPatch)
	}

	// Export results if output specified
	if output != "" {
		if err := exportPatchResults(ctx, results, output); err != nil {
			return fmt.Errorf("export failed: %w", err)
		}
		slog.Info("Patch results exported", "path", output, "count", len(results))
	}

	return nil
}

// printPatchResult prints a summary of patch measurement result
func printPatchResult(patch render.Patch) {
	fmt.Printf("\n=== Patch: %s ===\n", patch.Name)
	fmt.Printf("Delta E: %.2f\n", patch.DeltaE)

	// Reference values
	refX, refY, refZ := vec.Vector3D(patch.ReferenceXYZ).XYZ()
	fmt.Printf("Reference XYZ: %.2f %.2f %.2f\n", refX, refY, refZ)
	refL, refA, refB := vec.Vector3D(patch.Reference).XYZ()
	fmt.Printf("Reference LAB: %.2f %.2f %.2f\n", refL, refA, refB)

	// Measured values
	measX, measY, measZ := vec.Vector3D(patch.MeasuredXYZ).XYZ()
	fmt.Printf("Measured XYZ: %.2f %.2f %.2f\n", measX, measY, measZ)
	measL, measA, measB := vec.Vector3D(patch.Measured).XYZ()
	fmt.Printf("Measured LAB: %.2f %.2f %.2f\n", measL, measA, measB)

	// RGB
	R, G, B := vec.Vector3D(patch.RGB).XYZ()
	R = mathx.Clamp(R, 0, 255)
	G = mathx.Clamp(G, 0, 255)
	B = mathx.Clamp(B, 0, 255)
	fmt.Printf("RGB: #%02X%02X%02X\n", uint8(R), uint8(G), uint8(B))
}

// exportPatchResults exports patch measurement results to a file
func exportPatchResults(ctx context.Context, results []render.Patch, path string) error {
	format := detectFormat(path)
	slog.Debug("Exporting patch results", "path", path, "format", format, "count", len(results))

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer file.Close()

	switch format {
	case "csv":
		return exportPatchResultsCSV(ctx, file, results)
	case "json":
		return exportPatchResultsJSON(ctx, file, results)
	case "pb", "proto":
		return exportPatchResultsProto(ctx, file, results)
	default:
		return fmt.Errorf("unsupported export format: %s (supported: csv, json, pb)", format)
	}
}

// exportPatchResultsCSV exports patch results to CSV format
func exportPatchResultsCSV(ctx context.Context, w *os.File, results []render.Patch) error {
	writer := csv.NewWriter(w)
	defer writer.Flush()

	// Write header
	header := []string{"name", "delta_e", "ref_xyz_x", "ref_xyz_y", "ref_xyz_z", "ref_lab_l", "ref_lab_a", "ref_lab_b", "meas_xyz_x", "meas_xyz_y", "meas_xyz_z", "meas_lab_l", "meas_lab_a", "meas_lab_b", "rgb_r", "rgb_g", "rgb_b"}
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("failed to write CSV header: %w", err)
	}

	// Write data rows
	for _, patch := range results {
		refX, refY, refZ := vec.Vector3D(patch.ReferenceXYZ).XYZ()
		refL, refA, refB := vec.Vector3D(patch.Reference).XYZ()
		measX, measY, measZ := vec.Vector3D(patch.MeasuredXYZ).XYZ()
		measL, measA, measB := vec.Vector3D(patch.Measured).XYZ()
		R, G, B := vec.Vector3D(patch.RGB).XYZ()
		R = mathx.Clamp(R, 0, 255)
		G = mathx.Clamp(G, 0, 255)
		B = mathx.Clamp(B, 0, 255)

		row := []string{
			patch.Name,
			fmt.Sprintf("%.4f", patch.DeltaE),
			fmt.Sprintf("%.6f", refX), fmt.Sprintf("%.6f", refY), fmt.Sprintf("%.6f", refZ),
			fmt.Sprintf("%.6f", refL), fmt.Sprintf("%.6f", refA), fmt.Sprintf("%.6f", refB),
			fmt.Sprintf("%.6f", measX), fmt.Sprintf("%.6f", measY), fmt.Sprintf("%.6f", measZ),
			fmt.Sprintf("%.6f", measL), fmt.Sprintf("%.6f", measA), fmt.Sprintf("%.6f", measB),
			fmt.Sprintf("%.0f", R), fmt.Sprintf("%.0f", G), fmt.Sprintf("%.0f", B),
		}
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("failed to write CSV row: %w", err)
		}
	}

	return nil
}

// exportPatchResultsJSON exports patch results to JSON format
func exportPatchResultsJSON(ctx context.Context, w *os.File, results []render.Patch) error {
	// Create a simple JSON structure for patch results
	type PatchResult struct {
		Name         string    `json:"name"`
		DeltaE       float32   `json:"delta_e"`
		ReferenceXYZ []float32 `json:"reference_xyz"`
		ReferenceLAB []float32 `json:"reference_lab"`
		MeasuredXYZ  []float32 `json:"measured_xyz"`
		MeasuredLAB  []float32 `json:"measured_lab"`
		RGB          []float32 `json:"rgb"`
	}

	patchResults := make([]PatchResult, len(results))
	for i, patch := range results {
		refX, refY, refZ := vec.Vector3D(patch.ReferenceXYZ).XYZ()
		refL, refA, refB := vec.Vector3D(patch.Reference).XYZ()
		measX, measY, measZ := vec.Vector3D(patch.MeasuredXYZ).XYZ()
		measL, measA, measB := vec.Vector3D(patch.Measured).XYZ()
		R, G, B := vec.Vector3D(patch.RGB).XYZ()
		R = mathx.Clamp(R, 0, 255)
		G = mathx.Clamp(G, 0, 255)
		B = mathx.Clamp(B, 0, 255)

		patchResults[i] = PatchResult{
			Name:         patch.Name,
			DeltaE:       patch.DeltaE,
			ReferenceXYZ: []float32{refX, refY, refZ},
			ReferenceLAB: []float32{refL, refA, refB},
			MeasuredXYZ:  []float32{measX, measY, measZ},
			MeasuredLAB:  []float32{measL, measA, measB},
			RGB:          []float32{R, G, B},
		}
	}

	marshaller := jsonMarshaller.NewMarshaller()
	if err := marshaller.Marshal(w, patchResults); err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	return nil
}

// exportPatchResultsProto exports patch results to protobuf format
func exportPatchResultsProto(ctx context.Context, w *os.File, results []render.Patch) error {
	// For proto export, we could create a new ColorPatches with measured values
	// For now, just export as JSON (proto export can be added later if needed)
	return exportPatchResultsJSON(ctx, w, results)
}
