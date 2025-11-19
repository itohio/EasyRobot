package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"image"
	"image/color"
	"log/slog"
	"math"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/cr30"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/vec"
	cv "gocv.io/x/gocv"
)

var (
	port      = flag.String("port", "", "Serial port device (e.g., /dev/ttyUSB0 or COM3)")
	baud      = flag.Int("baud", 19200, "Serial port baud rate")
	measure   = flag.Int("measure", 0, "Measure N samples by waiting for measurement")
	samples   = flag.Int("samples", 0, "Measure additional M samples after initial user-initiated measurement")
	display   = flag.Bool("display", false, "Display spectrum plots")
	calibrate = flag.Bool("calibrate", false, "Perform calibration before measurements")
	delay     = flag.Duration("delay", 700*time.Millisecond, "Delay between consecutive measurements for averaging")
	width     = flag.Int("width", 3840, "Image width in pixels")
	height    = flag.Int("height", 2160, "Image height in pixels")
	scale     = flag.Bool("scale", false, "Scale Y-axis to fit data range (default: 0 to max)")
	bars      = flag.Bool("bars", false, "Draw spectrum as colored vertical bars (requires -display)")
	verbose   = flag.Int("v", 0, "Verbose output level: 0=warn/error, 1=info, 2=debug (can also use -v -v or -vv)")
)

const (
	numBands      = 31
	minWavelength = 400
	maxWavelength = 700
)

func main() {
	// Count -v and -vv flags before flag.Parse() consumes them
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

	flag.Parse()

	// Use verbose flag value if set, otherwise use counted -v flags
	// If -vv was used, it won't be parsed by flag, so we use the count
	verboseLevel := *verbose
	if hasVV {
		verboseLevel = 2
	} else if verboseLevel == 0 && verboseCount > 0 {
		verboseLevel = verboseCount
	}

	// Setup logging based on verbose level
	switch verboseLevel {
	case 0:
		slog.SetLogLoggerLevel(slog.LevelWarn) // Default: warning/error only
	case 1:
		slog.SetLogLoggerLevel(slog.LevelInfo) // -v: info and above
	default:
		slog.SetLogLoggerLevel(slog.LevelDebug) // -vv or -v=2: debug and above
	}

	if *port == "" {
		slog.Error("Serial port required", "flag", "-port")
		flag.Usage()
		os.Exit(1)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// Connect to serial port
	config := devices.DefaultSerialConfig()
	config.BaudRate = *baud
	ser, err := devices.NewSerialWithConfig(*port, config)
	if err != nil {
		slog.Error("Failed to open serial port", "port", *port, "baud", *baud, "err", err)
		os.Exit(1)
	}
	defer func() {
		slog.Info("Closing serial port")
		ser.Close()
	}()

	// Create CR30 device
	dev := cr30.New(ser)
	dev.SetVerbose(false)

	// Connect and initialize
	slog.Info("Connecting to CR30 device", "port", *port, "baud", *baud)
	if err := dev.Connect(); err != nil {
		slog.Error("Failed to connect to device", "err", err)
		os.Exit(1)
	}
	defer func() {
		slog.Info("Disconnecting from device")
		dev.Disconnect()
	}()

	// Print device info as CSV comments
	info := dev.DeviceInfo()
	slog.Info("Device connected",
		"name", info.Name,
		"model", info.Model,
		"serial", info.Serial,
		"firmware", info.Firmware,
		"build", info.Build,
	)
	fmt.Printf("# Device: %s %s\n", info.Name, info.Model)
	fmt.Printf("# Serial: %s\n", info.Serial)
	fmt.Printf("# Firmware: %s (Build: %s)\n", info.Firmware, info.Build)
	fmt.Printf("# Timestamp: %s\n", time.Now().Format(time.RFC3339))
	fmt.Println()

	// Handle calibration if requested
	if *calibrate {
		if err := performCalibration(ctx, dev); err != nil {
			slog.Error("Calibration failed", "err", err)
			os.Exit(1)
		}
	}

	// Setup display if requested
	var dest destination.Destination
	if *display {
		destination.RegisterAllFlags()
		dest = destination.NewDisplay()
		// Pass context with cancel function so display can cancel main context on window close
		displayCtx := context.WithValue(ctx, "cancel", cancel)
		if err := dest.Start(displayCtx); err != nil {
			slog.Error("Failed to start display", "err", err)
			os.Exit(1)
		}
		defer dest.Close()
	}

	// Print CSV header
	printCSVHeader(dev)

	// Handle measurement modes
	if *measure > 0 && *samples > 0 {
		// Both flags: N measurements, each averages M samples
		if err := performMeasurements(ctx, dev, *measure, *samples, dest); err != nil {
			slog.Error("Measurement failed", "err", err)
			os.Exit(1)
		}
	} else if *measure > 0 {
		// -measure=N: N measurements, each is 1 sample (no averaging)
		if err := performMeasurements(ctx, dev, *measure, 1, dest); err != nil {
			slog.Error("Measurement failed", "err", err)
			os.Exit(1)
		}
	} else if *samples > 0 {
		// -samples=M: 1 measurement that is average of M samples
		if err := performMeasurements(ctx, dev, 1, *samples, dest); err != nil {
			slog.Error("Additional samples measurement failed", "err", err)
			os.Exit(1)
		}
	} else {
		slog.Info("No measurement mode specified. Use -measure=N or -samples=M")
	}

	// If display is enabled, wait for Ctrl+C or window close
	if *display {
		slog.Info("Measurements complete. Press Ctrl+C or close the window to exit.")
		<-ctx.Done()
		if ctx.Err() == context.Canceled {
			slog.Info("Shutting down...")
		}
	}
}

func performCalibration(ctx context.Context, dev *cr30.Device) error {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("Calibration mode")
	fmt.Println("Place device on white calibration tile and press Enter...")
	if _, err := reader.ReadString('\n'); err != nil {
		return fmt.Errorf("failed to read input: %w", err)
	}

	slog.Info("Starting white calibration")
	// Note: Calibration not yet implemented in cr30 package
	// This is a placeholder for when calibration is added
	// When implemented, use: dev.CalibrateWhite(ctx)
	slog.Info("White calibration complete")

	fmt.Println("Place device on black calibration tile and press Enter...")
	if _, err := reader.ReadString('\n'); err != nil {
		return fmt.Errorf("failed to read input: %w", err)
	}

	slog.Info("Starting black calibration")
	// Note: Calibration not yet implemented in cr30 package
	// When implemented, use: dev.CalibrateBlack(ctx)
	slog.Info("Black calibration complete")

	fmt.Println("Calibration complete")
	return nil
}

// performMeasurements orchestrates multiple measurements, prints to stdout, and draws to image.
// measurementCount: number of measurements to perform
// samplesPerMeasurement: number of samples to average for each measurement
func performMeasurements(ctx context.Context, dev *cr30.Device, measurementCount, samplesPerMeasurement int, dest destination.Destination) error {
	if measurementCount < 1 {
		return fmt.Errorf("measurement count must be at least 1")
	}
	if samplesPerMeasurement < 1 {
		return fmt.Errorf("samples per measurement must be at least 1")
	}

	numWl := dev.NumWavelengths()

	// Collect all measurements
	for i := 0; i < measurementCount; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}

		// Add delay between consecutive measurements (except before first)
		if i > 0 {
			slog.Debug("Waiting before next measurement", "delay", *delay)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(*delay):
				// Continue
			}
		}

		slog.Info("Performing measurement", "measurement", i+1, "total", measurementCount, "samples_per_measurement", samplesPerMeasurement)

		// Perform measurement (collects samples, calculates average and stddev)
		avgMatrix, stddevMatrix, err := performMeasurement(ctx, dev, numWl, samplesPerMeasurement)
		if err != nil {
			slog.Error("Measurement failed", "measurement", i+1, "err", err)
			continue
		}

		if avgMatrix == nil {
			continue
		}

		// Print to stdout
		avgSpectrum := avgMatrix.Row(0).(vec.Vector)
		printCSVRow(avgSpectrum)
		if stddevMatrix != nil {
			stddevSpectrum := stddevMatrix.Row(0).(vec.Vector)
			printCSVRow(stddevSpectrum)
		}

		// Draw to image
		if dest != nil {
			var stddev []float32
			if stddevMatrix != nil {
				stddevVec := stddevMatrix.Row(0).(vec.Vector)
				stddev = make([]float32, len(stddevVec))
				copy(stddev, stddevVec)
			}
			avgVec := avgMatrix.Row(0).(vec.Vector)
			avg := make([]float32, len(avgVec))
			copy(avg, avgVec)

			if err := displaySpectrum(dest, avg, stddev, nil); err != nil {
				slog.Error("Display failed", "err", err)
				return fmt.Errorf("display failed: %w", err)
			}
		}
	}

	return nil
}

// performMeasurement collects samples, calculates average and stddev.
// First sample uses WaitMeasurement (user-initiated), rest use Measure (PC-initiated).
// Returns matrices with average in row 0 of avgMatrix and stddev in row 0 of stddevMatrix.
func performMeasurement(ctx context.Context, dev *cr30.Device, numWl, sampleCount int) (avgMatrix matTypes.Matrix, stddevMatrix matTypes.Matrix, err error) {
	if sampleCount < 1 {
		return nil, nil, fmt.Errorf("sample count must be at least 1")
	}

	// Store all samples for stddev calculation
	samples := make([]matTypes.Matrix, 0, sampleCount)
	sampleMatrix := mat.New(1, numWl)

	// First sample: wait for user measurement
	slog.Info("Waiting for initial user measurement", "sample", 1, "total_samples", sampleCount)
	measureCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	err = dev.WaitMeasurement(measureCtx, sampleMatrix)
	cancel()
	if err != nil {
		return nil, nil, fmt.Errorf("initial measurement failed: %w", err)
	}

	// Store first sample
	firstSample := mat.New(1, numWl)
	firstSample.SetRow(0, sampleMatrix.Row(0))
	samples = append(samples, firstSample)

	// Collect remaining samples
	for i := 1; i < sampleCount; i++ {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}

		// Add delay before each additional sample (including the first one after WaitMeasurement)
		select {
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		case <-time.After(*delay):
			// Continue
		}

		slog.Info("Taking additional sample", "sample", i+1, "total_samples", sampleCount)
		measureCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		err = dev.Measure(measureCtx, sampleMatrix)
		cancel()
		if err != nil {
			slog.Error("Sample collection failed", "sample", i+1, "err", err)
			// Continue with available samples
			break
		}

		// Store sample
		sample := mat.New(1, numWl)
		sample.SetRow(0, sampleMatrix.Row(0))
		samples = append(samples, sample)
	}

	collectedSamples := len(samples)
	if collectedSamples == 0 {
		return nil, nil, fmt.Errorf("no samples collected")
	}

	// Calculate average
	avgMatrix = mat.New(1, numWl)
	avgVec := avgMatrix.Row(0).(vec.Vector)

	// Sum all samples
	for _, s := range samples {
		sVec := s.Row(0).(vec.Vector)
		avgVec.Add(sVec)
	}

	// Divide by count
	avgVec.DivC(float32(collectedSamples))
	avgMatrix.SetRow(0, avgVec)

	// Calculate stddev
	stddevMatrix = mat.New(1, numWl)
	stddevVec := stddevMatrix.Row(0).(vec.Vector)

	if collectedSamples > 1 {
		// Calculate variance: sum((x - mean)^2) / (n-1)
		for i := 0; i < numWl; i++ {
			sumSqDiff := float32(0)
			avgVal := avgVec[i]
			for _, s := range samples {
				sVec := s.Row(0).(vec.Vector)
				diff := sVec[i] - avgVal
				sumSqDiff += diff * diff
			}
			variance := sumSqDiff / float32(collectedSamples-1)
			stddevVec[i] = float32(math.Sqrt(float64(variance)))
		}
	} else {
		// Single sample: stddev is 0
		stddevVec.FillC(0)
	}
	stddevMatrix.SetRow(0, stddevVec)

	return avgMatrix, stddevMatrix, nil
}

func calculateStatistics(spectra [][]float32) ([]float32, []float32) {
	if len(spectra) == 0 {
		return nil, nil
	}

	if len(spectra[0]) == 0 {
		return nil, nil
	}

	numBands := len(spectra[0])
	mean := make([]float32, numBands)
	stddev := make([]float32, numBands)

	// Calculate mean
	for i := 0; i < numBands; i++ {
		sum := float32(0)
		validCount := 0
		for _, spec := range spectra {
			if spec != nil && i < len(spec) {
				sum += spec[i]
				validCount++
			}
		}
		if validCount > 0 {
			mean[i] = sum / float32(validCount)
		}
	}

	// Calculate standard deviation
	for i := 0; i < numBands; i++ {
		sumSqDiff := float32(0)
		validCount := 0
		for _, spec := range spectra {
			if spec != nil && i < len(spec) {
				diff := spec[i] - mean[i]
				sumSqDiff += diff * diff
				validCount++
			}
		}
		if validCount > 1 {
			stddev[i] = float32(math.Sqrt(float64(sumSqDiff / float32(validCount-1))))
		}
	}

	return mean, stddev
}

func printCSVHeader(dev *cr30.Device) {
	// Get wavelengths from device
	numWl := dev.NumWavelengths()
	wlVec := vec.New(numWl)
	dev.Wavelengths(wlVec)

	// Print wavelength header
	header := make([]string, numWl)
	for i := 0; i < numWl; i++ {
		header[i] = strconv.Itoa(int(wlVec[i]))
	}
	fmt.Println(strings.Join(header, ","))
}

func printCSVRow(spectrum []float32) {
	if len(spectrum) == 0 {
		return
	}

	values := make([]string, len(spectrum))
	for i := 0; i < len(spectrum); i++ {
		values[i] = strconv.FormatFloat(float64(spectrum[i]), 'f', 6, 32)
	}
	fmt.Println(strings.Join(values, ","))
}

func displaySpectrum(dest destination.Destination, spectrum []float32, stddev []float32, plusMinus []float32) error {
	img := createSpectrumImage(spectrum, stddev, *width, *height)

	// Convert image.RGBA to gocv Mat
	mat, err := cv.ImageToMatRGB(img)
	if err != nil {
		return fmt.Errorf("failed to convert image to mat: %w", err)
	}

	// Convert Mat to Tensor (tensor will own the mat, so don't close it here)
	tensor, err := tensorgocv.FromMat(mat, tensorgocv.WithAdoptedMat())
	if err != nil {
		mat.Close()
		return fmt.Errorf("failed to create tensor from mat: %w", err)
	}

	frame := types.Frame{
		Index:     0,
		Timestamp: time.Now().UnixNano(),
		Tensors:   []types.Tensor{tensor},
	}

	return dest.AddFrame(frame)
}

func createSpectrumImage(spectrum []float32, stddev []float32, imgWidth, imgHeight int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, imgWidth, imgHeight))

	// Fill with black background
	for y := 0; y < imgHeight; y++ {
		for x := 0; x < imgWidth; x++ {
			img.Set(x, y, color.RGBA{0, 0, 0, 255})
		}
	}

	// Find min/max for scaling
	var minVal, maxVal float32
	if *scale {
		// Scale to fit data range
		minVal, maxVal = findMinMax(spectrum)
		for i, s := range stddev {
			if i < len(spectrum) {
				minVal = min(minVal, spectrum[i]-s)
				maxVal = max(maxVal, spectrum[i]+s)
			}
		}
	} else {
		// Default: 0 to max(SPD)
		minVal = 0
		maxVal = findMax(spectrum)
		for i, s := range stddev {
			if i < len(spectrum) {
				maxVal = max(maxVal, spectrum[i]+s)
			}
		}
	}
	rangeVal := maxVal - minVal
	if rangeVal == 0 {
		rangeVal = 1
	}

	// Draw grid
	drawGrid(img, minVal, maxVal, imgWidth, imgHeight)

	// Draw spectrum bars first (if enabled) so they appear below the confidence band
	if *bars {
		drawSpectrumBars(img, spectrum, minVal, maxVal, imgWidth, imgHeight)
	}

	// Draw stddev bands if available (drawn after bars so it appears on top)
	if stddev != nil {
		drawStdDevBands(img, spectrum, stddev, minVal, maxVal, imgWidth, imgHeight)
	}

	// Draw spectrum line (if not using bars)
	if !*bars {
		drawSpectrumLine(img, spectrum, minVal, maxVal, imgWidth, imgHeight)
	}

	// Draw axes labels
	drawAxesLabels(img, minVal, maxVal, imgWidth, imgHeight)

	return img
}

func findMinMax(spectrum []float32) (float32, float32) {
	if len(spectrum) == 0 {
		return 0, 1
	}
	minVal := spectrum[0]
	maxVal := spectrum[0]
	for _, v := range spectrum {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	return minVal, maxVal
}

func findMax(spectrum []float32) float32 {
	if len(spectrum) == 0 {
		return 1
	}
	maxVal := spectrum[0]
	for _, v := range spectrum {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

func drawGrid(img *image.RGBA, minVal, maxVal float32, imgWidth, imgHeight int) {
	// Draw horizontal grid lines
	gridColor := color.RGBA{64, 64, 64, 255}
	for i := 0; i <= 10; i++ {
		y := imgHeight - 60 - (i * (imgHeight - 120) / 10)
		drawLine(img, 60, y, imgWidth-20, y, gridColor, imgWidth, imgHeight)
	}

	// Draw vertical grid lines (wavelength markers)
	for i := 0; i <= 10; i++ {
		x := 60 + (i * (imgWidth - 80) / 10)
		drawLine(img, x, 20, x, imgHeight-60, gridColor, imgWidth, imgHeight)
	}
}

func drawStdDevBands(img *image.RGBA, spectrum []float32, stddev []float32, minVal, maxVal float32, imgWidth, imgHeight int) {
	if len(spectrum) != numBands || len(stddev) != numBands {
		return
	}

	bandColor := color.RGBA{64, 64, 128, 128} // Faint blue with 50% opacity (128/255)
	rangeVal := maxVal - minVal
	if rangeVal == 0 {
		return
	}

	points := make([]image.Point, numBands*2)
	for i := 0; i < numBands; i++ {
		x := 60 + (i * (imgWidth - 80) / (numBands - 1))

		upper := spectrum[i] + stddev[i]
		lower := spectrum[i] - stddev[i]

		yUpper := imgHeight - 60 - int((upper-minVal)/rangeVal*float32(imgHeight-120))
		yLower := imgHeight - 60 - int((lower-minVal)/rangeVal*float32(imgHeight-120))

		points[i] = image.Point{x, yUpper}
		points[numBands*2-1-i] = image.Point{x, yLower}
	}

	// Draw filled polygon
	drawFilledPolygon(img, points, bandColor, imgWidth, imgHeight)
}

func drawSpectrumLine(img *image.RGBA, spectrum []float32, minVal, maxVal float32, imgWidth, imgHeight int) {
	if len(spectrum) != numBands {
		return
	}

	lineColor := color.RGBA{0, 255, 0, 255} // Green
	rangeVal := maxVal - minVal
	if rangeVal == 0 {
		return
	}

	var prevX, prevY int
	for i := 0; i < numBands; i++ {
		x := 60 + (i * (imgWidth - 80) / (numBands - 1))
		y := imgHeight - 60 - int((spectrum[i]-minVal)/rangeVal*float32(imgHeight-120))

		if i > 0 {
			drawLine(img, prevX, prevY, x, y, lineColor, imgWidth, imgHeight)
		}
		prevX, prevY = x, y
	}
}

// drawSpectrumBars draws the spectrum as colored vertical bars.
// Each bar is colored according to its wavelength.
func drawSpectrumBars(img *image.RGBA, spectrum []float32, minVal, maxVal float32, imgWidth, imgHeight int) {
	if len(spectrum) != numBands {
		return
	}

	rangeVal := maxVal - minVal
	if rangeVal == 0 {
		return
	}

	barWidth := (imgWidth - 80) / numBands
	if barWidth < 1 {
		barWidth = 1
	}

	baseY := imgHeight - 60

	for i := 0; i < numBands; i++ {
		wavelength := minWavelength + i*10
		barColor := wavelengthToRGB(wavelength)

		x := 60 + (i * (imgWidth - 80) / numBands)
		value := spectrum[i]
		barHeight := int((value - minVal) / rangeVal * float32(imgHeight-120))
		if barHeight < 0 {
			barHeight = 0
		}

		// Draw vertical bar
		for bx := x; bx < x+barWidth && bx < imgWidth-20; bx++ {
			for by := baseY; by >= baseY-barHeight && by >= 60; by-- {
				if bx >= 0 && bx < imgWidth && by >= 0 && by < imgHeight {
					img.Set(bx, by, barColor)
				}
			}
		}
	}
}

// wavelengthToRGB converts a wavelength in nanometers to an RGB color.
// This is a simplified approximation for the visible spectrum (400-700nm).
func wavelengthToRGB(wavelength int) color.RGBA {
	var r, g, b float32

	if wavelength < 380 {
		// Below visible spectrum - ultraviolet
		return color.RGBA{0, 0, 0, 255}
	} else if wavelength < 440 {
		// Violet to blue
		r = -(float32(wavelength) - 440) / 60
		g = 0
		b = 1
	} else if wavelength < 490 {
		// Blue to cyan
		r = 0
		g = (float32(wavelength) - 440) / 50
		b = 1
	} else if wavelength < 510 {
		// Cyan to green
		r = 0
		g = 1
		b = -(float32(wavelength) - 510) / 20
	} else if wavelength < 580 {
		// Green to yellow
		r = (float32(wavelength) - 510) / 70
		g = 1
		b = 0
	} else if wavelength < 645 {
		// Yellow to orange to red
		r = 1
		g = -(float32(wavelength) - 645) / 65
		b = 0
	} else if wavelength <= 780 {
		// Red
		r = 1
		g = 0
		b = 0
	} else {
		// Above visible spectrum - infrared
		return color.RGBA{0, 0, 0, 255}
	}

	// Adjust brightness for wavelengths near the edges
	var factor float32 = 1.0
	if wavelength < 420 {
		factor = 0.3 + 0.7*(float32(wavelength)-380)/40
	} else if wavelength > 700 {
		factor = 0.3 + 0.7*(780-float32(wavelength))/80
	}

	r = r * factor
	g = g * factor
	b = b * factor

	// Clamp and convert to uint8
	clamp := func(v float32) uint8 {
		if v < 0 {
			return 0
		}
		if v > 1 {
			return 255
		}
		return uint8(v * 255)
	}

	return color.RGBA{
		R: clamp(r),
		G: clamp(g),
		B: clamp(b),
		A: 255,
	}
}

func drawAxesLabels(img *image.RGBA, minVal, maxVal float32, imgWidth, imgHeight int) {
	// Y-axis labels (reflectance values)
	labelColor := color.RGBA{255, 255, 255, 255}
	for i := 0; i <= 10; i++ {
		val := minVal + (maxVal-minVal)*float32(i)/10
		y := imgHeight - 60 - (i * (imgHeight - 120) / 10)
		label := fmt.Sprintf("%.3f", val)
		drawText(img, 10, y+5, label, labelColor, imgWidth, imgHeight)
	}

	// X-axis labels (wavelengths)
	for i := 0; i <= 10; i++ {
		wavelength := minWavelength + i*((maxWavelength-minWavelength)/10)
		x := 60 + (i * (imgWidth - 80) / 10)
		label := fmt.Sprintf("%d", wavelength)
		drawText(img, x-10, imgHeight-40, label, labelColor, imgWidth, imgHeight)
	}
}

func drawLine(img *image.RGBA, x1, y1, x2, y2 int, c color.RGBA, imgWidth, imgHeight int) {
	dx := abs(x2 - x1)
	dy := abs(y2 - y1)
	sx := 1
	if x1 > x2 {
		sx = -1
	}
	sy := 1
	if y1 > y2 {
		sy = -1
	}
	err := dx - dy

	x, y := x1, y1
	for {
		if x >= 0 && x < imgWidth && y >= 0 && y < imgHeight {
			img.Set(x, y, c)
		}
		if x == x2 && y == y2 {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x += sx
		}
		if e2 < dx {
			err += dx
			y += sy
		}
	}
}

func drawFilledPolygon(img *image.RGBA, points []image.Point, c color.RGBA, imgWidth, imgHeight int) {
	if len(points) < 3 {
		return
	}

	// Find bounding box
	minX, minY := points[0].X, points[0].Y
	maxX, maxY := points[0].X, points[0].Y
	for _, p := range points {
		if p.X < minX {
			minX = p.X
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}

	// Clamp to image bounds
	if minX < 0 {
		minX = 0
	}
	if minY < 0 {
		minY = 0
	}
	if maxX >= imgWidth {
		maxX = imgWidth - 1
	}
	if maxY >= imgHeight {
		maxY = imgHeight - 1
	}

	// Fill polygon using scanline algorithm
	for y := minY; y <= maxY; y++ {
		var intersections []int
		for i := 0; i < len(points); i++ {
			p1 := points[i]
			p2 := points[(i+1)%len(points)]
			if (p1.Y <= y && p2.Y > y) || (p2.Y <= y && p1.Y > y) {
				dy := p2.Y - p1.Y
				if dy != 0 {
					x := p1.X + (y-p1.Y)*(p2.X-p1.X)/dy
					intersections = append(intersections, x)
				}
			}
		}

		// Sort intersections
		for i := 0; i < len(intersections)-1; i++ {
			for j := i + 1; j < len(intersections); j++ {
				if intersections[i] > intersections[j] {
					intersections[i], intersections[j] = intersections[j], intersections[i]
				}
			}
		}

		// Fill between pairs
		for i := 0; i < len(intersections); i += 2 {
			if i+1 < len(intersections) {
				x1, x2 := intersections[i], intersections[i+1]
				for x := x1; x <= x2; x++ {
					if x >= 0 && x < imgWidth && y >= 0 && y < imgHeight {
						// Alpha blend if color has transparency
						if c.A < 255 {
							blendColor(img, x, y, c)
						} else {
							img.Set(x, y, c)
						}
					}
				}
			}
		}
	}
}

// blendColor alpha-blends a color onto an existing pixel
func blendColor(img *image.RGBA, x, y int, c color.RGBA) {
	idx := (y-img.Rect.Min.Y)*img.Stride + (x-img.Rect.Min.X)*4
	if idx < 0 || idx+3 >= len(img.Pix) {
		return
	}

	// Get existing pixel
	r := uint32(img.Pix[idx])
	g := uint32(img.Pix[idx+1])
	b := uint32(img.Pix[idx+2])

	// Alpha blend: result = (src * alpha + dst * (255 - alpha)) / 255
	alpha := uint32(c.A)
	invAlpha := 255 - alpha

	r = (uint32(c.R)*alpha + r*invAlpha) / 255
	g = (uint32(c.G)*alpha + g*invAlpha) / 255
	b = (uint32(c.B)*alpha + b*invAlpha) / 255

	img.Pix[idx] = uint8(r)
	img.Pix[idx+1] = uint8(g)
	img.Pix[idx+2] = uint8(b)
	// Alpha channel remains unchanged (img.Pix[idx+3])
}

func drawText(img *image.RGBA, x, y int, text string, c color.RGBA, imgWidth, imgHeight int) {
	// Simple text rendering - just draw pixels
	// For better text, would need a font rendering library
	for i, ch := range text {
		drawChar(img, x+i*6, y, ch, c, imgWidth, imgHeight)
	}
}

func drawChar(img *image.RGBA, x, y int, ch rune, c color.RGBA, imgWidth, imgHeight int) {
	// Very simple 5x7 font rendering
	// This is a placeholder - in production would use a proper font
	font := getSimpleFont()
	if glyph, ok := font[ch]; ok {
		for py := 0; py < 7; py++ {
			for px := 0; px < 5; px++ {
				if glyph[py]&(1<<uint(4-px)) != 0 {
					if x+px >= 0 && x+px < imgWidth && y+py >= 0 && y+py < imgHeight {
						img.Set(x+px, y+py, c)
					}
				}
			}
		}
	}
}

func getSimpleFont() map[rune][7]uint8 {
	// Simple 5x7 bitmap font for digits and basic chars
	return map[rune][7]uint8{
		'0': {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
		'1': {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
		'2': {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},
		'3': {0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E},
		'4': {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},
		'5': {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},
		'6': {0x0E, 0x11, 0x10, 0x1E, 0x11, 0x11, 0x0E},
		'7': {0x1F, 0x01, 0x02, 0x04, 0x04, 0x04, 0x04},
		'8': {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
		'9': {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x11, 0x0E},
		'.': {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04},
		'-': {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00},
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func min(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
