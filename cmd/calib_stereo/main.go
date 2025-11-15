package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/itohio/EasyRobot/cmd/calib_mono/corners"
	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/itohio/EasyRobot/x/marshaller/gocv"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	cv "gocv.io/x/gocv"
)

func main() {
	// Register destination flags
	destination.RegisterAllFlags()

	// Calibration-specific flags
	gridStr := flag.String("grid", "9,7", "Calibration grid shape (width,height)")
	samples := flag.Int("samples", 30, "Number of calibration samples to collect")
	output := flag.String("output", "stereo_camera.json", "Output file path")
	format := flag.String("format", "json", "Output format: json, yaml, gocv")
	testMode := flag.Bool("test", false, "Test mode - load existing calibration file")
	disparity := flag.Bool("disparity", false, "Calculate and display disparity map (test mode only)")
	help := flag.Bool("help", false, "Show help message")

	// Source flags for stereo (left and right)
	leftCamera := flag.String("left-camera", "", "Left camera device ID")
	rightCamera := flag.String("right-camera", "", "Right camera device ID")
	leftVideo := flag.String("left-video", "", "Left video file path")
	rightVideo := flag.String("right-video", "", "Right video file path")
	leftImages := flag.String("left-images", "", "Left image file or directory path")
	rightImages := flag.String("right-images", "", "Right image file or directory path")
	width := flag.Int("width", 640, "Frame width for cameras")
	height := flag.Int("height", 480, "Frame height for cameras")

	flag.Parse()

	if *help {
		flag.PrintDefaults()
		return
	}

	// Parse grid size
	gridWidth, gridHeight, err := parseGrid(*gridStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing grid: %v\n", err)
		os.Exit(1)
	}

	// Create left and right sources
	leftSrc, err := createSource(*leftCamera, *leftVideo, *leftImages, *width, *height, "left")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating left source: %v\n", err)
		flag.PrintDefaults()
		os.Exit(1)
	}

	rightSrc, err := createSource(*rightCamera, *rightVideo, *rightImages, *width, *height, "right")
	if err != nil {
		leftSrc.Close()
		fmt.Fprintf(os.Stderr, "Error creating right source: %v\n", err)
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Create destinations from flags
	dests := destination.NewAllDestinations()

	// Setup context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		cancel()
	}()

	// Start sources
	if err := leftSrc.Start(ctx); err != nil {
		leftSrc.Close()
		rightSrc.Close()
		fmt.Fprintf(os.Stderr, "Error starting left source: %v\n", err)
		os.Exit(1)
	}
	defer leftSrc.Close()

	if err := rightSrc.Start(ctx); err != nil {
		rightSrc.Close()
		fmt.Fprintf(os.Stderr, "Error starting right source: %v\n", err)
		os.Exit(1)
	}
	defer rightSrc.Close()

	// Start destinations
	for _, dest := range dests {
		if err := dest.Start(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "Error starting destination: %v\n", err)
			os.Exit(1)
		}
		defer dest.Close()
	}

	// Process frames
	if *testMode {
		if err := testCalibration(ctx, leftSrc, rightSrc, dests, *output, *disparity); err != nil {
			fmt.Fprintf(os.Stderr, "Error testing calibration: %v\n", err)
			os.Exit(1)
		}
	} else {
		processor := NewStereoCalibrationProcessor(image.Point{X: gridWidth, Y: gridHeight}, *samples)
		defer processor.Close()
		if err := calibrateStereo(ctx, leftSrc, rightSrc, processor, dests, *output, *format); err != nil {
			fmt.Fprintf(os.Stderr, "Error calibrating stereo cameras: %v\n", err)
			os.Exit(1)
		}
	}
}

func parseGrid(gridStr string) (width, height int, err error) {
	parts := strings.Split(gridStr, ",")
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("invalid grid format: %s (expected width,height)", gridStr)
	}
	width, err = strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil {
		return 0, 0, fmt.Errorf("invalid width: %w", err)
	}
	height, err = strconv.Atoi(strings.TrimSpace(parts[1]))
	if err != nil {
		return 0, 0, fmt.Errorf("invalid height: %w", err)
	}
	return width, height, nil
}

func createSource(cameraID, videoPath, imagesPath string, width, height int, side string) (source.Source, error) {
	count := 0

	if cameraID != "" {
		count++
		deviceID, err := strconv.Atoi(cameraID)
		if err != nil {
			return nil, fmt.Errorf("invalid %s camera device ID '%s': %w", side, cameraID, err)
		}
		return createCameraSource(deviceID, width, height), nil
	}

	if videoPath != "" {
		count++
		return createVideoSource(videoPath), nil
	}

	if imagesPath != "" {
		count++
		return createImageSource(imagesPath), nil
	}

	if count == 0 {
		return nil, fmt.Errorf("no %s source specified (use --left-camera/--right-camera, --left-video/--right-video, or --left-images/--right-images)", side)
	}

	return nil, fmt.Errorf("multiple %s sources specified (only one allowed)", side)
}

// Custom source implementations that wrap the baseSource
type customCameraSource struct {
	baseSource
	deviceID int
	width    int
	height   int
}

func createCameraSource(deviceID, width, height int) source.Source {
	return &customCameraSource{deviceID: deviceID, width: width, height: height}
}

func (s *customCameraSource) RegisterFlags() {}

func (s *customCameraSource) Start(ctx context.Context) error {
	opts := []types.Option{
		types.WithContext(ctx),
		gocv.WithVideoDevice(s.deviceID, s.width, s.height),
	}

	unmarshaller := gocv.NewUnmarshaller(opts...)
	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(nil, &stream); err != nil {
		return fmt.Errorf("failed to create camera source: %w", err)
	}

	s.stream = stream
	return s.baseSource.Start(ctx)
}

type customVideoSource struct {
	baseSource
	path string
}

func createVideoSource(path string) source.Source {
	return &customVideoSource{path: path}
}

func (s *customVideoSource) RegisterFlags() {}

func (s *customVideoSource) Start(ctx context.Context) error {
	opts := []types.Option{
		types.WithContext(ctx),
		gocv.WithPath(s.path),
	}

	unmarshaller := gocv.NewUnmarshaller(opts...)
	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(nil, &stream); err != nil {
		return fmt.Errorf("failed to create video source: %w", err)
	}

	s.stream = stream
	return s.baseSource.Start(ctx)
}

type customImageSource struct {
	baseSource
	path string
}

func createImageSource(path string) source.Source {
	return &customImageSource{path: path}
}

func (s *customImageSource) RegisterFlags() {}

func (s *customImageSource) Start(ctx context.Context) error {
	opts := []types.Option{
		types.WithContext(ctx),
		gocv.WithPath(s.path),
	}

	unmarshaller := gocv.NewUnmarshaller(opts...)
	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(nil, &stream); err != nil {
		return fmt.Errorf("failed to create image source: %w", err)
	}

	s.stream = stream
	return s.baseSource.Start(ctx)
}

// baseSource is a minimal implementation of the source interface
type baseSource struct {
	ctx     context.Context
	stream  types.FrameStream
	started bool
	frameCh <-chan types.Frame
	lastErr error
}

func (s *baseSource) Start(ctx context.Context) error {
	if s.started {
		return fmt.Errorf("source already started")
	}
	s.ctx = ctx
	s.started = true
	s.frameCh = s.stream.C
	return nil
}

func (s *baseSource) ReadFrame() (types.Frame, error) {
	if !s.started {
		return types.Frame{}, source.ErrSourceNotStarted
	}

	select {
	case <-s.ctx.Done():
		return types.Frame{}, s.ctx.Err()
	case frame, ok := <-s.frameCh:
		if !ok {
			return types.Frame{}, source.ErrSourceExhausted
		}
		if err, hasErr := frame.Metadata["error"]; hasErr {
			if errVal, ok := err.(error); ok {
				s.lastErr = errVal
				return types.Frame{}, errVal
			}
		}
		return frame, nil
	}
}

func (s *baseSource) Close() error {
	s.stream.Close()
	s.started = false
	return nil
}

// calibrateStereo performs stereo camera calibration
func calibrateStereo(ctx context.Context, leftSrc, rightSrc source.Source, processor *StereoCalibrationProcessor, dests []destination.Destination, outputPath, format string) error {
	fmt.Printf("Starting stereo camera calibration...\n")
	fmt.Printf("Grid size: %dx%d\n", processor.gridSize.X, processor.gridSize.Y)
	fmt.Printf("Target samples: %d\n", processor.targetSamples)

	leftDetector := corners.NewDetector(processor.gridSize)
	rightDetector := corners.NewDetector(processor.gridSize)

	for {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Read frames from both sources
		leftFrame, err := leftSrc.ReadFrame()
		if err != nil {
			if err == source.ErrSourceExhausted {
				break
			}
			continue
		}

		rightFrame, err := rightSrc.ReadFrame()
		if err != nil {
			if err == source.ErrSourceExhausted {
				break
			}
			continue
		}

		if len(leftFrame.Tensors) == 0 || len(rightFrame.Tensors) == 0 {
			continue
		}

		// Convert tensors to Mat
		leftMat, err := tensorToMat(leftFrame.Tensors[0])
		if err != nil {
			continue
		}

		rightMat, err := tensorToMat(rightFrame.Tensors[0])
		if err != nil {
			leftMat.Close()
			continue
		}

		// Process stereo frame for calibration
		found, err := processor.ProcessStereoFrame(leftMat, rightMat, leftDetector, rightDetector)
		if err != nil {
			leftMat.Close()
			rightMat.Close()
			continue
		}

		// Draw calibration visualization
		leftVis := leftMat.Clone()
		rightVis := rightMat.Clone()
		processor.DrawCorners(leftVis, rightVis, found)

		// Combine frames side-by-side for display
		combinedMat := combineFramesSideBySide(leftVis, rightVis)
		combinedTensor, err := matToTensor(combinedMat)
		if err != nil {
			leftMat.Close()
			rightMat.Close()
			leftVis.Close()
			rightVis.Close()
			combinedMat.Close()
			continue
		}

		// Create display frame
		displayFrame := types.Frame{
			Tensors:  []types.Tensor{combinedTensor},
			Metadata: leftFrame.Metadata,
		}

		// Send to all destinations
		for _, dest := range dests {
			if err := dest.AddFrame(displayFrame); err != nil {
				// Ignore destination errors
			}
		}

		leftMat.Close()
		rightMat.Close()
		leftVis.Close()
		rightVis.Close()
		combinedMat.Close()

		// Check if we have enough samples
		if processor.numSamples >= processor.targetSamples {
			fmt.Printf("\nCollected %d samples, computing calibration...\n", processor.numSamples)
			break
		}

		if processor.numSamples%5 == 0 {
			fmt.Printf("Collected %d/%d samples\n", processor.numSamples, processor.targetSamples)
		}
	}

	// Compute calibration
	calibration, err := processor.Calibrate()
	if err != nil {
		return fmt.Errorf("calibration failed: %w", err)
	}
	defer calibration.Close()

	// Save calibration
	if err := saveStereoCalibration(calibration, outputPath, format); err != nil {
		return fmt.Errorf("failed to save calibration: %w", err)
	}

	fmt.Printf("Calibration saved to %s\n", outputPath)
	fmt.Printf("Reprojection error: %.4f\n", calibration.ReprojectionError)

	return nil
}

// testCalibration tests stereo calibration by loading and applying it
func testCalibration(ctx context.Context, leftSrc, rightSrc source.Source, dests []destination.Destination, calibPath string, disparity bool) error {
	// Load calibration file
	calibration, err := loadStereoCalibration(calibPath)
	if err != nil {
		return fmt.Errorf("failed to load calibration: %w", err)
	}
	defer calibration.Close()

	fmt.Printf("Loaded calibration from %s\n", calibPath)
	fmt.Printf("Grid size: %dx%d\n", calibration.GridShape.X, calibration.GridShape.Y)
	fmt.Printf("Image size: %dx%d\n", calibration.ImageSize.X, calibration.ImageSize.Y)
	fmt.Printf("Samples used: %d\n", calibration.NumSamples)
	fmt.Printf("Reprojection error: %.4f\n", calibration.ReprojectionError)

	// Process frames and rectify
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Read frames from both sources
		leftFrame, err := leftSrc.ReadFrame()
		if err != nil {
			if err == source.ErrSourceExhausted {
				return nil
			}
			continue
		}

		rightFrame, err := rightSrc.ReadFrame()
		if err != nil {
			if err == source.ErrSourceExhausted {
				return nil
			}
			continue
		}

		if len(leftFrame.Tensors) == 0 || len(rightFrame.Tensors) == 0 {
			continue
		}

		// Convert tensors to Mat
		leftMat, err := tensorToMat(leftFrame.Tensors[0])
		if err != nil {
			continue
		}

		rightMat, err := tensorToMat(rightFrame.Tensors[0])
		if err != nil {
			leftMat.Close()
			continue
		}

		// Rectify frames
		leftRect, rightRect, err := calibration.Rectify(leftMat, rightMat)
		if err != nil {
			leftMat.Close()
			rightMat.Close()
			continue
		}

		// Combine frames side-by-side for display
		combinedMat := combineFramesSideBySide(leftRect, rightRect)
		combinedTensor, err := matToTensor(combinedMat)
		if err != nil {
			leftMat.Close()
			rightMat.Close()
			leftRect.Close()
			rightRect.Close()
			combinedMat.Close()
			continue
		}

		// Create display frame
		displayFrame := types.Frame{
			Tensors:  []types.Tensor{combinedTensor},
			Metadata: leftFrame.Metadata,
		}

		// Send to all destinations
		for _, dest := range dests {
			if err := dest.AddFrame(displayFrame); err != nil {
				// Ignore destination errors
			}
		}

		leftMat.Close()
		rightMat.Close()
		leftRect.Close()
		rightRect.Close()
		combinedMat.Close()
	}
}

// Helper functions
func tensorToMat(tensor types.Tensor) (cv.Mat, error) {
	return destination.TensorToMat(tensor)
}

func matToTensor(mat cv.Mat) (types.Tensor, error) {
	return tensorgocv.FromMat(mat, tensorgocv.WithAdoptedMat())
}

func combineFramesSideBySide(left, right cv.Mat) cv.Mat {
	// Combine two frames side-by-side
	// Create a new Mat with width = left.width + right.width, height = max(left.height, right.height)
	leftSize := left.Size()
	rightSize := right.Size()
	if len(leftSize) < 2 || len(rightSize) < 2 {
		return cv.NewMat()
	}

	leftHeight, leftWidth := leftSize[0], leftSize[1]
	rightHeight, rightWidth := rightSize[0], rightSize[1]

	combinedWidth := leftWidth + rightWidth
	combinedHeight := leftHeight
	if rightHeight > leftHeight {
		combinedHeight = rightHeight
	}

	combined := cv.NewMatWithSize(combinedHeight, combinedWidth, left.Type())
	defer combined.Close()

	// Copy left frame
	leftROI := combined.Region(image.Rect(0, 0, leftWidth, leftHeight))
	cv.Resize(left, &leftROI, image.Point{X: leftWidth, Y: leftHeight}, 0, 0, cv.InterpolationLinear)

	// Copy right frame
	rightROI := combined.Region(image.Rect(leftWidth, 0, combinedWidth, rightHeight))
	cv.Resize(right, &rightROI, image.Point{X: rightWidth, Y: rightHeight}, 0, 0, cv.InterpolationLinear)

	return combined.Clone()
}
