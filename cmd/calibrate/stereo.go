package main

import (
	"context"
	"fmt"
	"image"
	"os"
	"os/signal"
	"strconv"
	"syscall"

	"github.com/itohio/EasyRobot/cmd/calibrate/corners"
	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/itohio/EasyRobot/x/marshaller/gocv"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/spf13/cobra"
	cv "gocv.io/x/gocv"
)

func newCamStereoCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cam-stereo",
		Short: "Stereo camera calibration",
		Long:  "Calibrate a stereo camera pair using a checkerboard pattern. Collects calibration samples and saves stereo camera parameters.",
		RunE:  runCamStereo,
	}

	// Add destination flags
	destination.AddDestinationFlags(cmd)

	// Calibration-specific flags
	cmd.Flags().String("grid", "9,7", "Calibration grid shape (width,height)")
	cmd.Flags().Int("samples", 30, "Number of calibration samples to collect")
	cmd.Flags().String("output", "stereo_camera.json", "Output file path")
	cmd.Flags().String("format", "json", "Output format: json, yaml, gocv")
	cmd.Flags().Bool("test", false, "Test mode - load existing calibration file")
	cmd.Flags().Bool("disparity", false, "Calculate and display disparity map (test mode only)")

	// Source flags for stereo (left and right)
	cmd.Flags().String("left-camera", "", "Left camera device ID")
	cmd.Flags().String("right-camera", "", "Right camera device ID")
	cmd.Flags().String("left-video", "", "Left video file path")
	cmd.Flags().String("right-video", "", "Right video file path")
	cmd.Flags().String("left-images", "", "Left image file or directory path")
	cmd.Flags().String("right-images", "", "Right image file or directory path")
	cmd.Flags().Int("width", 640, "Frame width for cameras")
	cmd.Flags().Int("height", 480, "Frame height for cameras")

	return cmd
}

func runCamStereo(cmd *cobra.Command, args []string) error {
	// Read calibration-specific flags
	gridStr, _ := cmd.Flags().GetString("grid")
	samples, _ := cmd.Flags().GetInt("samples")
	output, _ := cmd.Flags().GetString("output")
	format, _ := cmd.Flags().GetString("format")
	testMode, _ := cmd.Flags().GetBool("test")
	disparity, _ := cmd.Flags().GetBool("disparity")

	// Read source flags
	leftCamera, _ := cmd.Flags().GetString("left-camera")
	rightCamera, _ := cmd.Flags().GetString("right-camera")
	leftVideo, _ := cmd.Flags().GetString("left-video")
	rightVideo, _ := cmd.Flags().GetString("right-video")
	leftImages, _ := cmd.Flags().GetString("left-images")
	rightImages, _ := cmd.Flags().GetString("right-images")
	width, _ := cmd.Flags().GetInt("width")
	height, _ := cmd.Flags().GetInt("height")

	// Parse grid size
	gridWidth, gridHeight, err := parseGrid(gridStr)
	if err != nil {
		return fmt.Errorf("error parsing grid: %w", err)
	}

	// Create left and right sources
	leftSrc, err := createStereoSource(leftCamera, leftVideo, leftImages, width, height, "left")
	if err != nil {
		return fmt.Errorf("error creating left source: %w", err)
	}

	rightSrc, err := createStereoSource(rightCamera, rightVideo, rightImages, width, height, "right")
	if err != nil {
		leftSrc.Close()
		return fmt.Errorf("error creating right source: %w", err)
	}

	// Create destinations from cobra command
	dests := destination.NewFromCobra(cmd)

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
		return fmt.Errorf("error starting left source: %w", err)
	}
	defer leftSrc.Close()

	if err := rightSrc.Start(ctx); err != nil {
		rightSrc.Close()
		return fmt.Errorf("error starting right source: %w", err)
	}
	defer rightSrc.Close()

	// Start destinations
	for _, dest := range dests {
		if err := dest.Start(ctx); err != nil {
			return fmt.Errorf("error starting destination: %w", err)
		}
		defer dest.Close()
	}

	// Process frames
	if testMode {
		if err := testStereoCalibration(ctx, leftSrc, rightSrc, dests, output, disparity); err != nil {
			return fmt.Errorf("error testing calibration: %w", err)
		}
	} else {
		processor := NewStereoCalibrationProcessor(image.Point{X: gridWidth, Y: gridHeight}, samples)
		defer processor.Close()
		if err := calibrateStereoCamera(ctx, leftSrc, rightSrc, processor, dests, output, format); err != nil {
			return fmt.Errorf("error calibrating stereo cameras: %w", err)
		}
	}

	return nil
}

func createStereoSource(cameraID, videoPath, imagesPath string, width, height int, side string) (source.Source, error) {
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

// Custom source implementations for stereo calibration
type cameraSource struct {
	baseSource
	deviceID int
	width    int
	height   int
}

func createCameraSource(deviceID, width, height int) source.Source {
	return &cameraSource{deviceID: deviceID, width: width, height: height}
}

func (s *cameraSource) RegisterFlags() {}

func (s *cameraSource) Start(ctx context.Context) error {
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

type videoSource struct {
	baseSource
	path string
}

func createVideoSource(path string) source.Source {
	return &videoSource{path: path}
}

func (s *videoSource) RegisterFlags() {}

func (s *videoSource) Start(ctx context.Context) error {
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

type imageSource struct {
	baseSource
	path string
}

func createImageSource(path string) source.Source {
	return &imageSource{path: path}
}

func (s *imageSource) RegisterFlags() {}

func (s *imageSource) Start(ctx context.Context) error {
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

