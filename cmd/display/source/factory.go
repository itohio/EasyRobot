package source

import (
	"bufio"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/gocv"
)

// Package-level flag variables shared by all sources
var (
	imagePaths FlagArray
	videoPaths FlagArray
	cameraIDs  FlagArray
	cameraWidth, cameraHeight int
	generateFlag bool
	enumerateCameras bool
)

// RegisterAllFlags registers flags for all source types.
// Call this before flag.Parse().
func RegisterAllFlags() {
	flag.Var(&imagePaths, "images", "Image file or directory path (can repeat)")
	flag.Var(&videoPaths, "video", "Video file path (can repeat)")
	flag.Var(&cameraIDs, "camera", "Camera device ID (can repeat)")
	flag.IntVar(&cameraWidth, "width", 640, "Frame width for cameras")
	flag.IntVar(&cameraHeight, "height", 480, "Frame height for cameras")
	flag.BoolVar(&generateFlag, "generate", false, "Generate test frames with spinning triangle")
	flag.BoolVar(&enumerateCameras, "enumerate-cameras", false, "Enumerate available cameras and configure interactively")
	// Note: Interest source flags will be registered separately if DNDM is enabled
}

// NewFromFlags creates a source based on which flags are set.
// Must be called after flag.Parse().
// Returns the selected source or an error if no source or multiple sources are specified.
func NewFromFlags() (Source, error) {
	slog.Info("Creating source from flags",
		"generate", generateFlag,
		"image_paths", len(imagePaths),
		"video_paths", len(videoPaths),
		"camera_ids", len(cameraIDs),
		"enumerate_cameras", enumerateCameras,
	)

	count := 0
	var selected Source
	var sourceType string

	if enumerateCameras {
		count++
		sourceType = "camera"
		camSrc, err := createEnumeratedCameraSource()
		if err != nil {
			return nil, fmt.Errorf("failed to create enumerated camera source: %w", err)
		}
		selected = camSrc
		slog.Info("Enumerated camera source selected")
	} else if len(cameraIDs) > 0 {
		// Regular camera source (not enumeration)
		count++
		sourceType = "camera"
		camSrc := NewCameraSource().(*cameraSource)
		camSrc.deviceIDs = cameraIDs
		camSrc.width = cameraWidth
		camSrc.height = cameraHeight
		selected = camSrc
		slog.Info("Camera source selected", "device_ids", cameraIDs, "width", cameraWidth, "height", cameraHeight)
	}

	if generateFlag {
		count++
		sourceType = "generator"
		genSrc := NewGeneratorSource().(*generatorSource)
		selected = genSrc
		slog.Info("Generator source selected")
	}

	if len(imagePaths) > 0 {
		count++
		sourceType = "images"
		imgSrc := NewImageSource().(*imageSource)
		imgSrc.paths = imagePaths
		selected = imgSrc
		slog.Info("Image source selected", "paths", imagePaths)
	}

	if len(videoPaths) > 0 {
		count++
		sourceType = "video"
		vidSrc := NewVideoSource().(*videoSource)
		vidSrc.paths = videoPaths
		selected = vidSrc
		slog.Info("Video source selected", "paths", videoPaths)
	}


	// Check for DNDM interest (must be exclusive)
	intSrc, err := NewInterestFromFlags()
	if err != nil {
		slog.Error("Failed to create interest source from flags", "err", err)
		return nil, err
	}
	if intSrc != nil {
		count++
		sourceType = "interest"
		selected = intSrc
		slog.Info("Interest source selected")
	}

	if count == 0 {
		slog.Error("No input source specified")
		return nil, fmt.Errorf("no input source specified (use --generate, --images, --video, --camera, --enumerate-cameras, or --interest)")
	}
	if count > 1 {
		slog.Error("Multiple sources specified", "count", count)
		return nil, fmt.Errorf("only one input type can be specified at a time")
	}

	slog.Info("Source created successfully", "type", sourceType)
	return selected, nil
}

// createEnumeratedCameraSource enumerates cameras and allows interactive configuration
func createEnumeratedCameraSource() (Source, error) {
	fmt.Println("=== Camera Enumeration ===")

	// Enumerate available cameras
	unmarshaller := gocv.NewUnmarshaller()
	var devices []gocv.CameraInfo
	err := unmarshaller.Unmarshal(strings.NewReader("list"), &devices)
	if err != nil {
		return nil, fmt.Errorf("failed to enumerate cameras: %w", err)
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("no cameras found")
	}

	fmt.Printf("Found %d camera(s):\n", len(devices))
	for _, dev := range devices {
		fmt.Printf("  [%d] %s (%s)\n", dev.ID, dev.Name, dev.Path)
		fmt.Printf("      Driver: %s\n", dev.Driver)
		fmt.Printf("      Formats: %d available\n", len(dev.SupportedFormats))
		if len(dev.SupportedFormats) > 0 {
			fmt.Printf("      Example format: %dx%d\n", dev.SupportedFormats[0].Width, dev.SupportedFormats[0].Height)
		}
		fmt.Printf("      Controls: %d available\n", len(dev.Controls))
		fmt.Println()
	}

	// Get user input for camera selection
	reader := bufio.NewReader(os.Stdin)
	selectedDevices := make([]int, 0)

	for {
		fmt.Print("Enter camera ID(s) to use (comma-separated, or 'all' for all cameras): ")
		input, err := reader.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("failed to read input: %w", err)
		}

		input = strings.TrimSpace(input)
		if input == "all" {
			for _, dev := range devices {
				selectedDevices = append(selectedDevices, dev.ID)
			}
			break
		}

		// Parse comma-separated IDs
		parts := strings.Split(input, ",")
		valid := true
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			id, err := strconv.Atoi(part)
			if err != nil {
				fmt.Printf("Invalid camera ID '%s'\n", part)
				valid = false
				break
			}

			// Check if ID exists
			found := false
			for _, dev := range devices {
				if dev.ID == id {
					found = true
					break
				}
			}
			if !found {
				fmt.Printf("Camera ID %d not found\n", id)
				valid = false
				break
			}

			selectedDevices = append(selectedDevices, id)
		}

		if valid && len(selectedDevices) > 0 {
			break
		}
		selectedDevices = selectedDevices[:0] // Reset for retry
	}

	// Get resolution for all cameras
	fmt.Print("Enter resolution (width x height, default 640x480): ")
	resInput, err := reader.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read resolution: %w", err)
	}

	width, height := 640, 480
	resInput = strings.TrimSpace(resInput)
	if resInput != "" {
		parts := strings.Split(resInput, "x")
		if len(parts) == 2 {
			if w, err := strconv.Atoi(strings.TrimSpace(parts[0])); err == nil {
				width = w
			}
			if h, err := strconv.Atoi(strings.TrimSpace(parts[1])); err == nil {
				height = h
			}
		}
	}

	// Get pixel format
	fmt.Print("Enter pixel format (default: MJPEG, options: MJPEG, YUYV, etc.): ")
	formatInput, err := reader.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read format: %w", err)
	}

	pixelFormat := "MJPEG"
	formatInput = strings.TrimSpace(formatInput)
	if formatInput != "" {
		pixelFormat = formatInput
	}

	// Get frame rate
	fmt.Print("Enter frame rate (default: 30): ")
	fpsInput, err := reader.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("failed to read frame rate: %w", err)
	}

	fps := 30
	fpsInput = strings.TrimSpace(fpsInput)
	if fpsInput != "" {
		if f, err := strconv.Atoi(fpsInput); err == nil && f > 0 {
			fps = f
		}
	}

	fmt.Printf("\nSelected configuration:\n")
	fmt.Printf("  Cameras: %v\n", selectedDevices)
	fmt.Printf("  Resolution: %dx%d\n", width, height)
	fmt.Printf("  Pixel Format: %s\n", pixelFormat)
	fmt.Printf("  Frame Rate: %d fps\n", fps)
	fmt.Println()

	// Create camera source with selected configuration
	camSrc := NewCameraSource().(*cameraSource)
	camSrc.width = width
	camSrc.height = height
	camSrc.pixelFormat = pixelFormat
	camSrc.frameRate = fps

	// Convert selected device IDs to strings for the FlagArray
	camSrc.deviceIDs = make(FlagArray, len(selectedDevices))
	for i, id := range selectedDevices {
		camSrc.deviceIDs[i] = strconv.Itoa(id)
	}

	return camSrc, nil
}

// GetCameraIDs returns the list of camera IDs that were specified via flags.
// This is useful for validation (e.g., ensuring exactly one camera for monocular calibration).
func GetCameraIDs() []string {
	return []string(cameraIDs)
}

