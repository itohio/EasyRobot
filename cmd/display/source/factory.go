package source

import (
	"bufio"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/gocv"
	"github.com/spf13/cobra"
)

// AddSourceFlags adds source flags to a cobra command.
// Call this on any command that needs source flags.
func AddSourceFlags(cmd *cobra.Command) {
	cmd.Flags().StringArray("images", nil, "Image file or directory path (can repeat)")
	cmd.Flags().StringArray("video", nil, "Video file path (can repeat)")
	cmd.Flags().StringArray("camera", nil, "Camera device ID or configuration (can repeat). Format: ID[:widthxheight[@fps[/format]]]. Examples: '0', '0:640x480', '0:640x480@30', '0:640x480@30/mjpeg', '0@30/mjpeg'")
	cmd.Flags().Int("width", 640, "Frame width for cameras")
	cmd.Flags().Int("height", 480, "Frame height for cameras")
	cmd.Flags().Bool("generate", false, "Generate test frames with spinning triangle")
	cmd.Flags().Bool("list-cameras", false, "List all available cameras with their parameters and exit (does not start streaming)")
	
	// Generator-specific flags
	cmd.Flags().Int("generate-width", 640, "Width of generated frames")
	cmd.Flags().Int("generate-height", 480, "Height of generated frames")
	cmd.Flags().Int("generate-fps", 30, "Frames per second for generation")
	cmd.Flags().Duration("generate-duration", 0, "Duration to generate frames (0 = infinite)")
	
	// Note: Interest source flags will be registered separately if DNDM is enabled
}

// NewFromCobra creates a source based on flags from a cobra command.
// Returns the selected source or an error if no source or multiple sources are specified.
func NewFromCobra(cmd *cobra.Command) (Source, error) {
	// Read flags from cobra command
	imagePaths, _ := cmd.Flags().GetStringArray("images")
	videoPaths, _ := cmd.Flags().GetStringArray("video")
	cameraIDs, _ := cmd.Flags().GetStringArray("camera")
	cameraWidth, _ := cmd.Flags().GetInt("width")
	cameraHeight, _ := cmd.Flags().GetInt("height")
	generateFlag, _ := cmd.Flags().GetBool("generate")
	enumerateCameras, _ := cmd.Flags().GetBool("list-cameras")

	slog.Info("Creating source from cobra command",
		"generate", generateFlag,
		"image_paths", len(imagePaths),
		"video_paths", len(videoPaths),
		"camera_ids", len(cameraIDs),
		"enumerate_cameras", enumerateCameras,
	)

	count := 0
	var selected Source
	var sourceType string

	// Note: enumerateCameras (list-cameras) should be handled in main.go before this function
	// It lists cameras and exits, so it should never reach here
	if len(cameraIDs) > 0 {
		// Regular camera source (not enumeration)
		count++
		sourceType = "camera"
		camSrc := NewCameraSource().(*cameraSource)
		camSrc.deviceIDs = FlagArray(cameraIDs)
		camSrc.width = cameraWidth
		camSrc.height = cameraHeight
		selected = camSrc
		slog.Info("Camera source selected", "device_ids", cameraIDs, "width", cameraWidth, "height", cameraHeight)
	}

	if generateFlag {
		count++
		sourceType = "generator"
		genSrc := NewGeneratorSource().(*generatorSource)
		
		// Set generator-specific flags
		genWidth, _ := cmd.Flags().GetInt("generate-width")
		genHeight, _ := cmd.Flags().GetInt("generate-height")
		genFPS, _ := cmd.Flags().GetInt("generate-fps")
		genDuration, _ := cmd.Flags().GetDuration("generate-duration")
		
		genSrc.width = genWidth
		genSrc.height = genHeight
		genSrc.fps = genFPS
		genSrc.duration = genDuration
		
		selected = genSrc
		slog.Info("Generator source selected", "width", genWidth, "height", genHeight, "fps", genFPS, "duration", genDuration)
	}

	if len(imagePaths) > 0 {
		count++
		sourceType = "images"
		imgSrc := NewImageSource().(*imageSource)
		imgSrc.paths = FlagArray(imagePaths)
		selected = imgSrc
		slog.Info("Image source selected", "paths", imagePaths)
	}

	if len(videoPaths) > 0 {
		count++
		sourceType = "video"
		vidSrc := NewVideoSource().(*videoSource)
		vidSrc.paths = FlagArray(videoPaths)
		selected = vidSrc
		slog.Info("Video source selected", "paths", videoPaths)
	}

	// Check for DNDM interest (must be exclusive)
	intSrc, err := NewInterestFromCobra(cmd)
	if err != nil {
		slog.Error("Failed to create interest source from cobra command", "err", err)
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
		return nil, fmt.Errorf("no input source specified (use --generate, --images, --video, --camera, --list-cameras, or --interest)")
	}
	if count > 1 {
		slog.Error("Multiple sources specified", "count", count)
		return nil, fmt.Errorf("only one input type can be specified at a time")
	}

	slog.Info("Source created successfully", "type", sourceType)
	return selected, nil
}

// IsListCamerasFlagSet returns true if the list-cameras flag is set in the cobra command.
func IsListCamerasFlagSet(cmd *cobra.Command) bool {
	enumerateCameras, _ := cmd.Flags().GetBool("list-cameras")
	return enumerateCameras
}

// ListCameras lists all available cameras with all their parameters and exits
func ListCameras() error {
	fmt.Println("=== Camera List ===")

	// Enumerate available cameras
	unmarshaller := gocv.NewUnmarshaller()
	var devices []gocv.CameraInfo
	err := unmarshaller.Unmarshal(strings.NewReader("list"), &devices)
	if err != nil {
		return fmt.Errorf("failed to enumerate cameras: %w", err)
	}

	if len(devices) == 0 {
		fmt.Println("No cameras found")
		return nil
	}

	fmt.Printf("Found %d camera(s):\n\n", len(devices))
	for i, dev := range devices {
		fmt.Printf("Camera %d:\n", i)
		fmt.Printf("  ID: %d\n", dev.ID)
		fmt.Printf("  Name: %s\n", dev.Name)
		fmt.Printf("  Path: %s\n", dev.Path)
		fmt.Printf("  Driver: %s\n", dev.Driver)
		fmt.Printf("  Card: %s\n", dev.Card)
		fmt.Printf("  Bus Info: %s\n", dev.BusInfo)
		fmt.Printf("  Capabilities: %v\n", dev.Capabilities)

		// List all supported formats
		fmt.Printf("  Supported Formats (%d):\n", len(dev.SupportedFormats))
		if len(dev.SupportedFormats) == 0 {
			fmt.Printf("    (none listed)\n")
		} else {
			for j, format := range dev.SupportedFormats {
				fmt.Printf("    [%d] %s - %s (%dx%d)\n", j, format.PixelFormat, format.Description, format.Width, format.Height)
			}
		}

		// List all controls with their parameters
		fmt.Printf("  Controls (%d):\n", len(dev.Controls))
		if len(dev.Controls) == 0 {
			fmt.Printf("    (none listed)\n")
		} else {
			for j, ctrl := range dev.Controls {
				fmt.Printf("    [%d] %s (%s)\n", j, ctrl.Name, ctrl.Description)
				fmt.Printf("        Type: %s\n", ctrl.Type)
				if ctrl.Type == "integer" {
					fmt.Printf("        Range: %d - %d (default: %d, step: %d)\n", ctrl.Min, ctrl.Max, ctrl.Default, ctrl.Step)
				} else if ctrl.Type == "boolean" {
					fmt.Printf("        Default: %v\n", ctrl.Default != 0)
				} else {
					fmt.Printf("        Default: %d\n", ctrl.Default)
				}
			}
		}
		fmt.Println()
	}

	return nil
}

// createEnumeratedCameraSource enumerates cameras and allows interactive configuration
// DEPRECATED: This function is no longer used. Use ListCameras() instead for listing,
// and use --camera flag with device IDs for streaming via cobra.
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

// GetCameraIDs returns the list of camera IDs from a cobra command.
// This is useful for validation (e.g., ensuring exactly one camera for monocular calibration).
func GetCameraIDs(cmd *cobra.Command) []string {
	cameraIDs, _ := cmd.Flags().GetStringArray("camera")
	return cameraIDs
}

