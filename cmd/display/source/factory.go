package source

import (
	"flag"
	"fmt"
	"log/slog"
)

// Package-level flag variables shared by all sources
var (
	imagePaths FlagArray
	videoPaths FlagArray
	cameraIDs  FlagArray
	cameraWidth, cameraHeight int
	generateFlag bool
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
	)

	count := 0
	var selected Source
	var sourceType string

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

	if len(cameraIDs) > 0 {
		count++
		sourceType = "camera"
		camSrc := NewCameraSource().(*cameraSource)
		camSrc.deviceIDs = cameraIDs
		camSrc.width = cameraWidth
		camSrc.height = cameraHeight
		selected = camSrc
		slog.Info("Camera source selected", "device_ids", cameraIDs, "width", cameraWidth, "height", cameraHeight)
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
		return nil, fmt.Errorf("no input source specified (use --generate, --images, --video, --camera, or --interest)")
	}
	if count > 1 {
		slog.Error("Multiple sources specified", "count", count)
		return nil, fmt.Errorf("only one input type can be specified at a time")
	}

	slog.Info("Source created successfully", "type", sourceType)
	return selected, nil
}

// GetCameraIDs returns the list of camera IDs that were specified via flags.
// This is useful for validation (e.g., ensuring exactly one camera for monocular calibration).
func GetCameraIDs() []string {
	return []string(cameraIDs)
}

