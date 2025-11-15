package source

import (
	"flag"
	"fmt"
)

// Package-level flag variables shared by all sources
var (
	imagePaths FlagArray
	videoPaths FlagArray
	cameraIDs  FlagArray
	cameraWidth, cameraHeight int
)

// RegisterAllFlags registers flags for all source types.
// Call this before flag.Parse().
func RegisterAllFlags() {
	flag.Var(&imagePaths, "images", "Image file or directory path (can repeat)")
	flag.Var(&videoPaths, "video", "Video file path (can repeat)")
	flag.Var(&cameraIDs, "camera", "Camera device ID (can repeat)")
	flag.IntVar(&cameraWidth, "width", 640, "Frame width for cameras")
	flag.IntVar(&cameraHeight, "height", 480, "Frame height for cameras")
	// Note: Interest source flags will be registered separately if DNDM is enabled
}

// NewFromFlags creates a source based on which flags are set.
// Must be called after flag.Parse().
// Returns the selected source or an error if no source or multiple sources are specified.
func NewFromFlags() (Source, error) {
	count := 0
	var selected Source

	if len(imagePaths) > 0 {
		count++
		imgSrc := NewImageSource().(*imageSource)
		imgSrc.paths = imagePaths
		selected = imgSrc
	}

	if len(videoPaths) > 0 {
		count++
		vidSrc := NewVideoSource().(*videoSource)
		vidSrc.paths = videoPaths
		selected = vidSrc
	}

	if len(cameraIDs) > 0 {
		count++
		camSrc := NewCameraSource().(*cameraSource)
		camSrc.deviceIDs = cameraIDs
		camSrc.width = cameraWidth
		camSrc.height = cameraHeight
		selected = camSrc
	}

	// Check for DNDM interest (must be exclusive)
	intSrc, err := NewInterestFromFlags()
	if err != nil {
		return nil, err
	}
	if intSrc != nil {
		count++
		selected = intSrc
	}

	if count == 0 {
		return nil, fmt.Errorf("no input source specified (use --images, --video, --camera, or --interest)")
	}
	if count > 1 {
		return nil, fmt.Errorf("only one input type can be specified at a time")
	}

	return selected, nil
}

// GetCameraIDs returns the list of camera IDs that were specified via flags.
// This is useful for validation (e.g., ensuring exactly one camera for monocular calibration).
func GetCameraIDs() []string {
	return []string(cameraIDs)
}

