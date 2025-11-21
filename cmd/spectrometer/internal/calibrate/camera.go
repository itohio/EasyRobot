package calibrate

import (
	"context"
	"fmt"
	"image"
	"image/color"

	"log/slog"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/spf13/cobra"
	cv "gocv.io/x/gocv"
)

// RunCameraCalibration runs the camera calibration command
func RunCameraCalibration(ctx context.Context, cmd *cobra.Command) error {
	slog.Info("Starting camera calibration")

	// Create camera source
	src, err := source.NewFromCobra(cmd)
	if err != nil {
		return fmt.Errorf("failed to create camera source: %w", err)
	}
	defer src.Close()

	// Start source
	if err := src.Start(ctx); err != nil {
		return fmt.Errorf("failed to start camera source: %w", err)
	}

	// Create display destination
	dests := destination.NewFromCobra(cmd)
	if len(dests) == 0 {
		return fmt.Errorf("no display destination configured (use --display flag or remove --no-display)")
	}
	displayDest := dests[0]

	// Start destination
	if err := displayDest.Start(ctx); err != nil {
		return fmt.Errorf("failed to start display destination: %w", err)
	}
	defer displayDest.Close()

	// Capture a frame
	slog.Info("Capturing frame from camera")
	frame, err := src.ReadFrame()
	if err != nil {
		return fmt.Errorf("failed to read frame: %w", err)
	}

	// Extract gocv Mat from frame
	mat, err := extractMatFromFrame(frame)
	if err != nil {
		return fmt.Errorf("failed to extract image from frame: %w", err)
	}
	defer mat.Close()

	// Detect strip bounding box
	slog.Info("Detecting strip bounding box")
	bbox, err := detectStripBoundingBox(&mat)
	if err != nil {
		return fmt.Errorf("failed to detect strip: %w", err)
	}

	if bbox == nil {
		slog.Warn("No strip detected in image")
	} else {
		slog.Info("Strip detected",
			"x", bbox.Min.X,
			"y", bbox.Min.Y,
			"width", bbox.Dx(),
			"height", bbox.Dy(),
		)

		// Draw bounding box on image
		drawBoundingBox(&mat, bbox)
	}

	// Create frame with annotated image
	annotatedFrame := createFrameFromMat(&mat, frame.Index)

	// Display the frame
	if err := displayDest.AddFrame(annotatedFrame); err != nil {
		return fmt.Errorf("failed to display frame: %w", err)
	}

	slog.Info("Camera calibration frame displayed", "bbox_detected", bbox != nil)

	return nil
}

// extractMatFromFrame extracts gocv.Mat from types.Frame
func extractMatFromFrame(frame types.Frame) (cv.Mat, error) {
	if len(frame.Tensors) == 0 {
		return cv.Mat{}, fmt.Errorf("frame has no tensors")
	}

	// Try to extract Mat from first tensor
	tensor := frame.Tensors[0]

	// Check if tensor is a gocv tensor (Accessor interface)
	if gocvTensor, ok := tensor.(tensorgocv.Accessor); ok {
		// Clone the mat so we can modify it
		mat, err := gocvTensor.MatClone()
		if err != nil {
			return cv.Mat{}, fmt.Errorf("failed to clone mat: %w", err)
		}
		return mat, nil
	}

	return cv.Mat{}, fmt.Errorf("frame tensor is not a gocv tensor")
}

// detectStripBoundingBox detects the spectral strip bounding box in the image
// This is a placeholder implementation - will be enhanced with actual detection algorithm
func detectStripBoundingBox(img *cv.Mat) (*image.Rectangle, error) {
	if img.Empty() {
		return nil, fmt.Errorf("image is empty")
	}

	// Convert to grayscale for processing
	gray := cv.NewMat()
	defer gray.Close()
	cv.CvtColor(*img, &gray, cv.ColorBGRToGray)

	// Apply threshold to find bright strip
	thresh := cv.NewMat()
	defer thresh.Close()
	cv.Threshold(gray, &thresh, 100, 255, cv.ThresholdBinary)

	// Find contours
	contours := cv.FindContours(thresh, cv.RetrievalExternal, cv.ChainApproxSimple)

	// Find largest contour (likely the strip)
	var largestRect image.Rectangle
	maxArea := 0.0

	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		rect := cv.BoundingRect(contour)
		area := float64(rect.Dx() * rect.Dy())

		if area > maxArea {
			maxArea = area
			largestRect = rect
		}
	}

	// Filter out very small detections
	if maxArea < 1000 {
		return nil, nil
	}

	return &largestRect, nil
}

// drawBoundingBox draws a bounding box on the image
func drawBoundingBox(img *cv.Mat, bbox *image.Rectangle) {
	if bbox == nil {
		return
	}

	// Draw rectangle in green
	green := color.RGBA{0, 255, 0, 255}
	cv.Rectangle(img, *bbox, green, 3)

	// Draw text label
	text := fmt.Sprintf("Strip: %dx%d", bbox.Dx(), bbox.Dy())
	pt := image.Pt(bbox.Min.X, bbox.Min.Y-10)
	white := color.RGBA{255, 255, 255, 255}
	cv.PutText(img, text, pt, cv.FontHersheyPlain, 1.0, white, 2)
}

// createFrameFromMat creates a types.Frame from gocv.Mat
func createFrameFromMat(mat *cv.Mat, index int) types.Frame {
	// Convert Mat to tensor (share the mat, don't adopt since we'll close it separately)
	tensor, err := tensorgocv.FromMat(*mat, tensorgocv.WithSharedMat())
	if err != nil {
		// If conversion fails, create frame with empty tensors
		return types.Frame{
			Index:   index,
			Tensors: []types.Tensor{},
			Metadata: map[string]interface{}{
				"source": "camera_calibration",
				"error":  err.Error(),
			},
		}
	}

	// Create frame
	frame := types.Frame{
		Index:   index,
		Tensors: []types.Tensor{tensor},
		Metadata: map[string]interface{}{
			"source": "camera_calibration",
		},
	}

	return frame
}
