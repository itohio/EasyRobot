package main

import (
	"fmt"
	"image"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	cv "gocv.io/x/gocv"
)

// parseGrid parses a grid size string in format "width,height"
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

// tensorToMat converts a tensor to gocv.Mat using the destination package helper
func tensorToMat(tensor types.Tensor) (cv.Mat, error) {
	return destination.TensorToMat(tensor)
}

// matToTensor converts a gocv.Mat to a tensor
func matToTensor(mat cv.Mat) (types.Tensor, error) {
	return tensorgocv.FromMat(mat, tensorgocv.WithAdoptedMat())
}

// combineFramesSideBySide combines two frames side-by-side
// Returns a new Mat that the caller must close
func combineFramesSideBySide(left, right cv.Mat) cv.Mat {
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

	// Copy left frame
	leftROI := combined.Region(image.Rect(0, 0, leftWidth, leftHeight))
	cv.Resize(left, &leftROI, image.Point{X: leftWidth, Y: leftHeight}, 0, 0, cv.InterpolationLinear)

	// Copy right frame
	rightROI := combined.Region(image.Rect(leftWidth, 0, combinedWidth, rightHeight))
	cv.Resize(right, &rightROI, image.Point{X: rightWidth, Y: rightHeight}, 0, 0, cv.InterpolationLinear)

	return combined
}

