package corners

import (
	"fmt"
	"image"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"gocv.io/x/gocv"
)

// Processor implements chain.Processor[types.Tensor, types.Tensor] for corner detection.
// It detects corners (e.g., chessboard corners for calibration) from input image tensors.
//
// Allocations: This processor preallocates gocv.Mat objects to avoid allocations
// in the Process() callstack. Tensors are created from adopted Mat objects.
type Processor struct {
	gridSize image.Point

	// Preallocated scratch space to avoid allocations in Process()
	scratchGray    gocv.Mat
	scratchCorners gocv.Mat

	// Preallocated for corner refinement
	scratchCriteria gocv.TermCriteria
}

// New creates a new corner detection processor.
func New(gridSize image.Point) *Processor {
	return &Processor{
		gridSize:        gridSize,
		scratchGray:     gocv.NewMat(),
		scratchCorners:  gocv.NewMat(),
		scratchCriteria: gocv.NewTermCriteria(gocv.Count+gocv.EPS, 30, 0.001),
	}
}

// Process implements chain.Processor[types.Tensor, types.Tensor].
// Reads input tensor(s) from src, detects corners, and writes output tensor(s) to dst.
//
// Input: Image tensor (gocv.Mat-backed)
// Output: Corner points tensor or image with corners drawn (depending on mode)
//
// If dst is nil, returns the estimated number of output tensors (1 tensor per input tensor).
func (p *Processor) Process(dst buffer.Writer[types.Tensor], src buffer.Reader[types.Tensor]) int {
	if dst == nil {
		// Size estimation: 1 tensor in -> 1 tensor out
		return src.Len()
	}

	// Read input tensor(s)
	inputTensors := make([]types.Tensor, 1)
	n, err := src.Read(inputTensors)
	if err != nil || n == 0 {
		return 0
	}

	inputTensor := inputTensors[0]

	// Convert tensor to gocv.Mat
	gocvTensor, ok := inputTensor.(tensorgocv.Tensor)
	if !ok {
		return 0
	}

	mat, err := gocvTensor.Mat()
	if err != nil {
		return 0
	}

	// Detect corners
	patternSize := image.Point{X: p.gridSize.X, Y: p.gridSize.Y}

	// Convert to grayscale if needed (using preallocated scratch)
	if mat.Channels() == 1 {
		// Already grayscale, copy to scratch for processing
		if p.scratchGray.Empty() || p.scratchGray.Rows() != mat.Rows() || p.scratchGray.Cols() != mat.Cols() {
			p.scratchGray.Close()
			p.scratchGray = gocv.NewMat()
		}
		mat.CopyTo(&p.scratchGray)
	} else {
		// Convert to grayscale
		if p.scratchGray.Empty() || p.scratchGray.Rows() != mat.Rows() || p.scratchGray.Cols() != mat.Cols() {
			p.scratchGray.Close()
			p.scratchGray = gocv.NewMat()
		}
		gocv.CvtColor(mat, &p.scratchGray, gocv.ColorBGRToGray)
	}

	// Find chessboard corners
	found := gocv.FindChessboardCornersSB(
		p.scratchGray,
		patternSize,
		&p.scratchCorners,
		gocv.CalibCBAdaptiveThresh|gocv.CalibCBNormalizeImage,
	)

	if !found || p.scratchCorners.Empty() {
		return 0
	}

	// Refine corner positions (CornersSubPix modifies the Mat in-place)
	winSize := image.Point{X: 11, Y: 11}
	zeroZone := image.Point{X: -1, Y: -1}
	if err := gocv.CornerSubPix(p.scratchGray, &p.scratchCorners, winSize, zeroZone, p.scratchCriteria); err != nil {
		return 0
	}

	// Create output tensor from corners Mat
	// Note: scratchCorners is a gocv.Mat, we adopt it
	outputTensor, err := tensorgocv.FromMat(p.scratchCorners, tensorgocv.WithAdoptedMat())
	if err != nil {
		return 0
	}

	// Create new scratch for next iteration (previous one is now owned by tensor)
	p.scratchCorners = gocv.NewMat()

	// Write output tensor
	outputTensors := []types.Tensor{outputTensor}
	written, _ := dst.Write(outputTensors)
	return written
}

// Reset resets the processor state.
func (p *Processor) Reset() {
	// Clear scratch space
	if !p.scratchGray.Empty() {
		p.scratchGray.Close()
		p.scratchGray = gocv.NewMat()
	}
	if !p.scratchCorners.Empty() {
		p.scratchCorners.Close()
		p.scratchCorners = gocv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	if !p.scratchGray.Empty() {
		p.scratchGray.Close()
	}
	if !p.scratchCorners.Empty() {
		p.scratchCorners.Close()
	}
}

// Draw draws detected corners on the given image tensor.
// The image tensor is modified in place.
// Returns an error if the tensor cannot be converted to gocv.Mat or if drawing fails.
func (p *Processor) Draw(dst types.Tensor, corners []gocv.Point2f, found bool) error {
	if !found || len(corners) == 0 {
		return nil
	}

	// Convert tensor to gocv.Mat
	gocvTensor, ok := dst.(tensorgocv.Tensor)
	if !ok {
		return nil // Skip if not a gocv tensor
	}

	mat, err := gocvTensor.Mat()
	if err != nil {
		return err
	}

	// Create Mat from corners slice
	cornersMat := gocv.NewMatWithSize(len(corners), 1, gocv.MatTypeCV32FC2)
	defer cornersMat.Close()

	// Fill the Mat with corner points
	for i, pt := range corners {
		cornersMat.SetFloatAt3(i, 0, 0, pt.X) // row i, col 0, channel 0 (x)
		cornersMat.SetFloatAt3(i, 0, 1, pt.Y) // row i, col 0, channel 1 (y)
	}

	patternSize := image.Point{X: p.gridSize.X, Y: p.gridSize.Y}

	if err := gocv.DrawChessboardCorners(&mat, patternSize, cornersMat, found); err != nil {
		return err
	}

	return nil
}

// Detect detects chessboard corners in an image.
// Returns true if corners were found, along with the corner points.
// The corners are refined to sub-pixel accuracy.
// This method can be used directly without going through the Process() method.
func (p *Processor) Detect(mat gocv.Mat) (bool, []gocv.Point2f, error) {
	if mat.Empty() {
		return false, nil, fmt.Errorf("empty mat")
	}

	// Convert to grayscale if needed (using preallocated scratch)
	if mat.Channels() == 1 {
		// Already grayscale, copy to scratch for processing
		if p.scratchGray.Empty() || p.scratchGray.Rows() != mat.Rows() || p.scratchGray.Cols() != mat.Cols() {
			p.scratchGray.Close()
			p.scratchGray = gocv.NewMat()
		}
		mat.CopyTo(&p.scratchGray)
	} else {
		// Convert to grayscale
		if p.scratchGray.Empty() || p.scratchGray.Rows() != mat.Rows() || p.scratchGray.Cols() != mat.Cols() {
			p.scratchGray.Close()
			p.scratchGray = gocv.NewMat()
		}
		gocv.CvtColor(mat, &p.scratchGray, gocv.ColorBGRToGray)
	}

	// Find chessboard corners
	patternSize := image.Point{X: p.gridSize.X, Y: p.gridSize.Y}

	if p.scratchCorners.Empty() {
		p.scratchCorners = gocv.NewMat()
	}

	found := gocv.FindChessboardCornersSB(
		p.scratchGray,
		patternSize,
		&p.scratchCorners,
		gocv.CalibCBAdaptiveThresh|gocv.CalibCBNormalizeImage,
	)

	if !found || p.scratchCorners.Empty() {
		return false, nil, nil
	}

	// Refine corner positions (CornersSubPix modifies the Mat in-place)
	winSize := image.Point{X: 11, Y: 11}
	zeroZone := image.Point{X: -1, Y: -1}
	if err := gocv.CornerSubPix(p.scratchGray, &p.scratchCorners, winSize, zeroZone, p.scratchCriteria); err != nil {
		return false, nil, fmt.Errorf("corner refinement failed: %w", err)
	}

	// Convert refined corners Mat to Point2fVector and then to slice
	cornersVec := gocv.NewPoint2fVectorFromMat(p.scratchCorners)
	defer cornersVec.Close()

	if cornersVec.Size() == 0 {
		return false, nil, nil
	}

	imgPts := cornersVec.ToPoints()

	// Validate we have the correct number of corners
	expectedCorners := p.gridSize.X * p.gridSize.Y
	if len(imgPts) != expectedCorners {
		return false, nil, fmt.Errorf("found %d corners, expected %d", len(imgPts), expectedCorners)
	}

	return true, imgPts, nil
}

// Detector is a type alias for Processor for backwards compatibility with cmd/calibrate.
type Detector = Processor

// NewDetector creates a new corner detector with the specified grid size.
// This is an alias for New() for backwards compatibility with cmd/calibrate.
func NewDetector(gridSize image.Point) *Detector {
	return New(gridSize)
}

// GenerateObjectPoints generates 3D object points for the calibration pattern.
// Assumes square size of 1.0 (normalized units).
func GenerateObjectPoints(gridSize image.Point) []gocv.Point3f {
	points := make([]gocv.Point3f, 0, gridSize.X*gridSize.Y)
	for y := 0; y < gridSize.Y; y++ {
		for x := 0; x < gridSize.X; x++ {
			points = append(points, gocv.Point3f{
				X: float32(x),
				Y: float32(y),
				Z: 0.0,
			})
		}
	}
	return points
}

// Draw draws detected corners on the image.
// If corners are nil or empty, nothing is drawn.
// This is a package-level function for convenience.
func Draw(mat gocv.Mat, gridSize image.Point, corners []gocv.Point2f, found bool) error {
	if !found || len(corners) == 0 {
		return nil
	}

	// Create Mat from corners slice
	cornersMat := gocv.NewMatWithSize(len(corners), 1, gocv.MatTypeCV32FC2)
	defer cornersMat.Close()

	// Fill the Mat with corner points
	for i, pt := range corners {
		cornersMat.SetFloatAt3(i, 0, 0, pt.X) // row i, col 0, channel 0 (x)
		cornersMat.SetFloatAt3(i, 0, 1, pt.Y) // row i, col 0, channel 1 (y)
	}

	patternSize := image.Point{X: gridSize.X, Y: gridSize.Y}

	if err := gocv.DrawChessboardCorners(&mat, patternSize, cornersMat, found); err != nil {
		return fmt.Errorf("failed to draw corners: %w", err)
	}

	return nil
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[types.Tensor, types.Tensor] = (*Processor)(nil)
