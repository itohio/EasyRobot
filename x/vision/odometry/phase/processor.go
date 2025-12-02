package phase

import (
	"image"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	visiontypes "github.com/itohio/EasyRobot/x/vision/types"
	"gocv.io/x/gocv"
)

// Processor implements chain.Processor[visiontypes.Displacement, types.Tensor] for phase correlation matching.
// It computes the displacement between two images using phase correlation (DFT-based).
// Useful for tracking frame-to-frame motion.
//
// Allocations: This processor preallocates Mat objects and DFT buffers to avoid allocations
// in the Process() callstack.
type Processor struct {
	// Image size for preallocation
	imgSize image.Point

	// Skip frames before computing correlation (for warmup)
	skip      int
	skipCount int

	// DFT sizes (optimal sizes for DFT computation)
	dftM int
	dftN int

	// Current and previous frame DFT buffers
	curDFT  gocv.Mat // Current frame DFT
	lastDFT gocv.Mat // Previous frame DFT

	// Scratch buffers for processing
	scratchGray        gocv.Mat
	scratchRealInput   gocv.Mat
	scratchImagInput   gocv.Mat
	scratchComplexIn   gocv.Mat
	scratchComplexR    gocv.Mat
	scratchComplexPC   gocv.Mat
	scratchRealPC      gocv.Mat // Real part of phase correlation result (DFT-sized)
	scratchImagPC      gocv.Mat // Imaginary part (DFT-sized, for splitting)
	scratchCorrelation gocv.Mat // Image-sized ROI from real part
}

// Option configures a phase correlation processor.
type Option func(*Processor)

// WithSkip sets the number of frames to skip before computing correlation (warmup period).
func WithSkip(skip int) Option {
	return func(p *Processor) {
		p.skip = skip
	}
}

// New creates a new phase correlation processor with the given image size.
// The image size is used to preallocate buffers for optimal DFT sizes.
func New(imgSize image.Point, opts ...Option) *Processor {
	// Calculate optimal DFT sizes
	dftM := gocv.GetOptimalDFTSize(imgSize.X - 1)
	dftN := gocv.GetOptimalDFTSize(imgSize.Y - 1)

	p := &Processor{
		imgSize:   imgSize,
		skip:      5, // Default: skip 5 frames for warmup
		skipCount: 0,
		dftM:      dftM,
		dftN:      dftN,
	}

	for _, opt := range opts {
		opt(p)
	}

	// Preallocate DFT buffers (complex matrices: 2 channels for real/imaginary)
	p.curDFT = gocv.NewMatWithSize(dftN, dftM, gocv.MatTypeCV64FC2)
	p.lastDFT = gocv.NewMatWithSize(dftN, dftM, gocv.MatTypeCV64FC2)

	// Preallocate scratch buffers
	p.scratchGray = gocv.NewMatWithSize(imgSize.Y, imgSize.X, gocv.MatTypeCV8UC1)
	p.scratchRealInput = gocv.NewMatWithSize(imgSize.Y, imgSize.X, gocv.MatTypeCV64FC1)
	p.scratchImagInput = gocv.NewMatWithSize(imgSize.Y, imgSize.X, gocv.MatTypeCV64FC1)
	p.scratchComplexIn = gocv.NewMatWithSize(imgSize.Y, imgSize.X, gocv.MatTypeCV64FC2)
	p.scratchComplexR = gocv.NewMatWithSize(dftN, dftM, gocv.MatTypeCV64FC2)
	p.scratchComplexPC = gocv.NewMatWithSize(dftN, dftM, gocv.MatTypeCV64FC2)
	p.scratchRealPC = gocv.NewMatWithSize(dftN, dftM, gocv.MatTypeCV64FC1)
	p.scratchImagPC = gocv.NewMatWithSize(dftN, dftM, gocv.MatTypeCV64FC1)
	p.scratchCorrelation = gocv.NewMatWithSize(imgSize.Y, imgSize.X, gocv.MatTypeCV64FC1)

	return p
}

// Process implements chain.Processor[visiontypes.Displacement, types.Tensor].
// Reads image tensor from src, computes phase correlation with previous frame, and writes displacement to dst.
//
// Input: Image tensor (gocv.Mat-backed, BGR or grayscale)
// Output: Displacement containing dx, dy, and peak correlation value
//
// Note: The processor maintains state (previous frame), so it needs sequential frame input.
func (p *Processor) Process(dst buffer.Writer[visiontypes.Displacement], src buffer.Reader[types.Tensor]) int {
	if dst == nil {
		return src.Len()
	}

	// Read input image tensor
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

	if mat.Empty() {
		return 0
	}

	// Convert to grayscale if needed
	if mat.Channels() > 1 {
		gocv.CvtColor(mat, &p.scratchGray, gocv.ColorBGRToGray)
	} else {
		mat.CopyTo(&p.scratchGray)
	}

	// Swap current and previous DFT buffers
	p.curDFT, p.lastDFT = p.lastDFT, p.curDFT

	// Process current frame (compute DFT)
	p.processImage(&p.scratchGray)

	// Skip frames during warmup
	if p.skipCount < p.skip {
		p.skipCount++
		displacement := visiontypes.Displacement{DX: 0, DY: 0, PeakValue: 0}
		outputs := []visiontypes.Displacement{displacement}
		written, _ := dst.Write(outputs)
		return written
	}

	// Compute phase correlation
	dx, dy, peakValue := p.phaseCorrelation()

	// Create output displacement
	displacement := visiontypes.Displacement{
		DX:        float64(dx),
		DY:        float64(dy),
		PeakValue: peakValue,
	}

	outputs := []visiontypes.Displacement{displacement}
	written, _ := dst.Write(outputs)
	return written
}

// processImage processes an image and computes its DFT.
// This is called for each frame to prepare it for phase correlation.
func (p *Processor) processImage(img *gocv.Mat) {
	// Convert image to float64 and create complex input
	img.ConvertTo(&p.scratchRealInput, gocv.MatTypeCV64F)

	// Zero imaginary part
	p.scratchImagInput.SetTo(gocv.NewScalar(0, 0, 0, 0))

	// Merge real and imaginary parts into complex matrix
	planes := []gocv.Mat{p.scratchRealInput, p.scratchImagInput}
	gocv.Merge(planes, &p.scratchComplexIn)

	// Zero current DFT buffer
	p.curDFT.SetTo(gocv.NewScalar(0, 0, 0, 0))

	// Copy complex input to top-left of DFT buffer (will be padded automatically)
	roiRect := image.Rect(0, 0, img.Cols(), img.Rows())
	roi := p.curDFT.Region(roiRect)
	p.scratchComplexIn.CopyTo(&roi)
	roi.Close()

	// Compute DFT (forward transform)
	gocv.DFT(p.curDFT, &p.curDFT, gocv.DftForward)
}

// phaseCorrelation computes phase correlation between current and previous frames.
// Returns displacement (dx, dy) and peak correlation value.
func (p *Processor) phaseCorrelation() (int, int, float64) {
	// Step 1: Multiply current DFT with complex conjugate of previous DFT
	gocv.MulSpectrums(p.curDFT, p.lastDFT, &p.scratchComplexR, 0)

	// Step 2: Compute inverse DFT with scaling
	// DftInverse | DftScale performs inverse transform and scales by 1/N
	gocv.DFT(p.scratchComplexR, &p.scratchComplexPC, gocv.DftInverse|gocv.DftScale)

	// Step 3: Split complex result into real and imaginary parts
	planes := gocv.Split(p.scratchComplexPC)
	planes[0].CopyTo(&p.scratchRealPC)
	planes[1].CopyTo(&p.scratchImagPC)
	// Close temporary planes
	for _, plane := range planes {
		plane.Close()
	}

	// Extract image-sized ROI from the real part (correlation result)
	roiRect := image.Rect(0, 0, p.imgSize.X, p.imgSize.Y)
	roi := p.scratchRealPC.Region(roiRect)
	roi.CopyTo(&p.scratchCorrelation)
	roi.Close()

	// Step 4: Find peak location using MinMaxLoc
	_, maxVal, _, maxLoc := gocv.MinMaxLoc(p.scratchCorrelation)

	// Step 5: Calculate displacement with wraparound handling
	// If peak is in the second half of the image, subtract image size to get negative displacement
	dx := maxLoc.X
	dy := maxLoc.Y

	// Handle wraparound: if displacement > size/2, subtract size to get negative shift
	if dx > p.imgSize.X/2 {
		dx -= p.imgSize.X
	}
	if dy > p.imgSize.Y/2 {
		dy -= p.imgSize.Y
	}

	return dx, dy, float64(maxVal)
}

// Reset resets the processor state (clears previous frame).
func (p *Processor) Reset() {
	p.skipCount = 0
	// Clear DFT buffers
	p.curDFT.SetTo(gocv.NewScalar(0, 0, 0, 0))
	p.lastDFT.SetTo(gocv.NewScalar(0, 0, 0, 0))
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	p.curDFT.Close()
	p.lastDFT.Close()
	p.scratchGray.Close()
	p.scratchRealInput.Close()
	p.scratchImagInput.Close()
	p.scratchComplexIn.Close()
	p.scratchComplexR.Close()
	p.scratchComplexPC.Close()
	p.scratchRealPC.Close()
	p.scratchImagPC.Close()
	p.scratchCorrelation.Close()
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[visiontypes.Displacement, types.Tensor] = (*Processor)(nil)
