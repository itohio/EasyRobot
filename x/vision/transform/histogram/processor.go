package histogram

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	visiontypes "github.com/itohio/EasyRobot/x/vision/types"
	"gocv.io/x/gocv"
	"image"
)

// Processor implements chain.Processor[visiontypes.ImageWithMask, types.Tensor] for histogram backprojection.
// It creates a mask by backprojecting a histogram onto an image.
//
// Allocations: This processor preallocates Mat objects to avoid allocations
// in the Process() callstack. Histograms are provided externally.
type Processor struct {
	// Histogram(s) to backproject - can be single or multiple
	histograms []gocv.Mat

	// Channels to use for backprojection (e.g., [0, 1] for HSV H and S channels)
	channels []int

	// Ranges for each channel (e.g., [0, 180, 0, 256] for HSV)
	ranges []float64

	// Color range filtering (optional)
	rangeMin []float32
	rangeMax []float32

	// Whether to combine multiple masks into one
	combineMasks bool

	// Whether to invert the resulting mask
	invert bool

	// Preallocated scratch space
	scratchHSV      gocv.Mat
	scratchMask     gocv.Mat
	scratchBackProj gocv.Mat
	discKernel     gocv.Mat // Morphological kernel for filtering
}

// Option configures a histogram backprojection processor.
type Option func(*Processor)

// WithHistogram adds a histogram to backproject.
// Can be called multiple times to add multiple histograms.
func WithHistogram(hist gocv.Mat) Option {
	return func(p *Processor) {
		p.histograms = append(p.histograms, hist)
	}
}

// WithChannels sets the channels to use for backprojection.
func WithChannels(channels []int) Option {
	return func(p *Processor) {
		p.channels = channels
	}
}

// WithRanges sets the value ranges for each channel.
func WithRanges(ranges []float64) Option {
	return func(p *Processor) {
		p.ranges = ranges
	}
}

// WithColorRange sets the color range filtering (for HSV images).
func WithColorRange(min, max []float32) Option {
	return func(p *Processor) {
		p.rangeMin = min
		p.rangeMax = max
	}
}

// WithCombineMasks sets whether to combine multiple masks into one.
func WithCombineMasks(combine bool) Option {
	return func(p *Processor) {
		p.combineMasks = combine
	}
}

// WithInvert sets whether to invert the resulting mask.
func WithInvert(invert bool) Option {
	return func(p *Processor) {
		p.invert = invert
	}
}

// New creates a new histogram backprojection processor.
// Default channels are [0, 1] (H and S from HSV).
// Default ranges are [0, 180, 0, 256] for HSV.
func New(opts ...Option) *Processor {
	p := &Processor{
		channels:     []int{0, 1}, // H and S channels from HSV
		ranges:       []float64{0, 180, 0, 256},
		combineMasks: false,
		invert:       false,
		discKernel:   gocv.GetStructuringElement(gocv.MorphEllipse, image.Pt(10, 10)),
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// Process implements chain.Processor[visiontypes.ImageWithMask, types.Tensor].
// Reads image tensor from src, performs histogram backprojection, and writes mask to dst.
//
// Input: Image tensor (gocv.Mat-backed, typically BGR or HSV)
// Output: ImageWithMask containing the original image and the mask
func (p *Processor) Process(dst buffer.Writer[visiontypes.ImageWithMask], src buffer.Reader[types.Tensor]) int {
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

	// Convert to HSV if needed (assuming input is BGR)
	// TODO: Make color space configurable
	gocv.CvtColor(mat, &p.scratchHSV, gocv.ColorBGRToHSV)

	// Process each histogram
	var masks []types.Tensor
	for _, hist := range p.histograms {
		if hist.Empty() {
			continue
		}

		// Calculate backprojection
		gocv.CalcBackProject([]gocv.Mat{p.scratchHSV}, p.channels, hist, &p.scratchBackProj, p.ranges, false)

		// Filter with morphological kernel
		gocv.Filter2D(p.scratchBackProj, &p.scratchMask, -1, p.discKernel, image.Pt(-1, -1), 0, gocv.BorderDefault)

		// Blur
		gocv.Blur(p.scratchMask, &p.scratchMask, image.Pt(50, 50))

		// Threshold
		gocv.Threshold(p.scratchMask, &p.scratchMask, 50, 255, gocv.ThresholdBinary)

		// Invert if requested
		if p.invert {
			gocv.BitwiseNot(p.scratchMask, &p.scratchMask)
		}

		// Create tensor from mask
		maskMat := p.scratchMask.Clone()
		maskTensor, err := tensorgocv.FromMat(maskMat, tensorgocv.WithAdoptedMat())
		if err != nil {
			continue
		}
		masks = append(masks, maskTensor)
	}

	if len(masks) == 0 {
		return 0
	}

	// Combine masks if requested
	var outputMask types.Tensor
	if p.combineMasks && len(masks) > 1 {
		// Combine all masks using bitwise OR
		combinedMask := masks[0]
		combinedGocv, ok := combinedMask.(tensorgocv.Tensor)
		if !ok {
			return 0
		}
		combinedMat, err := combinedGocv.Mat()
		if err != nil {
			return 0
		}

		for i := 1; i < len(masks); i++ {
			maskGocv, ok := masks[i].(tensorgocv.Tensor)
			if !ok {
				continue
			}
			maskMat, err := maskGocv.Mat()
			if err != nil {
				continue
			}
			gocv.BitwiseOr(combinedMat, maskMat, &combinedMat)
		}
		outputMask = combinedMask
	} else {
		outputMask = masks[0] // Use first mask if not combining
	}

	// Create output with image and mask
	output := visiontypes.ImageWithMask{
		Image: inputTensor, // Keep reference to original image
		Mask:  outputMask,
	}

	outputs := []visiontypes.ImageWithMask{output}
	written, _ := dst.Write(outputs)
	return written
}

// Reset resets the processor state.
func (p *Processor) Reset() {
	// Clear scratch space if needed
	if !p.scratchHSV.Empty() {
		p.scratchHSV.Close()
		p.scratchHSV = gocv.NewMat()
	}
	if !p.scratchMask.Empty() {
		p.scratchMask.Close()
		p.scratchMask = gocv.NewMat()
	}
	if !p.scratchBackProj.Empty() {
		p.scratchBackProj.Close()
		p.scratchBackProj = gocv.NewMat()
	}
}

// Close releases all resources held by the processor.
func (p *Processor) Close() {
	if !p.scratchHSV.Empty() {
		p.scratchHSV.Close()
	}
	if !p.scratchMask.Empty() {
		p.scratchMask.Close()
	}
	if !p.scratchBackProj.Empty() {
		p.scratchBackProj.Close()
	}
	if !p.discKernel.Empty() {
		p.discKernel.Close()
	}
	// Note: We don't close histograms as they are owned by the caller
}

// CalculateHistogram calculates a histogram from an image using a mask if provided.
// This is a helper function similar to the Python version's static method.
// Returns a normalized histogram Mat. The caller is responsible for closing the returned Mat.
func CalculateHistogram(img, mask gocv.Mat, channels []int, bins []int, ranges []float64, minRange, maxRange []float32) (gocv.Mat, error) {
	// Convert to HSV
	hsv := gocv.NewMat()
	defer hsv.Close()
	gocv.CvtColor(img, &hsv, gocv.ColorBGRToHSV)

	// Apply color range filtering
	filteredMask := gocv.NewMat()
	defer filteredMask.Close()
	if len(minRange) > 0 && len(maxRange) > 0 {
		gocv.InRange(hsv, gocv.NewScalar(minRange[0], minRange[1], minRange[2]), gocv.NewScalar(maxRange[0], maxRange[1], maxRange[2]), &filteredMask)
		if !mask.Empty() {
			gocv.BitwiseAnd(filteredMask, mask, &filteredMask)
		}
	} else {
		filteredMask = mask
	}

	// Calculate histogram
	hist := gocv.NewMat()
	gocv.CalcHist([]gocv.Mat{hsv}, channels, filteredMask, &hist, bins, ranges, false)

	// Normalize histogram
	normalized := gocv.NewMat()
	gocv.Normalize(hist, &normalized, 0, 256, gocv.NormMinMax)

	return normalized, nil
}

// Ensure Processor implements chain.Processor
var _ chain.Processor[visiontypes.ImageWithMask, types.Tensor] = (*Processor)(nil)

