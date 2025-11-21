package median

// Median filter for noise reduction while preserving edges

import (
	"sort"

	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Filter implements a median filter.
// This filter preserves edges while reducing noise.
type Filter struct {
	windowSize  int
	buffer      vec.Vector
	index       int
	initialized bool
}

// New creates a new median filter.
// windowSize must be odd (will be adjusted if even).
func New(windowSize int) *Filter {
	if windowSize <= 0 {
		panic("Median window size must be positive")
	}

	// Ensure window size is odd
	if windowSize%2 == 0 {
		windowSize++
	}

	return &Filter{
		windowSize:  windowSize,
		buffer:      vec.New(windowSize),
		index:       0,
		initialized: false,
	}
}

// Reset resets the filter state.
func (mf *Filter) Reset() {
	for i := range mf.buffer {
		mf.buffer[i] = 0
	}
	mf.index = 0
	mf.initialized = false
}

// Process processes a single sample and returns the filtered value.
func (mf *Filter) Process(sample float32) float32 {
	mf.buffer[mf.index] = sample
	mf.index = (mf.index + 1) % mf.windowSize
	mf.initialized = true

	// Need at least half the window size for valid output
	halfWindow := mf.windowSize / 2
	if !mf.initialized || mf.index < halfWindow {
		return sample
	}

	// Extract window values for sorting
	window := make([]float32, mf.windowSize)
	for i := 0; i < mf.windowSize; i++ {
		bufIdx := (mf.index - halfWindow + i) % mf.windowSize
		if bufIdx < 0 {
			bufIdx += mf.windowSize
		}
		window[i] = mf.buffer[bufIdx]
	}

	// Sort and return median
	sorted := make([]float32, len(window))
	copy(sorted, window)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	return sorted[halfWindow]
}

// ProcessBuffer processes an entire buffer of samples.
// Returns a new vector with filtered values.
func (mf *Filter) ProcessBuffer(input vecTypes.Vector) vec.Vector {
	inputVec := input.View().(vec.Vector)
	output := vec.New(len(inputVec))

	// Reset filter state
	mf.Reset()

	// Process each sample
	for i := range inputVec {
		output[i] = mf.Process(inputVec[i])
	}

	return output
}

// Median applies median filtering to a signal vector.
// This is a convenience function that creates a filter and processes the entire signal.
func Median(signal vecTypes.Vector, windowSize int) vec.Vector {
	filter := New(windowSize)
	return filter.ProcessBuffer(signal)
}
