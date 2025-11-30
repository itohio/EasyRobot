package features

import (
	"fmt"
	"log/slog"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	cv "gocv.io/x/gocv"
)

// ExampleUsage demonstrates how to use the feature extraction processor
// with the chain package using existing buffer.Ring.
func ExampleUsage() {
	// Create processor
	extractor, err := New()
	if err != nil {
		slog.Error("Failed to create feature extractor", "err", err)
		return
	}
	defer extractor.Close()

	// Create input buffer using buffer.Ring (generic, works with any type including Tensor)
	// In real usage, this would come from a camera or video source
	img := cv.IMRead("test.jpg", cv.IMReadColor)
	if img.Empty() {
		slog.Error("Failed to load image")
		return
	}
	defer img.Close()

	inputTensor := tensorgocv.WithAdoptedMat(img)
	inputBuf := buffer.NewRing(make([]types.Tensor, 0, 1))
	_, _ = inputBuf.Write([]types.Tensor{inputTensor})

	// Create output buffer
	outputBuf := buffer.NewRing(make([]types.Tensor, 0, 10))

	// Process
	written := extractor.Process(outputBuf, inputBuf)
	fmt.Printf("Processed %d tensor(s)\n", written)

	// Read results
	outputTensors := make([]types.Tensor, 1)
	n, err := outputBuf.Read(outputTensors)
	if err != nil {
		slog.Error("Failed to read output", "err", err)
		return
	}

	if n > 0 {
		// Get descriptor tensor
		descTensor := outputTensors[0]
		gocvDesc, ok := descTensor.(tensorgocv.Tensor)
		if ok {
			descMat, err := gocvDesc.Mat()
			if err == nil {
				fmt.Printf("Descriptor matrix: %dx%d\n", descMat.Rows(), descMat.Cols())
				// Note: Mat is owned by tensor, don't close here
			}
		}
	}
}

// ExampleChain demonstrates chaining multiple vision processors.
func ExampleChain() {
	// Create processors
	extractor, _ := New()
	defer extractor.Close()
	// matcher := matching.New() // Would need to implement
	// odometry := odometry.New()  // Would need to implement

	// Create buffer factory using buffer.Ring (generic buffer)
	factory := func(cap int) *buffer.Ring[types.Tensor] {
		return buffer.NewRing[types.Tensor](make([]types.Tensor, 0, cap))
	}

	// Create chain (all processors must have same input/output type)
	// For now, just feature extraction
	pipeline := chain.New(
		factory,
		extractor,
		// matcher,  // Would chain these
		// odometry,
	)

	// Create input/output buffers
	inputTensor := tensorgocv.WithAdoptedMat(cv.IMRead("test.jpg", cv.IMReadColor))
	inputBuf := buffer.NewRing(make([]types.Tensor, 0, 1))
	_, _ = inputBuf.Write([]types.Tensor{inputTensor})

	outputBuf := buffer.NewRing(make([]types.Tensor, 0, 10))

	// Process through chain
	written := pipeline.Process(outputBuf, inputBuf)
	fmt.Printf("Chain processed %d tensor(s)\n", written)
}

// ExampleZeroCopy demonstrates zero-copy in-place processing using ViewWriter/ViewReader.
// This is useful when you want to avoid copying tensors during processing.
func ExampleZeroCopy() {
	// Allocate ring buffer with capacity 1 for in-place processing
	tensors := make([]types.Tensor, 1)
	ring := buffer.NewRing(tensors)

	// Calculate input tensor in-place (access backing array directly via views)
	writableViews := ring.WriteView()
	if len(writableViews) > 0 && len(writableViews[0]) > 0 {
		// Create or modify tensor directly in backing array (zero-copy)
		img := cv.IMRead("test.jpg", cv.IMReadColor)
		writableViews[0][0] = tensorgocv.WithAdoptedMat(img)

		// Commit write (advance write pointer - no copy)
		_ = ring.CommitWrite(1)
	}

	// Extractor can access backing array directly for zero-copy processing
	extractor, _ := New()
	defer extractor.Close()

	// Create output buffer
	outputBuf := buffer.NewRing(make([]types.Tensor, 0, 10))

	// Process using standard Read() interface (which internally uses ReadView/CommitRead)
	// For zero-copy optimization, processors can use ReadView() + CommitRead() directly
	written := extractor.Process(outputBuf, ring)
	fmt.Printf("Processed %d tensor(s) with zero-copy\n", written)
}
