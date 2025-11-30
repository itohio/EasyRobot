# Vision Algorithms as Chain Processors

## Overview

This document describes how vision algorithms can implement the `chain.Processor` interface using Tensor types as input/output. This enables composable vision processing pipelines using the `x/chain` package.

## Design

### Using Existing Generic Buffers

The `x/buffer` package already provides generic buffers (Ring, Linear, etc.) that work with any type, including `types.Tensor`. No special TensorBuffer is needed:

```go
import "github.com/itohio/EasyRobot/x/buffer"

// Create buffer for tensors using generic Ring buffer
tensorBuf := buffer.NewRing(make([]types.Tensor, 0, 10))

// Write tensors
_, _ = tensorBuf.Write([]types.Tensor{inputTensor})

// Read tensors
outputTensors := make([]types.Tensor, 1)
n, err := tensorBuf.Read(outputTensors)
```

### Vision Processor Interface

Vision processors implement `chain.Processor[types.Tensor, types.Tensor]`:

```go
type Processor interface {
    chain.Processor[types.Tensor, types.Tensor]
}
```

### Example: Feature Extraction Processor

```go
type Processor struct {
    detector gocv.ORB
    // Preallocated scratch space to avoid allocations in Process()
    scratchMask cv.Mat
}

func New() (*Processor, error) {
    return &Processor{
        detector: gocv.NewORB(),
        scratchMask: cv.NewMat(),
    }, nil
}

func (p *Processor) Process(
    dst buffer.Writer[types.Tensor], 
    src buffer.Reader[types.Tensor],
) int {
    if dst == nil {
        // Size estimation: 1 tensor in -> 1 tensor out (with features)
        return src.Len()
    }
    
    // Preallocated slice to avoid allocation
    var inputTensors [1]types.Tensor
    n, err := src.Read(inputTensors[:])
    if err != nil || n == 0 {
        return 0
    }
    
    // Process: extract features and create output tensor
    // Use preallocated scratch mask
    keypoints, descriptors := p.detector.DetectAndCompute(mat, p.scratchMask)
    
    // Create output tensor (adopts Mat ownership)
    outputTensor := tensorgocv.WithAdoptedMat(descriptors)
    
    // Write output tensor (reuse preallocated slice)
    outputTensors := []types.Tensor{outputTensor}
    written, _ := dst.Write(outputTensors)
    return written
}

func (p *Processor) Reset() {
    // Clear scratch space if needed
}
```

**Important Notes**:
- All allocations must be preallocated in struct fields
- Use preallocated scratch buffers (cv.Mat, slices, etc.)
- Tensors are NOT guaranteed to be zero-initialized
- Methods called repeatedly (Process) MUST avoid allocations

## Usage Examples

### Single Processor

```go
extractor, _ := features.New()
defer extractor.Close()

// Use generic Ring buffer
inputBuf := buffer.NewRing(make([]types.Tensor, 0, 1))
_, _ = inputBuf.Write([]types.Tensor{inputTensor})

outputBuf := buffer.NewRing(make([]types.Tensor, 0, 10))

written := extractor.Process(outputBuf, inputBuf)
```

### Chained Processors

```go
// Chain: Feature Extraction -> Feature Matching -> Pose Estimation
factory := func(cap int) *buffer.Ring[types.Tensor] {
    return buffer.NewRing[types.Tensor](make([]types.Tensor, 0, cap))
}

extractor, _ := features.New()
// matcher, _ := matching.New()
// odometry, _ := odometry.New()

pipeline := chain.New(
    factory,
    extractor,
    // matcher,
    // odometry,
)

// Process through chain
written := pipeline.Process(outputBuf, inputBuf)
```

### Adapter for Type Conversion

```go
// Convert from Frame (with tensors) to Tensor stream
frameToTensor := NewFrameToTensorAdapter()
tensorToFrame := NewTensorToFrameAdapter()

adapter := chain.NewAdapter(
    factory,
    tensorToFrame,  // Tensor -> Frame
    frameToTensor,  // Frame -> Tensor
)
```

## Implementation Plan

1. **Use Existing Generic Buffers**
   - Use `buffer.Ring[types.Tensor]`, `buffer.Linear[types.Tensor]`, etc.
   - No need for custom TensorBuffer

2. **Refactor Feature Extraction** (`x/vision/extract/features`)
   - Implement `chain.Processor[types.Tensor, types.Tensor]`
   - Process one tensor at a time

3. **Implement Other Processors**
   - Feature Matching
   - Stereo Rectification
   - Stereo Matching
   - Visual Odometry

4. **Create Chain Utilities**
   - Helper functions for common chains
   - Factory functions for buffer creation

## Benefits

1. **Composability**: Chain multiple vision algorithms together
2. **Reusability**: Use same buffer/chain infrastructure as DSP
3. **Type Safety**: Generic types ensure correct tensor flow
4. **Streaming**: Process tensors one at a time (frame-by-frame)
5. **Testing**: Easy to test individual processors

## Considerations

1. **No Allocations in Process()**: All allocations must be preallocated in struct fields
   - Preallocate cv.Mat objects for scratch space
   - Preallocate slices for temporary data
   - Use destination pattern where possible

2. **Tensor Lifecycle**: Need to handle tensor Release() properly
   - Use `tensorgocv.WithAdoptedMat()` to transfer Mat ownership to tensor
   - Destinations with `WithRelease()` option will call Release() automatically

3. **Tensor Initialization**: Tensors are NOT guaranteed to be zero-initialized
   - Always initialize tensor data explicitly
   - Don't assume zero values

4. **Memory**: Buffers hold references to tensors (use SharedTensor if needed)
   - Use `tensor.WithRefcount()` for fan-out scenarios

5. **Size Estimation**: Some processors may change tensor dimensions
   - Return accurate size estimates when `dst` is nil

6. **Error Handling**: Processors should handle errors gracefully
   - Return 0 on error (no output produced)
   - Log errors appropriately

7. **Zero-Copy Processing (Advanced)**: For in-place processing scenarios
   - Use `ViewReader` and `ViewWriter` interfaces for zero-copy operations
   - `ReadView()` returns list of slices pointing to readable portions of buffer
   - `WriteView()` returns list of slices pointing to writable portions of buffer
   - `CommitRead(n)` and `CommitWrite(n)` advance pointers after processing
   - Processors can operate on views for batch processing (e.g., odometry with history)
   - Example: Allocate tensor array of size 1, calculate in-place, use WriteView + CommitWrite
   - Example: Processor accesses Ring backing array via ReadView and uses CommitRead
   - Useful for processors that need access to multiple frames (history) for batch normalization
   - Note: Read() and Write() internally use ReadView/WriteView and CommitRead/CommitWrite

