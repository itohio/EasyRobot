# Available Utilities Inventory

## Buffer Pooling

### Generic Pool (`x/math/primitive/generics/helpers/pool.go`)
```go
type Pool[T any] struct {
    // Tiered buffer pools for efficient memory reuse
}

func (p *Pool[T]) Get(n int) []T  // Get buffer of length n
func (p *Pool[T]) Put(buf []T)     // Return buffer to pool
```

**Usage:**
- Create pools for byte buffers: `var bytePool = &helpers.Pool[byte]{}`
- Create pools for typed buffers: `var int16Pool = &helpers.Pool[int16]{}`

### Tensor Data Pooling (`x/math/tensor/types/dtype.go`)
```go
func MakeTensorData(dt DataType, size int) any
func ReleaseTensorData(data any)
func CloneTensorDataTo(dst DataType, data any) any
```

**Supported types:**
- `FP32` → `[]float32` (pooled)
- `INT16` → `[]int16` (pooled)
- `INT32` → `[]int32` (pooled)
- `INT8` → `[]int8` (pooled)
- `UINT8` → `[]uint8` (not pooled, but available)

**Usage:**
```go
data := types.MakeTensorData(types.INT16, 1000) // Returns []int16 from pool
// ... use data ...
types.ReleaseTensorData(data) // Returns to pool
```

## Statistics Tracking

### Atomic Counters (`sync/atomic`)
```go
import "sync/atomic"

var frames int64
atomic.AddInt64(&frames, 1)
atomic.LoadInt64(&frames)
```

**Pattern from SPEC.md:**
```go
type stats struct {
    frames        int64  // Total frames processed
    capturedBuffers int64 // Buffer counts
    droppedBuffers  int64 // Overflow/underflow detection
    inBytes        int64 // Input byte statistics
    outBytes       int64 // Output byte statistics
}

func (s *stats) Health() map[string]int64 {
    return map[string]int64{
        "frames": atomic.LoadInt64(&s.frames),
        "droppedBuffers": atomic.LoadInt64(&s.droppedBuffers),
        // ...
    }
}
```

## Tensor Operations

### Available Operations (`x/math/tensor`)
- `Reshape(dst Tensor, newShape Shape) Tensor` - Reshape tensor
- `Slice(dst Tensor, dim int, start int, length int) Tensor` - Extract slice
- `Clone()` - Deep copy
- `Copy(src Tensor) Tensor` - Copy with type conversion
- `Fill(dst Tensor, value float64) Tensor` - Fill with value
- `Pad(dst Tensor, padding []int, value float64) Tensor` - Pad tensor

### Channel Concatenation Strategy
**No explicit Concat function**, so we'll manually combine channels:
1. Read each WAV file into separate tensors with shape `[frames, channels]`
2. Determine max frames (pad shorter files with zeros)
3. Create combined tensor with shape `[maxFrames, totalChannels]`
4. Copy data from each source tensor to appropriate channels

## Audio Format Conversion

### PCM to Tensor (Already Implemented)
- `fillTensorFromPCM()` in `unmarshaller.go` handles conversion
- Supports: INT16, INT32, FP32, UINT8

### Tensor to PCM (Already Implemented)
- `writeInt16PCM()`, `writeInt32PCM()`, `writeFloat32PCM()`, `writeUint8PCM()` in `marshaller.go`
- Handles format conversion when needed

## WAV File Operations

### Reading (`github.com/go-audio/wav`)
```go
decoder := wav.NewDecoder(file)
decoder.ReadInfo()
format := decoder.Format()
decoder.PCMBuffer(buf)
```

### Writing (`github.com/go-audio/wav`)
```go
encoder := wav.NewEncoder(file, sampleRate, bitDepth, channels, 1)
encoder.Write(buf)
```

## Implementation Plan

1. **Buffer Pooling**
   - Create `bytePool` for PCM byte buffers
   - Use `MakeTensorData`/`ReleaseTensorData` for tensor data
   - Pool sizes: frame-based (e.g., 1024 frames per buffer)

2. **Statistics Tracking**
   - Add `stats` struct with atomic counters
   - Track: frames, bytes, dropped buffers
   - Implement `Health()` method

3. **WAV Concatenation**
   - Read all files, verify same sample rate
   - Determine max frames, pad shorter files
   - Combine channels manually using tensor data access
   - Use `MakeTensorData` for combined tensor

