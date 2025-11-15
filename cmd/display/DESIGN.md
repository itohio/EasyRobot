# Image/Video/Camera Display with DNDM Support - Design Document

## Overview

This document describes the design for an image/video/camera reader and displayer that supports:
- Reading from images, video files, and cameras using GoCV unmarshallers
- Reading from DNDM interest channels (consumer mode)
- Publishing to DNDM intent channels (producer mode)
- Displaying frames using GoCV marshaller
- Unified command-line interface with mutually exclusive input options

**Note:** This utility provides reusable source and destination packages that can be used by `calib_mono` and `calib_stereo` utilities. The architecture allows custom processing code to be inserted between source and destination.

## Goals

1. Replace/enhance the existing `cmd/display` implementation with a more flexible architecture
2. Support multiple input sources: images, videos, cameras, DNDM interest/intent
3. Support multiple instances of the same input type (e.g., multiple image paths)
4. Use GoCV marshallers/unmarshallers for frame encoding/decoding
5. Integrate with DNDM for distributed communication
6. Maintain compatibility with existing display features (FPS, window handling, keyboard controls)

## Architecture

### Command-Line Interface

```bash
display --images <path1> [--images <path2> ...]
display --video <path1> [--video <path2> ...]
display --camera <device-id1> [--camera <device-id2> ...]
display --interest <route1> [--interest <route2> ...]
display --intent <route1> [--intent <route2> ...]
display --output <video-file>  # Output video file (e.g., output.mp4)
display --no-display  # Omit display (useful for --intent/--output only mode)
```

**Constraints:**
- Only one input type can be specified per run
- Multiple instances of the same type are allowed (creates a list)
- Each input type maps to a different unmarshaller/source handler
- **Display is always enabled by default** (unless `--no-display` is provided)
- **Destinations can be combined**: display, DNDM intent, and video file output can all be active simultaneously
- Use `--no-display` to omit display when only publishing via DNDM intent or recording to file

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.go                               │
│  1. Create source → registers flags                          │
│  2. Create destination(s) → register flags                   │
│  3. flag.Parse()                                             │
│  4. source.Start(ctx)                                        │
│  5. destination.Start(ctx)                                   │
│  6. Loop:                                                    │
│     - frame := source.ReadFrame()                            │
│     - [custom processing here] ← calib_mono/calib_stereo    │
│     - destination.AddFrame(frame)                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              cmd/display/source                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ ImageSource  │  │ VideoSource  │  │CameraSource  │      │
│  │ - Registers  │  │ - Registers  │  │ - Registers  │      │
│  │   --images   │  │   --video    │  │   --camera   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │InterestSource│  │              │                        │
│  │ - Registers  │  │              │                        │
│  │   --interest │  │              │                        │
│  └──────────────┘  └──────────────┘                        │
│                                                              │
│  Interface:                                                  │
│  - RegisterFlags()                                           │
│  - Start(ctx) error                                          │
│  - ReadFrame() (types.Frame, error)                         │
│  - Close() error                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ (fan-out)
                    ┌─────────┴─────────┐
                    │                   │
        ┌───────────▼───────────┐  ┌────▼──────────────┐
        │   Destination         │  │  Destination      │
        │   (Fan-out)           │  │  (Fan-out)        │
        └───────────────────────┘  └───────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Display          │  │ DNDM Intent      │  │ Video File       │
│ Destination      │  │ Destination      │  │ Destination      │
│ - Registers      │  │ - Registers      │  │ - Registers      │
│   --no-display   │  │   --intent       │  │   --output       │
│                  │  │                  │  │                  │
│ - GoCV window    │  │ - DNDM client    │  │ - GoCV writer    │
│ - Keyboard       │  │ - Intent routes  │  │ - Video encoder  │
│   handling       │  │ - Proto conv     │  │ - File output    │
└──────────────────┘  └──────────────────┘  └──────────────────┘

Interface:
- RegisterFlags()
- Start(ctx) error
- AddFrame(frame types.Frame) error
- Close() error
```

**Key Points:**
- **Source pattern**: Create source, it registers flags, then call Start() and ReadFrame()
- **Destination pattern**: Create destinations, they register flags, then call Start() and AddFrame()
- **Fan-out architecture**: Source → multiple destinations (display, intent, video file)
- **Custom processing**: Frames can be processed between source.ReadFrame() and destination.AddFrame()
- **Display is always enabled by default** (unless `--no-display` flag is provided)
- **All destinations run in parallel**: Display, DNDM intent, and video file can all be active simultaneously

## Input Source Types

### 1. Images (`--images`)

**Implementation:**
- Use `x/marshaller/gocv` unmarshaller with `WithPaths(paths...)` option
- Unmarshal to `*types.FrameStream`
- Each path can be:
  - Single image file (jpg, png, bmp, jpeg)
  - Directory containing images (enumerated lexicographically)
- Multiple `--images` flags create a list that gets combined

**Example:**
```go
unmarshaller := gocv.NewUnmarshaller(
    gocv.WithPaths(imagePaths...),
    gocv.WithContext(ctx),
)
var stream types.FrameStream
err := unmarshaller.Unmarshal(nil, &stream)
```

### 2. Video Files (`--video`)

**Implementation:**
- Use `x/marshaller/gocv` unmarshaller with video file paths
- Similar to images but uses `VideoCaptureFile`
- Multiple `--video` flags create synchronized video streams

**Example:**
```go
unmarshaller := gocv.NewUnmarshaller(
    gocv.WithPaths(videoPaths...),
    gocv.WithContext(ctx),
)
```

### 3. Cameras (`--camera`)

**Implementation:**
- Use `x/marshaller/gocv` unmarshaller with device specs
- Each `--camera <id>` flag creates a device source
- Optionally support `--width` and `--height` for resolution
- Multiple cameras can be synchronized

**Example:**
```go
sources := []sourceSpec{}
for _, deviceID := range deviceIDs {
    sources = append(sources, deviceSpec{
        ID: deviceID,
        Width: width,
        Height: height,
    })
}
// Configure unmarshaller with device sources
```

### 4. DNDM Interest (`--interest`)

**Implementation:**
- Connect to DNDM network using existing DNDM client
- Create Interest for each route specified
- Listen for incoming proto messages containing Tensor data
- Convert proto Tensor → gocv.Mat → types.Tensor → types.Frame
- Routes can subscribe to different tensor/image streams

**Proto Structure Needed:**
```protobuf
// In dndm/proto/types/core/tensor.proto (NEW)
syntax = "proto3";
package types.core;

import "easyrobot/types/core/types.proto"; // Or define own

// Frame message containing tensor data
message Frame {
  easyrobot.core.types.Tensor tensor = 1;  // Use EasyRobot tensor proto
  int64 timestamp = 2;
  int64 index = 3;
  map<string, string> metadata = 4;
}
```

**Example:**
```go
dndmClient := dndm.NewClient(...)
for _, route := range interestRoutes {
    interest, err := dndmClient.NewInterest(ctx, route)
    // Listen on interest channel for messages
    // Convert proto Frame → types.Frame
}
```

### 5. DNDM Intent Destination (`--intent`)

**Implementation:**
- Created as a destination: `destination.NewIntent(...)`
- Registers `--intent <route>` flags (can be repeated)
- Connects to DNDM network on Start()
- Creates Intent for each route specified
- **Runs in parallel with display and video file destinations**
- Converts types.Frame → proto Frame → proto.Message
- Publishes when Interest matches are found

**Important:** 
- When `--intent` is provided, the DNDM intent destination is **added in addition** to display
- Use `--no-display` to omit display when only publishing via DNDM intent
- All destinations (display, intent, video file) can be active simultaneously

**Example:**
```go
// Create DNDM intent destination
intentDest := destination.NewIntent(
    destination.WithDNDMClient(dndmClient),
)
// Registers --intent flags

// Later in main:
intentDest.Start(ctx)

// In processing loop:
for {
    frame := source.ReadFrame()
    // ... custom processing ...
    intentDest.AddFrame(frame)  // Sends to all registered intents
}
```

## Frame Processing Flow

### Frame Flow

1. **Source → Frame**: Each source type produces `types.Frame`
   - Contains `[]types.Tensor` (at least one image tensor)
   - Contains metadata map (path, timestamp, index, etc.)
   - Use `x/marshaller/gocv` unmarshaller internally
   - Source provides `ReadFrame()` method that returns frames one by one

2. **Frame Processing**: Direct frame-by-frame processing
   - Main loop reads frames: `frame := source.ReadFrame()`
   - **Custom processing can be inserted here** (e.g., calibration, filtering)
   - Frame is passed to destinations: `destination.AddFrame(frame)`

3. **Fan-out to Destinations**: Frame is distributed to all enabled destinations
   - **Display Destination**: Converts Frame → gocv.Mat → Window display
   - **DNDM Intent Destination**: Converts Frame → proto → DNDM intent.Send()
   - **Video File Destination**: Converts Frame → gocv.Mat → Video encoder → File
   - All destinations run in parallel from the same frame

### Usage Pattern

```go
// 1. Create source (registers --images, --video, --camera, or --interest flags)
source := source.New(...)

// 2. Create destinations (register --no-display, --intent, --output flags)
displayDest := destination.NewDisplay(...)
intentDest := destination.NewIntent(...)
videoDest := destination.NewVideo(...)

// 3. Parse flags
flag.Parse()

// 4. Start source and destinations
ctx := context.Background()
if err := source.Start(ctx); err != nil { ... }
if err := displayDest.Start(ctx); err != nil { ... }
if err := intentDest.Start(ctx); err != nil { ... }
if err := videoDest.Start(ctx); err != nil { ... }

// 5. Process frames
for {
    frame, err := source.ReadFrame()
    if err != nil { break }
    
    // Custom processing can be inserted here
    // e.g., calibration, filtering, detection, etc.
    processedFrame := myCustomProcessing(frame)
    
    // Send to all destinations
    if err := displayDest.AddFrame(processedFrame); err != nil { ... }
    if err := intentDest.AddFrame(processedFrame); err != nil { ... }
    if err := videoDest.AddFrame(processedFrame); err != nil { ... }
}

// 6. Cleanup
source.Close()
displayDest.Close()
intentDest.Close()
videoDest.Close()
```

### Destination Options

#### Display Destination

- **Display is always enabled by default** (unless `--no-display` flag is provided)
- Implemented in `cmd/display/destination/display.go`
- Uses `x/marshaller/gocv` display marshaller internally
- Converts `types.Frame` → `gocv.Mat` → Window display
- Support keyboard controls:
  - ESC to quit (triggers context cancellation)
  - Space to pause (future)
  - Arrow keys for navigation (future)
- Multiple windows: one window per frame (future enhancement)

#### DNDM Intent Destination

- Enabled when `--intent <route>` flags are provided
- Can have multiple routes (one intent per route)
- Converts `types.Frame` → proto Frame → DNDM intent.Send()
- Waits for Interest matches before sending
- Runs in parallel with display and video file destinations

#### Video File Destination

- Enabled when `--output <file>` flag is provided
- Uses GoCV video writer to encode frames to video file
- Supports common video formats (MP4, AVI, MOV, etc.)
- Converts `types.Frame` → `gocv.Mat` → Video encoder → File
- Runs in parallel with display and DNDM intent destinations

## DNDM Integration Details

### Proto Definitions

**Location:** `dndm/proto/types/core/tensor.proto` (NEW)

**Options:**
1. Reuse `easyrobot.core.types.Tensor` proto (import from EasyRobot)
2. Define own Tensor proto in DNDM (for decoupling)

**Recommendation:** Option 2 - Define own Tensor proto in DNDM for decoupling, but keep compatible structure.

**Frame Message:**
```protobuf
syntax = "proto3";
package types.core;

// Frame contains image/tensor data with metadata
message Frame {
  Tensor tensor = 1;           // Image tensor data
  int64 timestamp = 2;         // Frame timestamp (nanoseconds)
  int64 index = 3;             // Frame index
  map<string, string> metadata = 4;  // Additional metadata
}

// Tensor represents multi-dimensional array (similar to EasyRobot)
message Tensor {
  string dtype = 1;            // "uint8", "float32", etc.
  repeated int32 shape = 2;    // [height, width, channels] for images
  repeated float data_f32 = 3; // float32 data
  repeated double data_f64 = 4;// float64 data
  repeated int32 data_i32 = 5; // int32 data
  repeated int64 data_i64 = 6; // int64 data
  bytes data_bytes = 7;        // Raw bytes (e.g., encoded image)
}
```

### Conversion Functions

**Proto Tensor → GoCV Mat:**
```go
func protoTensorToMat(pb *core.Tensor) (cv.Mat, error) {
    // Decode based on dtype and shape
    // Handle uint8 for images (most common)
    // Convert to cv.Mat
}
```

**GoCV Mat → Proto Tensor:**
```go
func matToProtoTensor(mat cv.Mat) (*core.Tensor, error) {
    // Extract shape, dtype from mat
    // Serialize data based on type
    // Create proto Tensor
}
```

**Location:** `x/marshaller/dndm/convert.go` (NEW)

## Implementation Plan

### Phase 1: Source Package (`cmd/display/source`)
1. Define Source interface (RegisterFlags, Start, ReadFrame, Close)
2. Implement base source with common functionality
3. Implement ImageSource (--images)
4. Implement VideoSource (--video)
5. Implement CameraSource (--camera)
6. Validate mutually exclusive source flags

### Phase 2: Destination Package (`cmd/display/destination`)
1. Define Destination interface (RegisterFlags, Start, AddFrame, Close)
2. Implement base destination with common functionality
3. Implement DisplayDestination (always on, --no-display to disable)
4. Implement VideoFileDestination (--output)
5. Support keyboard handling in DisplayDestination

### Phase 3: DNDM Integration
1. Create `dndm/proto/types/core/tensor.proto` (or use common proto types)
2. Generate Go code
3. Implement InterestSource (--interest) in source package
4. Implement IntentDestination (--intent) in destination package
5. Conversion functions: proto ↔ types.Frame

### Phase 4: Main Implementation
1. Create main.go that uses source and destination packages
2. Implement flag parsing and validation
3. Implement main processing loop
4. Handle context cancellation (ESC key, signals)

### Phase 5: Integration and Testing
1. End-to-end testing for each source type
2. Test all destination combinations
3. DNDM network testing
4. Multi-window display testing (future)
5. Performance testing

## File Structure

```
cmd/display/
├── DESIGN.md (this file)
├── main.go              # Main entry point using source/destination packages
├── source/              # Source package (reusable)
│   ├── source.go        # Source interface and base implementation
│   ├── images.go        # Image source (--images)
│   ├── video.go         # Video source (--video)
│   ├── camera.go        # Camera source (--camera)
│   └── interest.go      # DNDM interest source (--interest)
├── destination/         # Destination package (reusable)
│   ├── destination.go   # Destination interface and base implementation
│   ├── display.go       # Display destination (always on, --no-display to disable)
│   ├── intent.go        # DNDM intent destination (--intent)
│   └── video.go         # Video file destination (--output)
└── README.md            # Usage documentation
```

**Package Design:**
- `cmd/display/source`: Provides Source interface and implementations
- `cmd/display/destination`: Provides Destination interface and implementations
- Both packages can be imported by `calib_mono` and `calib_stereo` for reuse

## Dependencies

- `github.com/itohio/EasyRobot/x/marshaller/gocv` - GoCV marshallers/unmarshallers
- `github.com/itohio/EasyRobot/x/marshaller/types` - Common types (Frame, Tensor, etc.)
- `github.com/itohio/dndm` - DNDM client
- `gocv.io/x/gocv` - GoCV bindings
- Proto definitions (see DNDM integration section)

## Open Questions

1. **Multi-source synchronization**: How to handle frames from multiple sources?
   - Parallel display (multiple windows)? <- yes, multiple windows
   - Source synchronization is handled by unmarshaller. Out of scope of cli command.

2. **Tensor proto location**: Should DNDM have its own Tensor proto or import from EasyRobot?
   - this to be determined. I have to define proto files for common types.

3. **Frame metadata**: What metadata is needed for DNDM frames?
   - TBD

4. **Error handling**: How to handle DNDM connection failures or route mismatches?
   - TBD

5. **Performance**: For high-frame-rate sources, should frames be dropped or buffered?
   - TBD

## Future Enhancements

- Multiple display windows (one per frame)
- Frame filtering/processing (direct processing, no pipeline needed)
- Additional video codecs and formats
- Configurable video encoding parameters (bitrate, fps, etc.)

