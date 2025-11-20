package v4l

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/vladimirvivien/go4vl/device"
	"github.com/vladimirvivien/go4vl/v4l2"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// v4lStream implements the CameraStream interface
type v4lStream struct {
	device     *v4l2.Device
	stream     *device.Stream
	options    Options
	format     Format
	controls   []types.ControlInfo
	frameChan  chan Frame
	ctx        context.Context
	cancel     context.CancelFunc
	running    bool
	mu         sync.RWMutex
	controller *v4lController
}

// newV4LStream creates a new V4L stream
func newV4LStream(dev *v4l2.Device, opts Options) (*v4lStream, error) {
	// Set pixel format
	pixFmt := v4l2.PixFormat{
		PixelFormat: v4l2.PixelFmt(opts.PixelFormat),
		Width:       uint32(opts.Width),
		Height:      uint32(opts.Height),
	}

	if err := dev.SetPixFormat(pixFmt); err != nil {
		return nil, fmt.Errorf("failed to set pixel format: %w", err)
	}

	// Get the actual format that was set
	actualFmt, err := dev.GetPixFormat()
	if err != nil {
		return nil, fmt.Errorf("failed to get pixel format: %w", err)
	}

	format := Format{
		Width:       int(actualFmt.Width),
		Height:      int(actualFmt.Height),
		PixelFormat: PixelFormat(actualFmt.PixelFormat),
		Field:       Field(actualFmt.Field),
	}

	// Create stream with buffer options
	stream, err := device.Open(dev.Path(),
		device.WithBufferSize(opts.BufferCount),
		device.WithPixFormat(actualFmt),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to open device stream: %w", err)
	}

	// Query controls
	var controls []types.ControlInfo
	ctrlList, err := dev.Controls()
	if err == nil {
		for _, ctrl := range ctrlList {
			// Map V4L2 control to shared name if possible
			ctrlName := string(ctrl.Name[:])
			if name, exists := controlIDToName[ControlID(ctrl.ID)]; exists {
				ctrlName = name
			}

			var menuItems []string
			// Query menu items if applicable
			if ctrl.Type == v4l2.CtrlTypeMenu {
				menu, err := dev.ControlMenu(ctrl.ID)
				if err == nil {
					for _, item := range menu {
						menuItems = append(menuItems, string(item.Name[:]))
					}
				}
			}

			ctrlInfo := ConvertControlToShared(
				ControlID(ctrl.ID),
				ctrlName,
				ControlType(ctrl.Type),
				ctrl.Min,
				ctrl.Max,
				ctrl.Default,
				ctrl.Step,
				menuItems,
			)

			controls = append(controls, ctrlInfo)
		}
	}

	ctx, cancel := context.WithCancel(opts.Context)

	// Create controller
	controller := &v4lController{
		device:   dev,
		controls: controls,
	}

	return &v4lStream{
		device:     dev,
		stream:     stream,
		options:    opts,
		format:     format,
		controls:   controls,
		frameChan:  make(chan Frame, opts.BufferCount*2), // Buffer channel
		ctx:        ctx,
		cancel:     cancel,
		controller: controller,
	}, nil
}

// Start begins frame capture
func (s *v4lStream) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		return fmt.Errorf("stream already running")
	}

	if err := s.stream.Start(ctx); err != nil {
		return fmt.Errorf("failed to start stream: %w", err)
	}

	s.running = true

	// Start capture goroutine
	go s.captureLoop(ctx)

	return nil
}

// Stop halts frame capture
func (s *v4lStream) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.running {
		return nil
	}

	s.running = false
	s.cancel()

	if err := s.stream.Stop(); err != nil {
		return fmt.Errorf("failed to stop stream: %w", err)
	}

	return nil
}

// Controller returns the camera controller for runtime control
func (s *v4lStream) Controller() types.CameraController {
	return s.controller
}

// Close closes the stream
func (s *v4lStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		s.Stop()
	}

	if s.stream != nil {
		err := s.stream.Close()
		s.stream = nil
		return err
	}

	return nil
}

// captureLoop runs the frame capture loop
func (s *v4lStream) captureLoop(ctx context.Context) {
	defer close(s.frameChan)

	index := 0
	for {
		select {
		case <-ctx.Done():
			return
		case <-s.ctx.Done():
			return
		default:
		}

		// Get frame from stream
		frame := <-s.stream.GetOutput()

		// Convert frame data to tensor
		tensor := s.createTensorFromFrame(frame)

		// Create frame metadata
		v4lFrame := Frame{
			Index:     index,
			Timestamp: time.Now().UnixNano(),
			Tensor:    tensor,
			Tensors:   []types.Tensor{tensor},
			Metadata: map[string]any{
				"device":      s.device.Path(),
				"format":      s.format,
				"frame_size":  len(frame),
				"timestamp":   time.Now().UnixNano(),
			},
		}

		// Send frame to channel (non-blocking)
		select {
		case s.frameChan <- v4lFrame:
			index++
		case <-ctx.Done():
			return
		case <-s.ctx.Done():
			return
		default:
			// Channel full, skip frame
			tensor.Release()
		}
	}
}

// createTensorFromFrame converts frame bytes to tensor
func (s *v4lStream) createTensorFromFrame(frame []byte) types.Tensor {
	width, height := s.format.Width, s.format.Height
	var channels int

	// Determine channels based on pixel format
	switch s.format.PixelFormat {
	case PixelFmtRGB24, PixelFmtBGR24:
		channels = 3
	case PixelFmtYUYV, PixelFmtUYVY:
		channels = 2
	case PixelFmtNV12:
		channels = 1 // Luminance plane
	case PixelFmtGREY:
		channels = 1
	case PixelFmtMJPEG:
		// For MJPEG, we need to decode first
		// This is a simplified approach - in practice, you'd need to decode
		channels = 3
	default:
		channels = 3 // Default assumption
	}

	// Create tensor from frame data
	return s.options.TensorFactory(frame, width, height, channels)
}

// MultiStream manages multiple synchronized streams
type MultiStream struct {
	streams   []Stream
	frameChan chan Frame
	ctx       context.Context
	cancel    context.CancelFunc
	running   bool
	mu        sync.RWMutex

	allowBestEffort bool
	sequential      bool
}

// NewMultiStream creates a synchronized multi-camera stream
func NewMultiStream(streams []Stream, opts ...Option) *MultiStream {
	var options Options
	for _, opt := range opts {
		opt.Apply(&options)
	}

	ctx, cancel := context.WithCancel(options.Context)

	return &MultiStream{
		streams:         streams,
		frameChan:       make(chan Frame, len(streams)*8),
		ctx:             ctx,
		cancel:          cancel,
		allowBestEffort: options.AllowBestEffort,
		sequential:      options.Sequential,
	}
}

// Start begins capture on all streams
func (ms *MultiStream) Start(ctx context.Context) error {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	if ms.running {
		return fmt.Errorf("multi-stream already running")
	}

	// Start all individual streams
	for _, stream := range ms.streams {
		if err := stream.Start(ctx); err != nil {
			return fmt.Errorf("failed to start stream: %w", err)
		}
	}

	ms.running = true

	// Start synchronization goroutine
	go ms.syncLoop(ctx)

	return nil
}

// Stop halts capture on all streams
func (ms *MultiStream) Stop() error {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	if !ms.running {
		return nil
	}

	ms.running = false
	ms.cancel()

	for _, stream := range ms.streams {
		stream.Stop()
	}

	return nil
}

// FrameChannel returns the synchronized frame channel
func (ms *MultiStream) FrameChannel() <-chan Frame {
	return ms.frameChan
}

// Close closes all streams
func (ms *MultiStream) Close() error {
	ms.Stop()

	var errors []error
	for _, stream := range ms.streams {
		if err := stream.Close(); err != nil {
			errors = append(errors, err)
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("errors closing streams: %v", errors)
	}
	return nil
}

// syncLoop synchronizes frames from multiple streams
func (ms *MultiStream) syncLoop(ctx context.Context) {
	defer close(ms.frameChan)

	if ms.sequential {
		ms.syncSequential(ctx)
	} else {
		ms.syncParallel(ctx)
	}
}

// syncSequential processes streams one after another
func (ms *MultiStream) syncSequential(ctx context.Context) {
	index := 0
	for _, stream := range ms.streams {
		streamChan := stream.FrameChannel()

		for {
			select {
			case frame, ok := <-streamChan:
				if !ok {
					return
				}

				// Update frame index for global ordering
				frame.Index = index
				frame.Metadata["global_index"] = index
				frame.Metadata["stream_count"] = len(ms.streams)

				select {
				case ms.frameChan <- frame:
					index++
				case <-ctx.Done():
					return
				case <-ms.ctx.Done():
					return
				}

			case <-ctx.Done():
				return
			case <-ms.ctx.Done():
				return
			}
		}
	}
}

// syncParallel synchronizes frames by index across all streams
func (ms *MultiStream) syncParallel(ctx context.Context) {
	type frameWithStream struct {
		frame  Frame
		stream int
	}

	// Create channels for each stream
	streamChans := make([]<-chan Frame, len(ms.streams))
	for i, stream := range ms.streams {
		streamChans[i] = stream.FrameChannel()
	}

	// Collect frames from all streams
	pendingFrames := make([][]Frame, len(ms.streams))
	currentIndex := 0

	for {
		select {
		case <-ctx.Done():
			return
		case <-ms.ctx.Done():
			return
		default:
		}

		// Check for new frames on all channels
		framesReady := true
		for i := range streamChans {
			for len(pendingFrames[i]) == 0 {
				select {
				case frame, ok := <-streamChans[i]:
					if !ok {
						// Stream ended
						if !ms.allowBestEffort {
							return
						}
						continue
					}
					pendingFrames[i] = append(pendingFrames[i], frame)
				case <-ctx.Done():
					return
				case <-ms.ctx.Done():
					return
				default:
					if !ms.allowBestEffort {
						framesReady = false
					}
					break
				}
			}
		}

		if !framesReady && !ms.allowBestEffort {
			continue
		}

		// Combine frames with same index
		var combinedFrames []Frame
		minIndex := -1

		for i := range pendingFrames {
			if len(pendingFrames[i]) > 0 {
				frame := pendingFrames[i][0]
				if minIndex == -1 || frame.Index < minIndex {
					minIndex = frame.Index
				}
			}
		}

		// Collect all frames with the current index
		for i := range pendingFrames {
			if len(pendingFrames[i]) > 0 && pendingFrames[i][0].Index == minIndex {
				combinedFrames = append(combinedFrames, pendingFrames[i][0])
				pendingFrames[i] = pendingFrames[i][1:] // Remove processed frame
			}
		}

		if len(combinedFrames) > 0 {
			// Create combined frame
			combinedFrame := Frame{
				Index:     currentIndex,
				Timestamp: time.Now().UnixNano(),
				Tensors:   make([]types.Tensor, len(combinedFrames)),
				Metadata: map[string]any{
					"global_index": currentIndex,
					"stream_count": len(ms.streams),
					"combined":     true,
				},
			}

			// Add tensors and metadata from all frames
			for i, frame := range combinedFrames {
				combinedFrame.Tensors[i] = frame.Tensor
				combinedFrame.Metadata[fmt.Sprintf("stream_%d", i)] = frame.Metadata
			}

			// Set primary tensor to first one for compatibility
			if len(combinedFrames) > 0 {
				combinedFrame.Tensor = combinedFrames[0].Tensor
			}

			select {
			case ms.frameChan <- combinedFrame:
				currentIndex++
			case <-ctx.Done():
				return
			case <-ms.ctx.Done():
				return
			}
		} else if !ms.allowBestEffort {
			// No frames ready and not allowing best effort
			time.Sleep(1 * time.Millisecond)
		}
	}
}

// v4lController implements the CameraController interface
type v4lController struct {
	device   *v4l2.Device
	controls []types.ControlInfo
}

// Controls returns available device controls
func (c *v4lController) Controls() []types.ControlInfo {
	return c.controls
}

// GetControl gets a control value by name
func (c *v4lController) GetControl(name string) (int32, error) {
	ctrlID, exists := controlNameToID[name]
	if !exists {
		return 0, fmt.Errorf("unknown control name: %s", name)
	}
	return c.device.GetControl(v4l2.CtrlID(ctrlID))
}

// SetControl sets a control value by name
func (c *v4lController) SetControl(name string, value int32) error {
	ctrlID, exists := controlNameToID[name]
	if !exists {
		return fmt.Errorf("unknown control name: %s", name)
	}
	return c.device.SetControl(v4l2.CtrlID(ctrlID), value)
}

// GetControls gets all control values
func (c *v4lController) GetControls() (map[string]int32, error) {
	result := make(map[string]int32)
	for _, ctrl := range c.controls {
		if value, err := c.GetControl(ctrl.Name); err == nil {
			result[ctrl.Name] = value
		}
	}
	return result, nil
}

// SetControls sets multiple control values
func (c *v4lController) SetControls(controls map[string]int32) error {
	for name, value := range controls {
		if err := c.SetControl(name, value); err != nil {
			return fmt.Errorf("failed to set control %s: %w", name, err)
		}
	}
	return nil
}
