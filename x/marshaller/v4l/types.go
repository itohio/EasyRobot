package v4l

import (
	"context"
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// CameraDevice implements the shared CameraDevice interface
type CameraDevice interface {
	// Info returns device information and capabilities
	Info() types.CameraInfo

	// Open opens the device with specified options
	Open(opts ...types.CameraOption) (types.CameraStream, error)

	// Close closes the device
	Close() error
}

// CameraStream implements the shared CameraStream interface
type CameraStream interface {
	// Start begins frame capture
	Start(ctx context.Context) error

	// Stop halts frame capture
	Stop() error

	// Controller returns the camera controller for runtime control
	Controller() types.CameraController

	// Close closes the stream
	Close() error
}

// CameraController implements the shared CameraController interface
type CameraController interface {
	// Controls returns available device controls
	Controls() []types.ControlInfo

	// SetControl sets a device control value by name
	SetControl(name string, value int32) error

	// GetControl gets a device control value by name
	GetControl(name string) (int32, error)

	// GetControls gets all control values
	GetControls() (map[string]int32, error)

	// SetControls sets multiple control values
	SetControls(controls map[string]int32) error
}

// Frame represents a captured video frame
type Frame struct {
	// Index is the frame sequence number
	Index int

	// Timestamp is capture timestamp (nanoseconds since epoch)
	Timestamp int64

	// Tensor contains the frame data as uint8 tensor (single camera)
	Tensor types.Tensor

	// Tensors contains multiple tensors (for synchronized multi-camera frames)
	Tensors []types.Tensor

	// Metadata contains frame-specific information
	Metadata map[string]any
}

// CapabilityFlags represents device capabilities
type CapabilityFlags uint32

const (
	CapVideoCapture CapabilityFlags = 1 << iota
	CapVideoOutput
	CapVideoOverlay
	CapVBI_CAPTURE
	CapVBI_OUTPUT
	CapSLICED_VBI_CAPTURE
	CapSLICED_VBI_OUTPUT
	CapRDS_CAPTURE
	CapVideoOutputOverlay
	CapHW_FREQ_SEEK
	CapRDS_OUTPUT
	CapVideoCaptureMplane
	CapVideoOutputMplane
	CapVideoM2M
	CapVideoM2MMplane
	CapTuner
	CapAudio
	CapRadio
	CapModulator
	CapReadWrite
	CapAsyncIO
	CapStreaming
	CapDeviceCaps
)

// V4L-specific constants for internal use
const (
	// Common pixel formats (for internal mapping)
	PixelFmtRGB24  PixelFormat = 'R' | 'G'<<8 | 'B'<<16 | '3'<<24  // RGB24
	PixelFmtBGR24  PixelFormat = 'B' | 'G'<<8 | 'R'<<16 | '3'<<24  // BGR24
	PixelFmtYUYV   PixelFormat = 'Y' | 'U'<<8 | 'Y'<<16 | 'V'<<24  // YUYV
	PixelFmtUYVY   PixelFormat = 'U' | 'Y'<<8 | 'V'<<16 | 'Y'<<24  // UYVY
	PixelFmtYUV420 PixelFormat = 'Y' | 'U'<<8 | '1'<<16 | '2'<<24  // YU12
	PixelFmtYUV422P PixelFormat = '4' | '2'<<8 | '2'<<16 | 'P'<<24 // 422P
	PixelFmtMJPEG  PixelFormat = 'M' | 'J'<<8 | 'P'<<16 | 'G'<<24  // MJPEG
	PixelFmtH264   PixelFormat = 'H' | '2'<<8 | '6'<<16 | '4'<<24  // H264
	PixelFmtNV12   PixelFormat = 'N' | 'V'<<8 | '1'<<16 | '2'<<24  // NV12
	PixelFmtGREY   PixelFormat = 'G' | 'R'<<8 | 'E'<<16 | 'Y'<<24  // GREY
)

// Format describes video stream format (internal V4L-specific)
type Format struct {
	// Width in pixels
	Width int

	// Height in pixels
	Height int

	// PixelFormat is the fourcc pixel format
	PixelFormat PixelFormat

	// FrameRate is frames per second
	FrameRate Fraction

	// Field order (interlaced/progressive)
	Field Field
}

// Fraction represents a fractional value
type Fraction struct {
	Numerator   int
	Denominator int
}

// Field represents interlacing field order
type Field int

const (
	FieldAny Field = iota
	FieldNone
	FieldTop
	FieldBottom
	FieldInterlaced
	FieldSeqTB
	FieldSeqBT
	FieldAlternate
	FieldInterlacedTB
	FieldInterlacedBT
)

// PixelFormat represents V4L2 pixel formats
type PixelFormat uint32

// ConvertToSharedFormat converts internal Format to shared VideoFormat
func (f Format) ToVideoFormat() types.VideoFormat {
	desc := pixelFormatDescription(f.PixelFormat)
	return types.VideoFormat{
		PixelFormat: f.PixelFormat,
		Description: desc,
		Width:       f.Width,
		Height:      f.Height,
		Metadata: map[string]any{
			"frame_rate": f.FrameRate,
			"field":      f.Field,
		},
	}
}

// Common pixel formats
const (
	PixelFmtRGB24  PixelFormat = 'R' | 'G'<<8 | 'B'<<16 | '3'<<24  // RGB24
	PixelFmtBGR24  PixelFormat = 'B' | 'G'<<8 | 'R'<<16 | '3'<<24  // BGR24
	PixelFmtYUYV   PixelFormat = 'Y' | 'U'<<8 | 'Y'<<16 | 'V'<<24  // YUYV
	PixelFmtUYVY   PixelFormat = 'U' | 'Y'<<8 | 'V'<<16 | 'Y'<<24  // UYVY
	PixelFmtYUV420 PixelFormat = 'Y' | 'U'<<8 | '1'<<16 | '2'<<24  // YU12
	PixelFmtYUV422P PixelFormat = '4' | '2'<<8 | '2'<<16 | 'P'<<24 // 422P
	PixelFmtMJPEG  PixelFormat = 'M' | 'J'<<8 | 'P'<<16 | 'G'<<24  // MJPEG
	PixelFmtH264   PixelFormat = 'H' | '2'<<8 | '6'<<16 | '4'<<24  // H264
	PixelFmtNV12   PixelFormat = 'N' | 'V'<<8 | '1'<<16 | '2'<<24  // NV12
	PixelFmtGREY   PixelFormat = 'G' | 'R'<<8 | 'E'<<16 | 'Y'<<24  // GREY
)

// ControlID represents V4L2 control identifiers
type ControlID uint32

// Common control IDs (internal mapping)
const (
	CtrlBrightness ControlID = iota
	CtrlContrast
	CtrlSaturation
	CtrlHue
	CtrlAutoWhiteBalance
	CtrlDoWhiteBalance
	CtrlRedBalance
	CtrlBlueBalance
	CtrlGamma
	CtrlExposure
	CtrlAutogain
	CtrlGain
	CtrlHFlip
	CtrlVFlip
	CtrlPowerLineFrequency
	CtrlHueAuto
	CtrlWhiteBalanceTemperature
	CtrlSharpness
	CtrlBacklightCompensation
	CtrlChromaAGC
	CtrlColorKiller
	CtrlColorEffects
	CtrlAutobrightness
	CtrlBandStopFilter
	CtrlRot
	CtrlBgColor
	CtrlChromaGain
	CtrlIlluminator1
	CtrlIlluminator2
)

// ControlType represents the type of control (internal mapping)
type ControlType int

const (
	CtrlTypeInteger ControlType = iota
	CtrlTypeBoolean
	CtrlTypeMenu
	CtrlTypeButton
	CtrlTypeInteger64
	CtrlTypeCtrlClass
	CtrlTypeString
	CtrlTypeBitmask
	CtrlTypeIntegerMenu
)

// Control mappings from V4L2 IDs to shared names
var controlNameToID = map[string]ControlID{
	types.CameraControlBrightness:            CtrlBrightness,
	types.CameraControlContrast:              CtrlContrast,
	types.CameraControlSaturation:            CtrlSaturation,
	types.CameraControlHue:                   CtrlHue,
	types.CameraControlGamma:                 CtrlGamma,
	types.CameraControlExposure:              CtrlExposure,
	types.CameraControlGain:                  CtrlGain,
	types.CameraControlSharpness:             CtrlSharpness,
	types.CameraControlWhiteBalanceTemp:      CtrlWhiteBalanceTemperature,
	types.CameraControlAutoWhiteBalance:      CtrlAutoWhiteBalance,
	types.CameraControlBacklightCompensation: CtrlBacklightCompensation,
	types.CameraControlHFlip:                 CtrlHFlip,
	types.CameraControlVFlip:                 CtrlVFlip,
}

// controlIDToName maps V4L2 control IDs to shared names
var controlIDToName = make(map[ControlID]string)

func init() {
	for name, id := range controlNameToID {
		controlIDToName[id] = name
	}
}

// controlTypeToString converts ControlType to shared string format
func controlTypeToString(ct ControlType) string {
	switch ct {
	case CtrlTypeInteger, CtrlTypeInteger64:
		return "integer"
	case CtrlTypeBoolean:
		return "boolean"
	case CtrlTypeMenu:
		return "menu"
	case CtrlTypeButton:
		return "button"
	case CtrlTypeString:
		return "string"
	case CtrlTypeBitmask:
		return "bitmask"
	case CtrlTypeIntegerMenu:
		return "integer_menu"
	default:
		return "unknown"
	}
}

// pixelFormatDescription returns a human-readable description of a pixel format
func pixelFormatDescription(pf PixelFormat) string {
	switch pf {
	case PixelFmtRGB24:
		return "RGB24 (24-bit RGB)"
	case PixelFmtBGR24:
		return "BGR24 (24-bit BGR)"
	case PixelFmtYUYV:
		return "YUYV (YUV 4:2:2 packed)"
	case PixelFmtUYVY:
		return "UYVY (YUV 4:2:2 packed)"
	case PixelFmtYUV420:
		return "YU12 (YUV 4:2:0 planar)"
	case PixelFmtYUV422P:
		return "422P (YUV 4:2:2 planar)"
	case PixelFmtMJPEG:
		return "MJPEG (Motion JPEG)"
	case PixelFmtH264:
		return "H264 (H.264 compressed)"
	case PixelFmtNV12:
		return "NV12 (YUV 4:2:0 semi-planar)"
	case PixelFmtGREY:
		return "GREY (8-bit grayscale)"
	default:
		return fmt.Sprintf("Unknown (0x%08x)", uint32(pf))
	}
}

// ConvertControlToShared converts internal ControlInfo to shared ControlInfo
func ConvertControlToShared(id ControlID, name string, ctrlType ControlType, min, max, def, step int32, menuItems []string) types.ControlInfo {
	return types.ControlInfo{
		ID:          id,
		Name:        name,
		Description: name,
		Type:        controlTypeToString(ctrlType),
		Min:         min,
		Max:         max,
		Default:     def,
		Step:        step,
		MenuItems:   menuItems,
		Metadata: map[string]any{
			"v4l_id": id,
		},
	}
}
