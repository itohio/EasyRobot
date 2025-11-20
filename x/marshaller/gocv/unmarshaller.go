package gocv

import (
	"bufio"
	"errors"
	"fmt"
	"image"
	"io"
	"strings"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// Unmarshaller decodes GoCV payloads.
type Unmarshaller struct {
	opts              types.Options
	cfg               config
	activeControllers map[int]types.CameraController // device ID -> controller
}

// NewUnmarshaller constructs a GoCV unmarshaller.
func NewUnmarshaller(opts ...types.Option) types.Unmarshaller {
	baseOpts, baseCfg := applyOptions(types.Options{}, defaultConfig(), opts)
	return &Unmarshaller{
		opts:              baseOpts,
		cfg:               baseCfg,
		activeControllers: make(map[int]CameraController),
	}
}

// Format identifies the unmarshaller.
func (u *Unmarshaller) Format() string {
	return "gocv"
}

// CameraController returns the camera controller for the specified device ID.
func (u *Unmarshaller) CameraController(deviceID int) types.CameraController {
	return u.activeControllers[deviceID]
}

// Unmarshal decodes into the provided destination.
func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
	if dst == nil {
		return types.NewError("unmarshal", "gocv", "nil destination", nil)
	}

	localOpts, localCfg := applyOptions(u.opts, u.cfg, opts)

	switch out := dst.(type) {
	case *cv.Mat:
		return u.unmarshalMat(r, out)
	case **cv.Mat:
		if out == nil {
			return types.NewError("unmarshal", "gocv", "nil **gocv.Mat", nil)
		}
		var mat cv.Mat
		if err := u.unmarshalMat(r, &mat); err != nil {
			return err
		}
		*out = &mat
		return nil
	case *image.Image:
		return u.unmarshalImage(r, out)
	case **image.Image:
		if out == nil {
			return types.NewError("unmarshal", "gocv", "nil **image.Image", nil)
		}
		var img image.Image
		if err := u.unmarshalImage(r, &img); err != nil {
			return err
		}
		*out = &img
		return nil
	case *types.Tensor:
		tensor, err := u.unmarshalTensor(r, localCfg, localOpts.DestinationType)
		if err != nil {
			return err
		}
		*out = tensor
		return nil
	case *tensorgocv.Tensor:
		tensor, err := u.unmarshalTensor(r, localCfg, localOpts.DestinationType)
		if err != nil {
			return err
		}
		if tensor == nil {
			return types.NewError("unmarshal", "gocv", "tensor decode returned nil", nil)
		}
		gocvTensor, ok := tensor.(tensorgocv.Tensor)
		if !ok {
			return types.NewError("unmarshal", "gocv", "decoded tensor not GoCV-backed", nil)
		}
		*out = gocvTensor
		return nil
	case *types.FrameStream:
		stream, err := u.unmarshalStream(r, localCfg)
		if err != nil {
			return err
		}
		*out = stream
		return nil
	case *cv.Net:
		net, err := u.unmarshalNet(r, localCfg)
		if err != nil {
			return err
		}
		*out = net
		return nil
	case **cv.Net:
		if out == nil {
			return types.NewError("unmarshal", "gocv", "nil **gocv.Net", nil)
		}
		net, err := u.unmarshalNet(r, localCfg)
		if err != nil {
			return err
		}
		*out = &net
		return nil
	case *[]CameraInfo:
		devices, err := u.unmarshalCameraList(r)
		if err != nil {
			return err
		}
		*out = devices
		return nil
	default:
		return types.NewError("unmarshal", "gocv", fmt.Sprintf("unsupported destination %T", dst), nil)
	}
}

func (u *Unmarshaller) unmarshalMat(r io.Reader, dst *cv.Mat) error {
	if dst == nil {
		return types.NewError("unmarshal", "gocv", "nil *gocv.Mat", nil)
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return types.NewError("unmarshal", "gocv", "read mat bytes", err)
	}
	mat, err := decodeMatBytes(data, cv.IMReadColor)
	if err != nil {
		return types.NewError("unmarshal", "gocv", "decode mat", err)
	}
	*dst = mat
	return nil
}

func (u *Unmarshaller) unmarshalImage(r io.Reader, dst *image.Image) error {
	if dst == nil {
		return types.NewError("unmarshal", "gocv", "nil *image.Image", nil)
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return types.NewError("unmarshal", "gocv", "read image bytes", err)
	}
	img, err := decodeImageBytes(data)
	if err != nil {
		return types.NewError("unmarshal", "gocv", "decode image", err)
	}
	*dst = img
	return nil
}

func (u *Unmarshaller) unmarshalTensor(r io.Reader, cfg config, dstType tensortypes.DataType) (types.Tensor, error) {
	mat := cv.Mat{}
	if err := u.unmarshalMat(r, &mat); err != nil {
		return nil, err
	}
	return matToTensor(mat, cfg, dstType)
}

func (u *Unmarshaller) unmarshalStream(r io.Reader, cfg config) (types.FrameStream, error) {
	// Try to read protobuf manifest first
	if r != nil {
		// Use a buffered reader that can be used for both manifest and text paths
		bufReader := bufio.NewReader(r)
		
		manifest, err := readManifest(bufReader)
		if err != nil {
			return types.FrameStream{}, err
		}
		
		// If manifest was read, use it to configure the stream
		if manifest != nil {
			manifestToConfig(manifest, &cfg)
		} else if len(cfg.stream.sources) == 0 {
			// Fall back to legacy text format if no manifest and no configured sources
			// Use the buffered reader which still has all the data
			paths, err := readPaths(bufReader)
			if err != nil {
				return types.FrameStream{}, err
			}
			for _, p := range paths {
				cfg.stream.sources = append(cfg.stream.sources, sourceSpec{Path: p})
			}
		}
	}

	streams, err := buildStreams(cfg)
	if err != nil {
		return types.FrameStream{}, err
	}

	// Register camera controllers for video devices
	u.registerCameraControllers(streams)

	return newFrameStream(cfg.ctx, streams, cfg.stream.allowBestEffort, cfg.stream.sequential)
}

func (u *Unmarshaller) unmarshalNet(r io.Reader, cfg config) (cv.Net, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return cv.Net{}, types.NewError("unmarshal", "gocv", "read DNN payload", err)
	}
	if len(data) == 0 {
		return cv.Net{}, types.NewError("unmarshal", "gocv", "empty DNN payload", errors.New("empty"))
	}
	net, err := loadNetFromBytes(data, cfg)
	if err != nil {
		return cv.Net{}, err
	}
	return net, nil
}

func (u *Unmarshaller) unmarshalCameraList(r io.Reader) ([]types.CameraInfo, error) {
	// Check if this is a "list" command
	if r != nil {
		buf := make([]byte, 4)
		n, err := r.Read(buf)
		if err != nil && err != io.EOF {
			return nil, types.NewError("unmarshal", "gocv", "read list command", err)
		}
		command := strings.TrimSpace(string(buf[:n]))
		if command != "list" {
			return nil, types.NewError("unmarshal", "gocv", "expected 'list' command, got: "+command, nil)
		}
	}

	// Enumerate video devices
	devices, err := enumerateVideoDevices()
	if err != nil {
		return nil, types.NewError("unmarshal", "gocv", "enumerate devices", err)
	}

	return devices, nil
}

func (u *Unmarshaller) registerCameraControllers(streams []sourceStream) {
	for _, stream := range streams {
		if loader, ok := stream.(*videoDeviceLoader); ok {
			if controller := loader.CameraController(); controller != nil {
				u.activeControllers[loader.spec.ID] = controller
			}
		}
	}
}

func readPaths(r io.Reader) ([]string, error) {
	var (
		scanner = bufio.NewScanner(r)
		paths   []string
	)
	for scanner.Scan() {
		path := strings.TrimSpace(scanner.Text())
		if path == "" {
			continue
		}
		paths = append(paths, path)
	}
	if err := scanner.Err(); err != nil {
		return nil, types.NewError("unmarshal", "gocv", "read paths", err)
	}
	return paths, nil
}
