package gocv

import (
	"fmt"
	"os"
	"path/filepath"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

func loadNetFromBytes(data []byte, cfg config) (cv.Net, error) {
	format := cfg.dnnFormat
	if format == "" {
		format = "onnx"
	}

	switch format {
	case "onnx":
		return loadNetUsingTemp(data, ".onnx", cfg, func(modelPath, configPath string) cv.Net {
			return cv.ReadNet(modelPath, configPath)
		})
	default:
		return cv.Net{}, types.NewError("unmarshal", "gocv",
			fmt.Sprintf("unsupported DNN format %s", format), nil)
	}
}

func loadNetUsingTemp(data []byte, ext string, cfg config, loader func(modelPath, configPath string) cv.Net) (cv.Net, error) {
	tempDir, err := os.MkdirTemp("", "gocv-dnn-*")
	if err != nil {
		return cv.Net{}, types.NewError("unmarshal", "gocv", "create temp dir", err)
	}
	defer os.RemoveAll(tempDir)

	modelPath := filepath.Join(tempDir, "model"+ext)
	if err := os.WriteFile(modelPath, data, 0o600); err != nil {
		return cv.Net{}, types.NewError("unmarshal", "gocv", "write temp model", err)
	}

	net := loader(modelPath, "")
	if net.Empty() {
		return cv.Net{}, types.NewError("unmarshal", "gocv", "load network", fmt.Errorf("empty net"))
	}

	if cfg.netBackend != cv.NetBackendDefault {
		net.SetPreferableBackend(cfg.netBackend)
	}
	if cfg.netTarget != cv.NetTargetCPU {
		net.SetPreferableTarget(cfg.netTarget)
	}

	return net, nil
}
