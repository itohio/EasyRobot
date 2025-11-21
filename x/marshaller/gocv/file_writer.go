package gocv

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

type fileWriter struct {
	cfg     config
	targets []string
	summary bytes.Buffer
}

func newFileWriter(targets []string, cfg config) (*fileWriter, error) {
	if len(targets) == 0 {
		return &fileWriter{cfg: cfg}, nil
	}
	clean := make([]string, 0, len(targets))
	for _, dir := range targets {
		if strings.TrimSpace(dir) == "" {
			continue
		}
		clean = append(clean, dir)
	}
	return &fileWriter{
		cfg:     cfg,
		targets: clean,
	}, nil
}

// WriteFrame implements StreamSink interface.
func (fw *fileWriter) WriteFrame(frame types.Frame) error {
	return fw.Write(frame)
}

// Write writes a frame to disk (legacy method name, kept for backward compatibility).
func (fw *fileWriter) Write(frame types.Frame) error {
	if len(fw.targets) == 0 || len(frame.Tensors) == 0 {
		return nil
	}

	names := extractNameList(frame.Metadata)
	if len(names) == 0 {
		return nil
	}

	limit := len(frame.Tensors)
	if len(names) < limit {
		limit = len(names)
	}
	if len(fw.targets) < limit {
		limit = len(fw.targets)
	}

	for i := 0; i < limit; i++ {
		name := strings.TrimSpace(names[i])
		if name == "" {
			continue
		}
		filename := name
		if filepath.Ext(filename) == "" {
			filename += normalizeImageFormat(fw.cfg.codec.imageEncoding)
		}

		destDir := fw.targets[i]
		fullPath := filepath.Join(destDir, filename)

		mat, err := tensorToMat(frame.Tensors[i])
		if err != nil {
			return err
		}
		data, err := encodeMatBytes(mat, fw.cfg.codec.imageEncoding)
		mat.Close()
		// Release objects that implement Releaser after writing if WithRelease is enabled
		if fw.cfg.ReleaseAfterProcessing {
			if releaser, ok := frame.Tensors[i].(types.Releaser); ok {
				releaser.Release()
			}
		}
		if err != nil {
			return err
		}
		if err := os.WriteFile(fullPath, data, 0o666); err != nil {
			return err
		}
		fmt.Fprintf(&fw.summary, "%s\n", fullPath)
	}

	return nil
}

func (fw *fileWriter) Close() error {
	return nil
}

func (fw *fileWriter) Summary() []byte {
	return fw.summary.Bytes()
}
