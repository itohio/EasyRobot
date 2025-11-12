package gocv

import (
	"fmt"
	"os"
)

func buildStreams(cfg config) ([]sourceStream, error) {
	specs, err := resolveSources(cfg.sources)
	if err != nil {
		return nil, err
	}

	streams := make([]sourceStream, 0, len(specs))
	for _, spec := range specs {
		switch spec.Kind {
		case sourceKindVideoDevice:
			stream, err := newVideoDeviceLoader(*spec.Device, cfg)
			if err != nil {
				return nil, err
			}
			streams = append(streams, stream)
		case sourceKindVideoFile:
			stream, err := newVideoFileLoader(spec.Path, cfg)
			if err != nil {
				return nil, err
			}
			streams = append(streams, stream)
		case sourceKindDirectory:
			if err := ensureDirExists(spec.Path); err != nil {
				return nil, err
			}
			streams = append(streams, newFolderLoader(spec.Path, cfg))
		case sourceKindFileList:
			streams = append(streams, newFileListLoader(spec.Path, spec.Files, cfg))
		case sourceKindSingle, sourceKindUnknown:
			streams = append(streams, newImageLoader(spec.Path, cfg))
		default:
			return nil, fmt.Errorf("gocv: unsupported source kind %v", spec.Kind)
		}
	}
	return streams, nil
}

func ensureDirExists(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("gocv: stat dir %s: %w", path, err)
	}
	if !info.IsDir() {
		return fmt.Errorf("gocv: path %s is not a directory", path)
	}
	return nil
}
