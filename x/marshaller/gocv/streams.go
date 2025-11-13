package gocv

import "fmt"

func buildStreams(cfg config) ([]sourceStream, error) {
	specs, err := resolveSources(cfg)
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
		case sourceKindFileList:
			stream, err := newFileListLoader(spec.Path, spec.Files, cfg)
			if err != nil {
				return nil, err
			}
			streams = append(streams, stream)
		case sourceKindSingle, sourceKindUnknown:
			streams = append(streams, newImageLoader(spec.Path, cfg))
		default:
			return nil, fmt.Errorf("gocv: unsupported source kind %v", spec.Kind)
		}
	}
	return streams, nil
}
