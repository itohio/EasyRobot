package gocv

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"
)

type fileListLoader struct {
	label string
	files []string
	cfg   config
	index int
}

func newFileListLoader(label string, files []string, cfg config) (sourceStream, error) {
	if len(files) == 0 {
		return nil, fmt.Errorf("gocv: empty file list for %s", label)
	}
	sorted := append([]string(nil), files...)
	if cfg.sorter != nil {
		sorted = cfg.sorter(sorted)
	} else {
		sort.Strings(sorted)
	}
	return &fileListLoader{
		label: label,
		files: sorted,
		cfg:   cfg,
	}, nil
}

func (l *fileListLoader) Next(ctx context.Context) (frameItem, bool, error) {
	if l.index >= len(l.files) {
		return frameItem{}, false, nil
	}

	select {
	case <-ctx.Done():
		return frameItem{}, false, ctx.Err()
	default:
	}

	path := l.files[l.index]
	l.index++

	item, err := loadFrameFromFile(path, l.cfg)
	if err != nil {
		return frameItem{}, false, err
	}

	if item.metadata == nil {
		item.metadata = map[string]any{}
	}
	if _, ok := item.metadata["source"]; !ok {
		item.metadata["source"] = "filelist"
	}
	item.metadata["set"] = l.label
	item.metadata["relative"] = filepath.Base(path)
	item.metadata["set_index"] = l.index - 1
	item.metadata["filename"] = filepath.Base(path)
	item.metadata["index"] = l.index - 1
	if _, ok := item.metadata["name"]; !ok {
		item.metadata["name"] = []string{filepath.Base(path)}
	}

	return item, true, nil
}

func (l *fileListLoader) Close() error {
	return nil
}
