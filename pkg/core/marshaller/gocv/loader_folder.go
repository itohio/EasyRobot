package gocv

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

type folderLoader struct {
	path  string
	label string
	cfg   config
	files []string
	index int
}

func newFolderLoader(path string, cfg config) sourceStream {
	return &folderLoader{
		path:  filepath.Clean(path),
		label: filepath.Clean(path),
		cfg:   cfg,
	}
}

func newFileListLoader(label string, files []string, cfg config) sourceStream {
	cp := append([]string(nil), files...)
	sort.Strings(cp)
	return &folderLoader{
		label: label,
		cfg:   cfg,
		files: cp,
	}
}

func (l *folderLoader) Next(ctx context.Context) (frameItem, bool, error) {
	if err := l.ensureFiles(); err != nil {
		return frameItem{}, false, err
	}

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
	if _, exists := item.metadata["source"]; !exists {
		item.metadata["source"] = "folder"
	}
	if l.path != "" {
		item.metadata["folder"] = l.path
	}
	if l.label != "" {
		item.metadata["set"] = l.label
	}
	item.metadata["timestamp"] = time.Now().UnixNano()
	item.metadata["relative"] = filepath.Base(path)
	item.metadata["set_index"] = l.index - 1

	return item, true, nil
}

func (l *folderLoader) ensureFiles() error {
	if l.files != nil {
		return nil
	}
	if l.path == "" {
		return fmt.Errorf("gocv: no folder path provided for loader")
	}
	dirEntries, err := os.ReadDir(l.path)
	if err != nil {
		return fmt.Errorf("gocv: read dir %s: %w", l.path, err)
	}
	files := make([]string, 0, len(dirEntries))
	for _, entry := range dirEntries {
		if entry.IsDir() {
			continue
		}
		full := filepath.Join(l.path, entry.Name())
		switch classifyPath(full) {
		case sourceKindSingle:
			files = append(files, full)
		default:
			if filepath.Ext(full) == ".mat" {
				files = append(files, full)
			}
		}
	}
	sort.Strings(files)
	l.files = files
	return nil
}

func (l *folderLoader) Close() error {
	return nil
}
