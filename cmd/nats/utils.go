package main

import (
	"errors"
	"os"
	"path"
	"strings"
)

func ReadFileList(path string) []string {
	info, err := os.Stat(path)
	if err != nil {
		panic(err)
	}

	if !info.IsDir() {
		return ReadList(path)
	}

	files, err := OSReadDir(path)
	if err != nil {
		panic(err)
	}
	return files
}

func OSReadDir(root string) ([]string, error) {
	var files []string
	f, err := os.Open(root)
	if err != nil {
		return files, err
	}
	fileInfo, err := f.Readdir(-1)
	f.Close()
	if err != nil {
		return files, err
	}

	for _, file := range fileInfo {
		name := file.Name()
		if isImagePath(name) {

		}
		files = append(files, path.Join(root, name))
	}
	return files, nil
}

func ReadList(file string) []string {
	panic(errors.New("not supported"))
	return nil
}

func isImagePath(name string) bool {
	return strings.HasSuffix(name, ".bmp") ||
		strings.HasSuffix(name, ".jpg") ||
		strings.HasSuffix(name, ".png") ||
		strings.HasSuffix(name, ".jpeg")
}
