package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path"
	"strings"
	"time"

	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/core/pipeline/steps/fps"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"
	"github.com/foxis/EasyRobot/pkg/vision/display"
	"github.com/foxis/EasyRobot/pkg/vision/reader"
	"github.com/foxis/EasyRobot/pkg/vision/writer"
)

func main() {
	help := flag.Bool("help", false, "Help")
	hide := flag.Bool("hide", false, "Hide image preview")
	block := flag.Bool("block", false, "Block on send")
	device := flag.Int("dev", 0, "Camera device")
	out := flag.String("out", "", "Output path")
	ext := flag.String("ext", "png", "Output file extension")
	file := flag.String("file", "", "Video file")
	width := flag.Int("width", 640, "Width")
	height := flag.Int("height", 480, "Height")
	imagesPath := flag.String("images", "", "Path to either image list or folder with images")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	flag.Parse()

	if *help {
		flag.PrintDefaults()
		return
	}

	pipe := pipeline.New()

	var input pipeline.Step
	var output pipeline.Step
	var sink pipeline.Step
	var err error

	opts := []plugin.Option{}

	if len(*imagesPath) != 0 {
		paths := ReadFileList(*imagesPath)
		opts = append(opts, reader.WithReaderGoCV(paths))
	} else if len(*file) != 0 {
		opts = append(opts, reader.WithVideoReaderGoCV(*file))
	} else {
		opts = append(opts, reader.WithDeviceReaderGoCVResolution(*device, *width, *height))
	}

	opts = append(opts, plugin.WithBlocking(*block))
	input, err = steps.NewReader(opts...)
	if err != nil {
		panic(err)
	}

	fps, err := fps.New(fps.WithNumFrames(10), plugin.WithBlocking(*block))
	if err != nil {
		panic(err)
	}

	fout, err := steps.NewFanOut(plugin.WithBlocking(*block), plugin.WithClose(false))
	if err != nil {
		panic(err)
	}

	if !*hide {
		output, err = display.NewGoCV(
			plugin.WithClose(*out != ""),
			display.WithKey(store.IMAGE),
			plugin.WithBlocking(*block),
		)
		if err != nil {
			panic(err)
		}
	}

	keys, err := pipe.ConnectSteps(input, fps, fout, output)
	if err != nil {
		panic(err)
	}

	if *out == "" {
		sink, err = writer.NewNull(plugin.WithClose(*hide))
	} else {
		sink, err = writer.NewGoCV(plugin.WithClose(*hide), writer.WithKey(store.IMAGE), writer.WithExtension(*ext), writer.WithPrefix(*out))
	}
	_, err = pipe.ConnectSteps(fout, sink)
	if err != nil {
		panic(err)
	}

	pipe.Run(ctx)

	for data := range keys {
		if *hide {
			continue
		}

		v, ok := data.Get(store.USER_KEY_CODE)
		if !ok {
			continue
		}

		key := v.(int)

		if key != -1 {
			fmt.Println(key)
		}
		if key == 27 {
			cancel()
			time.Sleep(time.Second * 3)
			return
		}
	}
}

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
