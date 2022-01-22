package main

import (
	"context"

	"github.com/itohio/EasyRobot/pkg/core/options"
	"github.com/itohio/EasyRobot/pkg/core/pipeline"
	"github.com/itohio/EasyRobot/pkg/core/pipeline/steps"
	"github.com/itohio/EasyRobot/pkg/core/pipeline/steps/fps"
	"github.com/itohio/EasyRobot/pkg/core/plugin"
	"github.com/itohio/EasyRobot/pkg/core/store"
	"github.com/itohio/EasyRobot/pkg/vision/display"
	"github.com/itohio/EasyRobot/pkg/vision/reader"
	"github.com/itohio/EasyRobot/pkg/vision/writer"
)

func setupProducer(ctx context.Context, pipe pipeline.Pipeline, nc pipeline.Step, width, height, device int, file, imagesPath, out, ext string, block, hide bool) (<-chan store.Store, error) {
	var input pipeline.Step
	var output pipeline.Step
	var sink pipeline.Step
	var err error

	opts := []options.Option{}

	if len(imagesPath) != 0 {
		paths := ReadFileList(imagesPath)
		opts = append(opts, reader.WithReaderGoCV(paths))
	} else if len(file) != 0 {
		opts = append(opts, reader.WithVideoReaderGoCV(file))
	} else {
		opts = append(opts, reader.WithDeviceReaderGoCVResolution(device, width, height))
	}

	opts = append(opts, plugin.WithBlocking(block))
	input, err = steps.NewReader(opts...)
	if err != nil {
		panic(err)
	}

	fps, err := fps.New(fps.WithNumFrames(10), plugin.WithBlocking(block))
	if err != nil {
		panic(err)
	}

	fout, err := steps.NewFanOut(plugin.WithBlocking(block), plugin.WithClose(false))
	if err != nil {
		panic(err)
	}

	if !hide {
		output, err = display.NewGoCV(
			plugin.WithClose(out != ""),
			display.WithKey(store.IMAGE),
			plugin.WithBlocking(block),
		)
		if err != nil {
			panic(err)
		}
	}

	keys, err := pipe.ConnectSteps(input, fps, fout, output)
	if err != nil {
		panic(err)
	}

	_, err = pipe.ConnectSteps(fout, nc)
	if err != nil {
		panic(err)
	}

	if out == "" {
		sink, err = writer.NewNull(plugin.WithClose(hide))
	} else {
		sink, err = writer.NewGoCV(plugin.WithClose(hide), writer.WithKey(store.IMAGE), writer.WithExtension(ext), writer.WithPrefix(out))
	}
	_, err = pipe.ConnectSteps(fout, sink)
	if err != nil {
		panic(err)
	}
	return keys, nil
}
