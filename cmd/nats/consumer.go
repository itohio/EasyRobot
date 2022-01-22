package main

import (
	"context"

	"github.com/itohio/EasyRobot/pkg/core/pipeline"
	"github.com/itohio/EasyRobot/pkg/core/pipeline/steps"
	"github.com/itohio/EasyRobot/pkg/core/pipeline/steps/fps"
	"github.com/itohio/EasyRobot/pkg/core/plugin"
	"github.com/itohio/EasyRobot/pkg/core/store"
	"github.com/itohio/EasyRobot/pkg/vision/display"
	"github.com/itohio/EasyRobot/pkg/vision/writer"
)

func setupConsumer(ctx context.Context, pipe pipeline.Pipeline, nc pipeline.Step, out, ext string, block, hide bool) (<-chan store.Store, error) {
	fps, err := fps.New(fps.WithNumFrames(10), plugin.WithBlocking(block))
	if err != nil {
		panic(err)
	}

	fout, err := steps.NewFanOut(plugin.WithBlocking(block), plugin.WithClose(false))
	if err != nil {
		panic(err)
	}

	output, err := display.NewGoCV(
		plugin.WithClose(out != ""),
		display.WithKey(store.IMAGE),
		plugin.WithBlocking(block),
	)
	if err != nil {
		panic(err)
	}

	keys, err := pipe.ConnectSteps(nc, fps, fout, output)
	if err != nil {
		panic(err)
	}

	var sink pipeline.Step
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
