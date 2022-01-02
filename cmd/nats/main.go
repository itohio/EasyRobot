package main

import (
	"context"
	"flag"
	"fmt"
	"time"

	"github.com/itohio/EasyRobot/pkg/core/options"
	"github.com/itohio/EasyRobot/pkg/core/pipeline"
	"github.com/itohio/EasyRobot/pkg/core/pipeline/steps"
	"github.com/itohio/EasyRobot/pkg/core/pipeline/steps/fps"
	"github.com/itohio/EasyRobot/pkg/core/plugin"
	"github.com/itohio/EasyRobot/pkg/core/store"
	"github.com/itohio/EasyRobot/pkg/core/transport/nats"
	"github.com/itohio/EasyRobot/pkg/vision/display"
	"github.com/itohio/EasyRobot/pkg/vision/reader"
	"github.com/itohio/EasyRobot/pkg/vision/writer"
	natsgo "github.com/nats-io/nats.go"
)

func main() {
	help := flag.Bool("help", false, "Help")
	hide := flag.Bool("hide", false, "Hide image preview")
	block := flag.Bool("block", false, "Block on send")
	device := flag.Int("dev", 0, "Camera device")
	imagesPath := flag.String("images", "", "Path to either image list or folder with images")
	file := flag.String("file", "", "Video file")
	out := flag.String("out", "", "Output path")
	ext := flag.String("ext", "png", "Output file extension")
	width := flag.Int("width", 640, "Width")
	height := flag.Int("height", 480, "Height")
	natsUrls := flag.String("urls", natsgo.DefaultURL, "The nats server URLs (separated by comma)")
	natsCreds := flag.String("creds", "", "User Credentials File")
	subscribe := flag.String("subscribe", "easyrobot.camera", "Topic to subscribe")
	publish := flag.String("publish", "easyrobot.camera", "Topic to publish")
	consume := flag.Bool("consume", false, "Consume topic")

	flag.Parse()

	if *help {
		flag.PrintDefaults()
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pipe := pipeline.New()

	ncStep, err := nats.NewGoCV(
		nats.WithUrls(*natsUrls),
		nats.WithCredentials(*natsCreds),
		nats.WithPublish(*publish),
		nats.WithPublish(*subscribe),
	)
	if err != nil {
		panic(err)
	}

	if *consume {
		setupConsumer(ctx, pipe, ncStep, *out, *ext, *hide)
	} else {
		setupProducer(ctx, pipe, ncStep, *width, *height, *device, *file, *imagesPath)
	}

	var input pipeline.Step
	var output pipeline.Step
	var sink pipeline.Step
	var err error

	opts := []options.Option{}

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
