package main

import (
	"context"
	"flag"
	"fmt"
	"time"

	"github.com/itohio/EasyRobot/pkg/core/pipeline"
	"github.com/itohio/EasyRobot/pkg/core/store"
	"github.com/itohio/EasyRobot/pkg/core/transport/nats"
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

	var keys <-chan store.Store
	if *consume {
		keys, err = setupConsumer(ctx, pipe, ncStep, *out, *ext, *block, *hide)
	} else {
		keys, err = setupProducer(ctx, pipe, ncStep, *width, *height, *device, *file, *imagesPath, *out, *ext, *block, *hide)
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
