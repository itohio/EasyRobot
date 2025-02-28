package writer

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"strings"

	"github.com/itohio/EasyRobot/pkg/core/options"
	"github.com/itohio/EasyRobot/pkg/core/pipeline"
	"github.com/itohio/EasyRobot/pkg/core/pipeline/steps"
	"github.com/itohio/EasyRobot/pkg/core/plugin"
	"github.com/itohio/EasyRobot/pkg/core/store"
)

const IMAGES_NAME = "wimg"

func init() {
	pipeline.Register(IMAGES_NAME, NewImages)
}

func NewImages(opts ...options.Option) (pipeline.Step, error) {
	o := writerOpts{
		base: plugin.DefaultOptions(),
		ext:  "png",
	}
	options.ApplyOptions(&o, opts...)
	options.ApplyOptions(&o.base, opts...)
	newOpts := opts
	newOpts = append(newOpts, WithImagesWriter(o.prefix, o.ext, o.keys))
	return steps.NewSink(newOpts...)
}

func WithImagesWriter(prefix, extension string, keys []store.FQDNType) options.Option {
	return steps.WithNamedSinkFunc(IMAGES_NAME, sink_images(writerOpts{prefix: prefix, ext: extension, keys: keys}))
}

func sink_images(w writerOpts) steps.SinkFunc {
	sink := sink_null(w.keys)

	return func(data store.Store) error {
		index, err := data.Index()
		if err != nil {
			return nil
		}

		for _, key := range w.keys {
			val, ok := data.Get(key)
			if !ok {
				return nil
			}

			path := fmt.Sprintf("%s%s-%08d.%s", w.prefix, strings.ToLower(key.String()), index, w.ext)

			if wr, ok := val.(store.ValueWriter); ok {
				out, err := os.Create(path)
				if err != nil {
					return err
				}
				defer out.Close()

				return wr.Write(out, path)
			}

			img, ok := val.(*image.Image)
			if !ok {
				return nil
			}

			out, err := os.Create(path)
			if err != nil {
				return err
			}
			defer out.Close()

			switch w.ext {
			case "png":
				png.Encode(out, *img)
			case "jpg":
				fallthrough
			case "jpeg":
				jpeg.Encode(out, *img, &jpeg.Options{Quality: 90})
			}
		}

		if w.base.Close {
			return sink(data)
		}
		return nil
	}
}
