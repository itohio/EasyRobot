package writer

import (
	"encoding/binary"
	"fmt"
	"image/jpeg"
	"image/png"
	"io"
	"os"
	"strings"

	. "github.com/foxis/EasyRobot/pkg/core/logger"
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"

	"gocv.io/x/gocv"
)

const GOCV_NAME = "wicv"

func init() {
	pipeline.Register(GOCV_NAME, NewGoCV)
}

func NewGoCV(opts ...plugin.Option) (pipeline.Step, error) {
	o := writerOpts{
		base: plugin.DefaultOptions(),
		ext:  "png",
	}
	plugin.ApplyOptions(&o, opts...)
	plugin.ApplyOptions(&o.base, opts...)
	newOpts := opts
	newOpts = append(newOpts, WithWriterGoCV(o.prefix, o.ext, o.keys))
	return steps.NewSink(newOpts...)
}

func WithWriterGoCV(prefix, extension string, keys []store.FQDNType) plugin.Option {
	return steps.WithNamedSinkFunc(GOCV_NAME, sink_gocv(writerOpts{prefix: prefix, ext: extension, keys: keys}))
}

func sink_gocv(w writerOpts) steps.SinkFunc {
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
			Log.Debug().Str("path", path).Msg("write")
			if wr, ok := val.(store.ValueWriter); ok {
				out, err := os.Create(path)
				if err != nil {
					Log.Error().Str("path", path).Err(err).Msg("write")
					return err
				}
				defer out.Close()

				return wr.Write(out, path)
			}

			mat, ok := val.(*gocv.Mat)
			if !ok || mat == nil {
				return nil
			}

			out, err := os.Create(path)
			if err != nil {
				return err
			}
			defer out.Close()

			switch w.ext {
			case "png":
				img, err := mat.ToImage()
				if err != nil {
					return nil
				}
				png.Encode(out, img)
			case "jpg":
				fallthrough
			case "jpeg":
				img, err := mat.ToImage()
				if err != nil {
					return nil
				}
				jpeg.Encode(out, img, &jpeg.Options{Quality: 90})
			case "mat":
				write_mat(out, *mat)
			}
		}

		if w.base.Close {
			return sink(data)
		}
		return nil
	}
}

func write_mat(out io.Writer, mat gocv.Mat) {
	var (
		magic uint64 = 0xabcdef0012345678
		sizes []int  = mat.Size()
		bytes []byte = mat.ToBytes()
	)

	if err := binary.Write(out, binary.LittleEndian, magic); err != nil {
		return
	}
	if err := binary.Write(out, binary.LittleEndian, uint8(len(sizes))); err != nil {
		return
	}
	if err := binary.Write(out, binary.LittleEndian, sizes); err != nil {
		return
	}
	if err := binary.Write(out, binary.LittleEndian, mat.Type()); err != nil {
		return
	}
	if err := binary.Write(out, binary.LittleEndian, uint64(len(bytes))); err != nil {
		return
	}
	if err := binary.Write(out, binary.LittleEndian, bytes); err != nil {
		return
	}
}
