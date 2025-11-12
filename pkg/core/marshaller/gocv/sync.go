package gocv

import (
	"context"
	"sync"
	"time"

	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
)

type frameItem struct {
	tensors  []types.Tensor
	metadata map[string]any
}

type sourceStream interface {
	Next(ctx context.Context) (frameItem, bool, error)
	Close() error
}

func newFrameStream(baseCtx context.Context, streams []sourceStream, allowBestEffort bool, sequential bool) (types.FrameStream, error) {
	ctx, cancel := context.WithCancel(baseCtx)
	output := make(chan types.Frame)
	var once sync.Once
	done := func() {
		once.Do(func() {
			cancel()
			for _, stream := range streams {
				if stream != nil {
					_ = stream.Close()
				}
			}
		})
	}

	go func() {
		defer close(output)
		defer done()

		if len(streams) == 0 {
			return
		}

		index := 0

		if sequential {
			for streamIdx, stream := range streams {
				for {
					select {
					case <-ctx.Done():
						return
					default:
					}

					item, cont, err := stream.Next(ctx)
					if err != nil {
						output <- errorFrame(err, index)
						return
					}
					if !cont {
						break
					}

					frame := combineFrameItems([]frameItem{item}, index)
					frame.Metadata["stream_index"] = streamIdx
					output <- frame
					index++
				}
			}
			return
		}

		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			items := make([]frameItem, len(streams))
			ok := true

			for i, stream := range streams {
				item, cont, err := stream.Next(ctx)
				if err != nil {
					output <- errorFrame(err, index)
					return
				}
				if !cont {
					ok = false
					break
				}
				items[i] = item
			}

			if !ok {
				return
			}

			frame := combineFrameItems(items, index)
			output <- frame
			index++
		}
	}()

	return types.NewFrameStream(output, done), nil
}

func combineFrameItems(items []frameItem, index int) types.Frame {
	timestamp := time.Now().UnixNano()
	meta := map[string]any{}
	var tensors []types.Tensor
	sources := make([]map[string]any, 0, len(items))

	for _, item := range items {
		if len(item.tensors) > 0 {
			tensors = append(tensors, item.tensors...)
		}
		if len(item.metadata) > 0 {
			copied := make(map[string]any, len(item.metadata))
			for k, v := range item.metadata {
				copied[k] = v
			}
			sources = append(sources, copied)
		} else {
			sources = append(sources, map[string]any{})
		}
	}

	if len(sources) == 1 {
		for k, v := range sources[0] {
			meta[k] = v
		}
	} else {
		meta["sources"] = sources
	}

	return types.Frame{
		Index:     index,
		Timestamp: timestamp,
		Metadata:  meta,
		Tensors:   tensors,
	}
}

func errorFrame(err error, index int) types.Frame {
	return types.Frame{
		Index:     index,
		Timestamp: time.Now().UnixNano(),
		Metadata: map[string]any{
			"error": err,
		},
	}
}
