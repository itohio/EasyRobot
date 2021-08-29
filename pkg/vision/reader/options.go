package reader

import "github.com/foxis/EasyRobot/pkg/core/options"

type readerOpts struct {
	paths  []string
	id     int
	fname  string
	width  int
	height int
	index  int
}

func WithPaths(paths []string) options.Option {
	return func(opt interface{}) {
		if o, ok := opt.(*readerOpts); ok {
			o.paths = paths
		}
	}
}
func WithId(id int) options.Option {
	return func(opt interface{}) {
		if o, ok := opt.(*readerOpts); ok {
			o.id = id
		}
	}
}
func WithFileName(fname string) options.Option {
	return func(opt interface{}) {
		if o, ok := opt.(*readerOpts); ok {
			o.fname = fname
		}
	}
}
func WithResolution(width, height int) options.Option {
	return func(opt interface{}) {
		if o, ok := opt.(*readerOpts); ok {
			o.width = width
			o.height = height
		}
	}
}
