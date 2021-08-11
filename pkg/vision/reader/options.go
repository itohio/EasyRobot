package reader

import "github.com/foxis/EasyRobot/pkg/core/plugin"

type readerOpts struct {
	paths  []string
	id     int
	fname  string
	width  int
	height int
	index  int
}

func WithPaths(paths []string) plugin.Option {
	return func(opt interface{}) {
		if o, ok := opt.(*readerOpts); ok {
			o.paths = paths
		}
	}
}
func WithId(id int) plugin.Option {
	return func(opt interface{}) {
		if o, ok := opt.(*readerOpts); ok {
			o.id = id
		}
	}
}
func WithFileName(fname string) plugin.Option {
	return func(opt interface{}) {
		if o, ok := opt.(*readerOpts); ok {
			o.fname = fname
		}
	}
}
func WithResolution(width, height int) plugin.Option {
	return func(opt interface{}) {
		if o, ok := opt.(*readerOpts); ok {
			o.width = width
			o.height = height
		}
	}
}
