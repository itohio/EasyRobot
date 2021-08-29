package plugin

import "github.com/foxis/EasyRobot/pkg/core/options"

type Options struct {
	Name         string `opts:"name"`
	Blocking     bool   `opts:"block"`
	Enabled      bool   `opts:"enable"`
	BufferSize   int    `opts:"buf_size"`
	NoEOS        bool   `opts:"no_eos"`
	IgnoreErrors bool   `opts:"ignore_err"`
	Close        bool   `opts:"close"`
}

/// Create options struct with default values
func DefaultOptions() Options {
	return Options{
		Enabled: true,
		Close:   true,
	}
}

/// Set plugin Name
func WithName(name string) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.Name = name
		}
	}
}

/// Set buffer size for sync/join
func WithBufferSize(size int) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.BufferSize = size
		}
	}
}

/// Sets whether output should be blocking.
/// Non blocking will skip data.
func WithBlocking(blocking bool) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.Blocking = blocking
		}
	}
}

/// Skip processing
func WithEnable(b bool) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.Enabled = b
		}
	}
}

/// Stop the step if End Of Stream is detected
func WithEOSExit(exit bool) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.NoEOS = !exit
		}
	}
}

/// Ignore errors
func WithIgnoreErrors(ignore bool) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.IgnoreErrors = ignore
		}
	}
}

/// Update Options struct with values inside the map.
/// Usecase: write/read options to JSON.
func WithMapping(opt map[string]interface{}) options.Option {
	return func(o interface{}) {
		fillStruct(o, opt)
	}
}

/// Close keys where supported(display/sink/bridge)
func WithClose(close bool) options.Option {
	return func(o interface{}) {
		if opt, ok := o.(*Options); ok {
			opt.Close = close
		}
	}
}
