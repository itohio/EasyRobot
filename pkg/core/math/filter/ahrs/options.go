package ahrs

type Options struct {
	GainP, GainI    float32
	HasAccelerator  bool
	HasGyroscope    bool
	HasMagnetometer bool
}

type Option func(cfg *Options)

func defaultOptions() Options {
	return Options{
		GainP:          0.1,
		GainI:          0.1,
		HasAccelerator: true,
		HasGyroscope:   true,
	}
}

func applyOptions(optionsStructPtr *Options, opts ...Option) {
	for _, opt := range opts {
		opt(optionsStructPtr)
	}
}

func WithKP(kp float32) Option {
	return func(o *Options) {
		o.GainP = kp
	}
}

func WithKI(ki float32) Option {
	return func(o *Options) {
		o.GainI = ki
	}
}

func WithAccelerator(b bool) Option {
	return func(o *Options) {
		o.HasAccelerator = b
	}
}

func WithGyroscope(b bool) Option {
	return func(o *Options) {
		o.HasGyroscope = b
	}
}

func WithMagnetometer(b bool) Option {
	return func(o *Options) {
		o.HasMagnetometer = b
	}
}
