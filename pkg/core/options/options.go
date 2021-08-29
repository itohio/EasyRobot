package options

type Option func(cfg interface{})

/// Apply options using option funcs
func ApplyOptions(optionsStructPtr interface{}, opts ...Option) {
	for _, opt := range opts {
		opt(optionsStructPtr)
	}
}
