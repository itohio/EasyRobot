package gocv

type ownershipMode uint8

const (
	ownershipClone ownershipMode = iota
	ownershipAdopt
	ownershipShare
)

type constructorConfig struct {
	ownership ownershipMode
}

// Option customises tensor construction from an existing gocv.Mat.
type Option func(*constructorConfig)

// WithAdoptedMat creates a tensor that takes ownership of the provided Mat
// without cloning. The caller must not use or Close the mat after passing it
// to FromMat.
func WithAdoptedMat() Option {
	return func(cfg *constructorConfig) {
		cfg.ownership = ownershipAdopt
	}
}

// WithSharedMat creates a tensor that references the provided Mat without
// taking ownership. Release becomes a no-op and the caller remains responsible
// for closing the Mat.
func WithSharedMat() Option {
	return func(cfg *constructorConfig) {
		cfg.ownership = ownershipShare
	}
}

// WithClonedMat forces cloning of the incoming Mat even when other options are
// provided. This is the default behaviour.
func WithClonedMat() Option {
	return func(cfg *constructorConfig) {
		cfg.ownership = ownershipClone
	}
}
