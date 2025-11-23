package internal

type Signed interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64
}

type Unsigned interface {
	~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

// Integer is a constraint for all integer types.
type Integer interface {
	Signed | Unsigned
}

type Float interface {
	~float64 | ~float32
}

type NumberComplex interface {
	~complex64
}

type NumberReal interface {
	Float | Signed
}

// Number types that can be used with generic operations
type Number interface {
	NumberReal | NumberComplex
}

// Types that need clamping when converting to int8 (sorted by type size)
type ClampableToInt8 interface {
	~float64 | ~int64 | ~float32 | ~int | ~int32 | ~int16
}

// Types that need clamping when converting to int16 (sorted by type size)
type ClampableToInt16 interface {
	~float64 | ~int64 | ~float32 | ~int | ~int32
}

// Types that need clamping when converting to int32 (sorted by type size)
type ClampableToInt32 interface {
	~float64 | ~int64 | ~float32 | ~int
}

// Types that need clamping when converting to int64 (sorted by type size)
type ClampableToInt64 interface {
	~float64 | ~float32
}
