package types

import "github.com/itohio/EasyRobot/x/types/internal"

type NumberComplex = internal.NumberComplex

type NumberReal = internal.NumberReal

// Number types that can be used with generic operations
type Number = internal.Number

type Integer = internal.Integer

type Float = internal.Float

type Signed = internal.Signed

type Unsigned = internal.Unsigned

// Types that need clamping when converting to int8 (sorted by type size)
type ClampableToInt8 = internal.ClampableToInt8

// Types that need clamping when converting to int16 (sorted by type size)
type ClampableToInt16 = internal.ClampableToInt16

// Types that need clamping when converting to int32 (sorted by type size)
type ClampableToInt32 = internal.ClampableToInt32

// Types that need clamping when converting to int64 (sorted by type size)
type ClampableToInt64 = internal.ClampableToInt64

type ClampableToInt = internal.ClampableToInt
