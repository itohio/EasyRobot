package csv

import (
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

type withHeader bool

func (h withHeader) Apply(opts *types.Options) {
	if opts.Metadata == nil {
		opts.Metadata = make(map[string]string)
	}
	if bool(h) {
		opts.Metadata["header"] = "true"
	} else {
		opts.Metadata["header"] = "false"
	}
}

// WithHeader enables/disables CSV header row.
// When enabled, the first row contains column names.
func WithHeader(enabled bool) types.Option {
	return withHeader(enabled)
}

type withDataType types.DataType

func (dt withDataType) Apply(opts *types.Options) {
	if opts.Metadata == nil {
		opts.Metadata = make(map[string]string)
	}
	// Store in metadata so it can be checked after comment parsing
	opts.Metadata["dtype"] = dataTypeToString(types.DataType(dt))
	// Also set DestinationType directly
	opts.DestinationType = types.DataType(dt)
}

// dataTypeToString converts DataType to string (helper for options).
func dataTypeToString(dtype types.DataType) string {
	switch dtype {
	case types.FP32:
		return "FP32"
	case types.FP64:
		return "FP64"
	case types.INT32:
		return "INT32"
	case types.INT16:
		return "INT16"
	case types.INT8:
		return "INT8"
	case types.INT64:
		return "INT64"
	case types.UINT8:
		return "UINT8"
	default:
		return "UNKNOWN"
	}
}

// WithDataType sets the data type hint for unmarshalling.
// This is useful when CSV data is ambiguous (e.g., all integers could be int32 or float32).
// If not specified, unmarshaller will infer from data or use default (FP32).
// This option takes precedence over dtype comments in the CSV file.
func WithDataType(dtype types.DataType) types.Option {
	return withDataType(dtype)
}

type withOrientation string

func (o withOrientation) Apply(opts *types.Options) {
	if opts.Metadata == nil {
		opts.Metadata = make(map[string]string)
	}
	opts.Metadata["orientation"] = string(o)
}

// WithOrientation sets how vectors/arrays should be read from CSV.
// "row" (default): Read from a single row (horizontal)
// "column": Read from a single column (vertical)
func WithOrientation(orientation string) types.Option {
	if orientation != "row" && orientation != "column" {
		orientation = "row" // default
	}
	return withOrientation(orientation)
}
