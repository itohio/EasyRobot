package csv

import (
	"encoding/csv"
	"fmt"
	"io"
	"reflect"
	"strconv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// dataTypeCommentPrefix is the prefix for data type comments in CSV files.
const dataTypeCommentPrefix = "# dtype:"

// Marshaller implements CSV-based marshalling.
type Marshaller struct {
	opts types.Options
}

// NewMarshaller creates a new CSV marshaller.
func NewMarshaller(opts ...types.Option) *Marshaller {
	m := &Marshaller{
		opts: types.Options{},
	}
	for _, opt := range opts {
		opt.Apply(&m.opts)
	}
	return m
}

// Format returns the format name.
func (m *Marshaller) Format() string {
	return "csv"
}

// Marshal encodes value to CSV format.
func (m *Marshaller) Marshal(w io.Writer, value any, opts ...types.Option) error {
	// Apply additional options
	localOpts := m.opts
	for _, opt := range opts {
		opt.Apply(&localOpts)
	}

	if value == nil {
		return types.NewError("marshal", "csv", "nil value", nil)
	}

	// Write data type comment if tensor/matrix/vector and dtype is known
	dtype := m.detectDataType(value)
	if dtype != types.DT_UNKNOWN {
		comment := fmt.Sprintf("%s %s\n", dataTypeCommentPrefix, dtypeToString(dtype))
		if _, err := w.Write([]byte(comment)); err != nil {
			return types.NewError("marshal", "csv", "failed to write dtype comment", err)
		}
	}

	writer := csv.NewWriter(w)
	defer writer.Flush()

	// Check for header option
	hasHeader := localOpts.Metadata != nil && localOpts.Metadata["header"] == "true"

	// Type switch for domain objects
	switch v := value.(type) {
	case types.Tensor:
		return m.marshalTensor(writer, v, hasHeader)
	case matTypes.Matrix:
		return m.marshalMatrix(writer, v, hasHeader)
	case vecTypes.Vector:
		return m.marshalVector(writer, v, hasHeader, localOpts)
	}

	// Check for slices/arrays
	rv := reflect.ValueOf(value)
	if rv.Kind() == reflect.Slice || rv.Kind() == reflect.Array {
		return m.marshalSlice(writer, rv, hasHeader, localOpts)
	}

	return types.NewError("marshal", "csv", fmt.Sprintf("unsupported type: %T", value), nil)
}

// marshalTensor marshals a tensor to CSV (1D/2D only).
func (m *Marshaller) marshalTensor(writer *csv.Writer, t types.Tensor, hasHeader bool) error {
	if t.Empty() {
		return types.NewError("marshal", "csv", "empty tensor", nil)
	}

	shape := t.Shape()
	if len(shape) > 2 {
		return types.NewError("marshal", "csv", "tensor must be 1D or 2D for CSV", nil)
	}

	data := t.Data()
	if data == nil {
		return types.NewError("marshal", "csv", "tensor data is nil", nil)
	}

	// Type assert based on dtype
	var dataSlice []float32
	switch t.DataType() {
	case types.FP32:
		var ok bool
		dataSlice, ok = data.([]float32)
		if !ok {
			return types.NewError("marshal", "csv", "tensor data is not []float32", nil)
		}
	default:
		return types.NewError("marshal", "csv", fmt.Sprintf("unsupported tensor dtype: %v", t.DataType()), nil)
	}

	if len(shape) == 1 {
		// 1D tensor: single row
		if hasHeader {
			header := make([]string, shape[0])
			for i := 0; i < shape[0]; i++ {
				header[i] = fmt.Sprintf("col_%d", i)
			}
			if err := writer.Write(header); err != nil {
				return types.NewError("marshal", "csv", "failed to write header", err)
			}
		}
		// Write as single row
		row := make([]string, shape[0])
		for i := 0; i < shape[0]; i++ {
			row[i] = formatFloat32(dataSlice[i])
		}
		if err := writer.Write(row); err != nil {
			return types.NewError("marshal", "csv", "failed to write row", err)
		}
		return nil
	}

	// 2D tensor: rows x cols
	rows := shape[0]
	cols := shape[1]

	// Write header if requested
	if hasHeader {
		header := make([]string, cols)
		for i := 0; i < cols; i++ {
			header[i] = fmt.Sprintf("col_%d", i)
		}
		if err := writer.Write(header); err != nil {
			return types.NewError("marshal", "csv", "failed to write header", err)
		}
	}

	// Write data rows
	for i := 0; i < rows; i++ {
		row := make([]string, cols)
		for j := 0; j < cols; j++ {
			row[j] = formatFloat32(dataSlice[i*cols+j])
		}
		if err := writer.Write(row); err != nil {
			return types.NewError("marshal", "csv", fmt.Sprintf("failed to write row %d", i), err)
		}
	}

	return nil
}

// marshalMatrix marshals a matrix to CSV.
func (m *Marshaller) marshalMatrix(writer *csv.Writer, mat matTypes.Matrix, hasHeader bool) error {
	rows := mat.Rows()
	cols := mat.Cols()

	if rows == 0 || cols == 0 {
		return types.NewError("marshal", "csv", "empty matrix", nil)
	}

	// Write header if requested
	if hasHeader {
		header := make([]string, cols)
		for i := 0; i < cols; i++ {
			header[i] = fmt.Sprintf("col_%d", i)
		}
		if err := writer.Write(header); err != nil {
			return types.NewError("marshal", "csv", "failed to write header", err)
		}
	}

	// Write data rows
	for i := 0; i < rows; i++ {
		rowVec := mat.Row(i)
		rowVecView := rowVec.View().(vec.Vector)
		row := make([]string, cols)
		for j := 0; j < cols; j++ {
			row[j] = formatFloat32(rowVecView[j])
		}
		if err := writer.Write(row); err != nil {
			return types.NewError("marshal", "csv", fmt.Sprintf("failed to write row %d", i), err)
		}
	}

	return nil
}

// marshalVector marshals a vector to CSV (single row by default, or single column if orientation=column).
func (m *Marshaller) marshalVector(writer *csv.Writer, v vecTypes.Vector, hasHeader bool, opts types.Options) error {
	length := v.Len()
	if length == 0 {
		return types.NewError("marshal", "csv", "empty vector", nil)
	}

	vecView := v.View().(vec.Vector)

	// Check orientation option
	orientation := "row" // default
	if opts.Metadata != nil {
		if o, ok := opts.Metadata["orientation"]; ok {
			orientation = o
		}
	}

	if orientation == "column" {
		// Write as single column (vertical)
		if hasHeader {
			if err := writer.Write([]string{"value"}); err != nil {
				return types.NewError("marshal", "csv", "failed to write header", err)
			}
		}
		// Write each value as a row
		for i := 0; i < length; i++ {
			if err := writer.Write([]string{formatFloat32(vecView[i])}); err != nil {
				return types.NewError("marshal", "csv", fmt.Sprintf("failed to write row %d", i), err)
			}
		}
		return nil
	}

	// Default: Write as single row (horizontal)
	// Write header if requested
	if hasHeader {
		header := make([]string, length)
		for i := 0; i < length; i++ {
			header[i] = fmt.Sprintf("col_%d", i)
		}
		if err := writer.Write(header); err != nil {
			return types.NewError("marshal", "csv", "failed to write header", err)
		}
	}

	// Write data as single row
	row := make([]string, length)
	for i := 0; i < length; i++ {
		row[i] = formatFloat32(vecView[i])
	}
	if err := writer.Write(row); err != nil {
		return types.NewError("marshal", "csv", "failed to write row", err)
	}

	return nil
}

// marshalSlice marshals a slice/array to CSV (single row by default, or single column if orientation=column).
func (m *Marshaller) marshalSlice(writer *csv.Writer, rv reflect.Value, hasHeader bool, opts types.Options) error {
	length := rv.Len()
	if length == 0 {
		return types.NewError("marshal", "csv", "empty slice", nil)
	}

	// Determine element type
	elemType := rv.Type().Elem()
	kind := elemType.Kind()

	// Check orientation option
	orientation := "row" // default
	if opts.Metadata != nil {
		if o, ok := opts.Metadata["orientation"]; ok {
			orientation = o
		}
	}

	if orientation == "column" {
		// Write as single column (vertical)
		if hasHeader {
			if err := writer.Write([]string{"value"}); err != nil {
				return types.NewError("marshal", "csv", "failed to write header", err)
			}
		}

		// Write data based on element type
		switch kind {
		case reflect.Float32:
			values := rv.Interface().([]float32)
			return m.writeFloat32Column(writer, values)
		case reflect.Float64:
			values := rv.Interface().([]float64)
			return m.writeFloat64Column(writer, values)
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			return m.writeIntColumn(writer, rv)
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			return m.writeUintColumn(writer, rv)
		case reflect.String:
			values := rv.Interface().([]string)
			return m.writeStringColumn(writer, values)
		default:
			return types.NewError("marshal", "csv", fmt.Sprintf("unsupported slice element type: %s", kind), nil)
		}
	}

	// Default: Write as single row (horizontal)
	// Write header if requested
	if hasHeader {
		header := make([]string, length)
		for i := 0; i < length; i++ {
			header[i] = fmt.Sprintf("col_%d", i)
		}
		if err := writer.Write(header); err != nil {
			return types.NewError("marshal", "csv", "failed to write header", err)
		}
	}

	// Write data as single row
	row := make([]string, length)
	switch kind {
	case reflect.Float32:
		values := rv.Interface().([]float32)
		for i, val := range values {
			row[i] = formatFloat32(val)
		}
	case reflect.Float64:
		values := rv.Interface().([]float64)
		for i, val := range values {
			row[i] = formatFloat64(val)
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		for i := 0; i < length; i++ {
			row[i] = strconv.FormatInt(rv.Index(i).Int(), 10)
		}
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		for i := 0; i < length; i++ {
			row[i] = strconv.FormatUint(rv.Index(i).Uint(), 10)
		}
	case reflect.String:
		values := rv.Interface().([]string)
		copy(row, values)
	default:
		return types.NewError("marshal", "csv", fmt.Sprintf("unsupported slice element type: %s", kind), nil)
	}

	if err := writer.Write(row); err != nil {
		return types.NewError("marshal", "csv", "failed to write row", err)
	}

	return nil
}

// writeFloat32Column writes a float32 column.
func (m *Marshaller) writeFloat32Column(writer *csv.Writer, values []float32) error {
	for _, val := range values {
		if err := writer.Write([]string{formatFloat32(val)}); err != nil {
			return types.NewError("marshal", "csv", "failed to write row", err)
		}
	}
	return nil
}

// writeFloat64Column writes a float64 column.
func (m *Marshaller) writeFloat64Column(writer *csv.Writer, values []float64) error {
	for _, val := range values {
		if err := writer.Write([]string{formatFloat64(val)}); err != nil {
			return types.NewError("marshal", "csv", "failed to write row", err)
		}
	}
	return nil
}

// writeIntColumn writes an integer column.
func (m *Marshaller) writeIntColumn(writer *csv.Writer, rv reflect.Value) error {
	for i := 0; i < rv.Len(); i++ {
		val := rv.Index(i).Int()
		if err := writer.Write([]string{strconv.FormatInt(val, 10)}); err != nil {
			return types.NewError("marshal", "csv", "failed to write row", err)
		}
	}
	return nil
}

// writeUintColumn writes an unsigned integer column.
func (m *Marshaller) writeUintColumn(writer *csv.Writer, rv reflect.Value) error {
	for i := 0; i < rv.Len(); i++ {
		val := rv.Index(i).Uint()
		if err := writer.Write([]string{strconv.FormatUint(val, 10)}); err != nil {
			return types.NewError("marshal", "csv", "failed to write row", err)
		}
	}
	return nil
}

// writeStringColumn writes a string column.
func (m *Marshaller) writeStringColumn(writer *csv.Writer, values []string) error {
	for _, val := range values {
		if err := writer.Write([]string{val}); err != nil {
			return types.NewError("marshal", "csv", "failed to write row", err)
		}
	}
	return nil
}

// formatFloat32 formats a float32 value for CSV.
func formatFloat32(val float32) string {
	return strconv.FormatFloat(float64(val), 'g', -1, 32)
}

// formatFloat64 formats a float64 value for CSV.
func formatFloat64(val float64) string {
	return strconv.FormatFloat(val, 'g', -1, 64)
}

// detectDataType detects the data type from a value.
func (m *Marshaller) detectDataType(value any) types.DataType {
	switch v := value.(type) {
	case types.Tensor:
		return v.DataType()
	case matTypes.Matrix:
		// Matrix is always float32
		return types.FP32
	case vecTypes.Vector:
		// Vector is always float32
		return types.FP32
	}

	// Check for slices/arrays via reflection
	rv := reflect.ValueOf(value)
	if rv.Kind() == reflect.Slice || rv.Kind() == reflect.Array {
		if rv.Len() == 0 {
			return types.DT_UNKNOWN
		}
		elemType := rv.Type().Elem()
		switch elemType.Kind() {
		case reflect.Float32:
			return types.FP32
		case reflect.Float64:
			return types.FP64
		case reflect.Int32:
			return types.INT32
		case reflect.Int16:
			return types.INT16
		case reflect.Int8:
			return types.INT8
		case reflect.Uint8:
			return types.UINT8
		}
	}

	return types.DT_UNKNOWN
}

// dtypeToString converts DataType to string representation.
func dtypeToString(dtype types.DataType) string {
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
