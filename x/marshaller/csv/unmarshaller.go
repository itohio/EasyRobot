package csv

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/shape"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Unmarshaller implements CSV-based unmarshalling.
type Unmarshaller struct {
	opts types.Options
}

// NewUnmarshaller creates a new CSV unmarshaller.
func NewUnmarshaller(opts ...types.Option) *Unmarshaller {
	u := &Unmarshaller{
		opts: types.Options{},
	}
	for _, opt := range opts {
		opt.Apply(&u.opts)
	}
	return u
}

// Format returns the format name.
func (u *Unmarshaller) Format() string {
	return "csv"
}

// Unmarshal decodes value from CSV format.
func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
	// Apply additional options
	localOpts := u.opts
	for _, opt := range opts {
		opt.Apply(&localOpts)
	}

	// Use buffered reader to peek at first line for dtype comment
	bufReader := bufio.NewReader(r)

	// Read and parse dtype comment if present
	dtypeFromComment, err := u.readDataTypeComment(bufReader)
	if err != nil {
		return types.NewError("unmarshal", "csv", "failed to read dtype comment", err)
	}

	// If dtype found in comment, use it (unless explicitly overridden)
	if dtypeFromComment != types.DT_UNKNOWN && localOpts.DestinationType == types.DT_UNKNOWN {
		localOpts.DestinationType = dtypeFromComment
	}

	reader := csv.NewReader(bufReader)
	records, err := reader.ReadAll()
	if err != nil {
		return types.NewError("unmarshal", "csv", "failed to read CSV", err)
	}

	if len(records) == 0 {
		return types.NewError("unmarshal", "csv", "empty CSV file", nil)
	}

	// Check for header option
	hasHeader := localOpts.Metadata != nil && localOpts.Metadata["header"] == "true"

	// Auto-detect header: if header option is not set, check if first row is all numeric
	// If first row is all numeric, it's data (not header) - startRow = 0
	// If first row is not all numeric and header option not set, still treat as data
	// User must explicitly set WithHeader(true) to skip non-numeric first row
	startRow := 0
	if hasHeader {
		startRow = 1
	} else if len(records) > 0 && u.isAllNumeric(records[0]) {
		// First row is all numeric, definitely data (not header)
		startRow = 0
	}

	if len(records) <= startRow {
		return types.NewError("unmarshal", "csv", "no data rows", nil)
	}

	// Determine destination type and unmarshal
	dstValue := reflect.ValueOf(dst)
	if dstValue.Kind() != reflect.Ptr {
		return types.NewError("unmarshal", "csv", "destination must be a pointer", nil)
	}

	dstElem := dstValue.Elem()
	dstType := dstElem.Type()

	// Check if destination is a pointer to interface
	if dstType.Kind() == reflect.Interface {
		// Try to infer from hint or default to tensor
		hint := localOpts.Hint
		if hint == "" {
			hint = "tensor"
		}
		return u.unmarshalToInterface(dst, records, startRow, hint, localOpts)
	}

	// Check for interface types by checking if dst can hold the interface
	// This handles *types.Tensor, *matTypes.Matrix, *vecTypes.Vector
	if dstType.Implements(reflect.TypeOf((*types.Tensor)(nil)).Elem()) {
		return u.unmarshalToTensor(dst, records, startRow, localOpts)
	}
	if dstType.Implements(reflect.TypeOf((*matTypes.Matrix)(nil)).Elem()) {
		return u.unmarshalToMatrix(dst, records, startRow, localOpts)
	}
	if dstType.Implements(reflect.TypeOf((*vecTypes.Vector)(nil)).Elem()) {
		return u.unmarshalToVector(dst, records, startRow, localOpts)
	}

	// Check for slice types (but not vec.Vector which is []float32)
	if dstType.Kind() == reflect.Slice {
		// vec.Vector is []float32, but we want to handle it as Vector interface
		// Check if it's specifically vec.Vector by checking the underlying type
		if dstType == reflect.TypeOf(vec.Vector(nil)) {
			return u.unmarshalToVector(dst, records, startRow, localOpts)
		}
		return u.unmarshalToSlice(dst, records, startRow, localOpts)
	}

	return types.NewError("unmarshal", "csv", fmt.Sprintf("unsupported destination type: %s", dstType), nil)
}

// unmarshalToInterface unmarshals to an interface type (Tensor, Matrix, Vector).
func (u *Unmarshaller) unmarshalToInterface(dst any, records [][]string, startRow int, hint string, opts types.Options) error {
	switch hint {
	case "tensor":
		var t types.Tensor
		if err := u.unmarshalToTensor(&t, records, startRow, opts); err != nil {
			return err
		}
		return u.assignToDst(dst, t)
	case "matrix":
		var m matTypes.Matrix
		if err := u.unmarshalToMatrix(&m, records, startRow, opts); err != nil {
			return err
		}
		return u.assignToDst(dst, m)
	case "vector":
		var v vecTypes.Vector
		if err := u.unmarshalToVector(&v, records, startRow, opts); err != nil {
			return err
		}
		return u.assignToDst(dst, v)
	default:
		return types.NewError("unmarshal", "csv", fmt.Sprintf("unknown hint: %s", hint), nil)
	}
}

// unmarshalToTensor unmarshals CSV to a tensor (1D or 2D).
func (u *Unmarshaller) unmarshalToTensor(dst any, records [][]string, startRow int, opts types.Options) error {
	if len(records) <= startRow {
		return types.NewError("unmarshal", "csv", "no data rows", nil)
	}

	dataRows := records[startRow:]
	numRows := len(dataRows)
	if numRows == 0 {
		return types.NewError("unmarshal", "csv", "no data rows", nil)
	}

	numCols := len(dataRows[0])
	if numCols == 0 {
		return types.NewError("unmarshal", "csv", "empty rows", nil)
	}

	// Validate all rows have same number of columns
	for i, row := range dataRows {
		if len(row) != numCols {
			return types.NewError("unmarshal", "csv", fmt.Sprintf("row %d has %d columns, expected %d", i+startRow, len(row), numCols), nil)
		}
	}

	// Determine shape
	// If hint says tensor and single row, it's 1D
	// Otherwise, if multiple rows, it's 2D
	var shape shape.Shape
	hint := opts.Hint
	if hint == "tensor" && numRows == 1 {
		// 1D tensor: single row
		shape = tensor.NewShape(numCols)
	} else if numRows > 1 {
		// 2D tensor: rows x cols
		shape = tensor.NewShape(numRows, numCols)
	} else {
		// Default: single row is 1D
		shape = tensor.NewShape(numCols)
	}

	// Get tensor factory
	tensorFactory := opts.TensorFactory
	if tensorFactory == nil {
		tensorFactory = func(dtype types.DataType, shape types.Shape) types.Tensor {
			return tensor.New(dtype, shape)
		}
	}

	// Determine dtype (default to FP32, or use destination type)
	dtype := types.FP32
	if opts.DestinationType != 0 {
		dtype = opts.DestinationType
	}

	// Create tensor
	t := tensorFactory(dtype, shape)

	// Parse and fill data
	data := t.Data()
	if data == nil {
		return types.NewError("unmarshal", "csv", "tensor data is nil", nil)
	}

	// Parse and fill data based on shape
	if len(shape) == 1 {
		// 1D: single row
		if err := u.fillTensor1D(t, dataRows, dtype, startRow); err != nil {
			return err
		}
	} else {
		// 2D: multiple rows
		if err := u.fillTensor2D(t, dataRows, dtype, numCols, startRow); err != nil {
			return err
		}
	}

	return u.assignToDst(dst, t)
}

// unmarshalToMatrix unmarshals CSV to a matrix.
func (u *Unmarshaller) unmarshalToMatrix(dst any, records [][]string, startRow int, opts types.Options) error {
	if len(records) <= startRow {
		return types.NewError("unmarshal", "csv", "no data rows", nil)
	}

	dataRows := records[startRow:]
	numRows := len(dataRows)
	if numRows == 0 {
		return types.NewError("unmarshal", "csv", "no data rows", nil)
	}

	numCols := len(dataRows[0])
	if numCols == 0 {
		return types.NewError("unmarshal", "csv", "empty rows", nil)
	}

	// Validate all rows have same number of columns
	for i, row := range dataRows {
		if len(row) != numCols {
			return types.NewError("unmarshal", "csv", fmt.Sprintf("row %d has %d columns, expected %d", i+startRow, len(row), numCols), nil)
		}
	}

	// Create matrix
	m := mat.New(numRows, numCols)

	// Parse and fill data
	for i, row := range dataRows {
		rowVec := m.Row(i).View().(vec.Vector)
		for j, cell := range row {
			val, err := parseFloat32(cell)
			if err != nil {
				return types.NewError("unmarshal", "csv", fmt.Sprintf("row %d, col %d: %v", i+startRow, j, err), err)
			}
			rowVec[j] = val
		}
	}

	return u.assignToDst(dst, m)
}

// unmarshalToVector unmarshals CSV to a vector (single row or single column).
func (u *Unmarshaller) unmarshalToVector(dst any, records [][]string, startRow int, opts types.Options) error {
	if len(records) <= startRow {
		return types.NewError("unmarshal", "csv", "no data rows", nil)
	}

	// Check orientation option
	orientation := "row" // default
	if opts.Metadata != nil {
		if o, ok := opts.Metadata["orientation"]; ok {
			orientation = o
		}
	}

	if orientation == "column" {
		// Read from single column (vertical)
		dataRows := records[startRow:]
		if len(dataRows) == 0 {
			return types.NewError("unmarshal", "csv", "no data rows", nil)
		}

		// Check if single column
		if len(dataRows[0]) != 1 {
			return types.NewError("unmarshal", "csv", "vector column mode requires single column", nil)
		}

		// Validate all rows have single column
		for i, row := range dataRows {
			if len(row) != 1 {
				return types.NewError("unmarshal", "csv", fmt.Sprintf("row %d has %d columns, expected 1", i+startRow, len(row)), nil)
			}
		}

		// Create vector
		v := vec.New(len(dataRows))

		// Parse and fill data from single column
		for i, row := range dataRows {
			val, err := parseFloat32(row[0])
			if err != nil {
				return types.NewError("unmarshal", "csv", fmt.Sprintf("row %d: %v", i+startRow, err), err)
			}
			v[i] = val
		}

		return u.assignToDst(dst, v)
	}

	// Default: Read from single row (horizontal)
	dataRow := records[startRow]
	if len(dataRow) == 0 {
		return types.NewError("unmarshal", "csv", "empty row", nil)
	}

	// Create vector
	v := vec.New(len(dataRow))

	// Parse and fill data from single row
	for i, cell := range dataRow {
		val, err := parseFloat32(cell)
		if err != nil {
			return types.NewError("unmarshal", "csv", fmt.Sprintf("col %d: %v", i, err), err)
		}
		v[i] = val
	}

	return u.assignToDst(dst, v)
}

// unmarshalToSlice unmarshals CSV to a slice (single row or single column).
func (u *Unmarshaller) unmarshalToSlice(dst any, records [][]string, startRow int, opts types.Options) error {
	if len(records) <= startRow {
		return types.NewError("unmarshal", "csv", "no data rows", nil)
	}

	// Check orientation option
	orientation := "row" // default
	if opts.Metadata != nil {
		if o, ok := opts.Metadata["orientation"]; ok {
			orientation = o
		}
	}

	if orientation == "column" {
		// Read from single column (vertical)
		dataRows := records[startRow:]
		if len(dataRows) == 0 {
			return types.NewError("unmarshal", "csv", "no data rows", nil)
		}

		// Check if single column
		if len(dataRows[0]) != 1 {
			return types.NewError("unmarshal", "csv", "slice column mode requires single column", nil)
		}

		// Validate all rows have single column
		for i, row := range dataRows {
			if len(row) != 1 {
				return types.NewError("unmarshal", "csv", fmt.Sprintf("row %d has %d columns, expected 1", i+startRow, len(row)), nil)
			}
		}

		dstValue := reflect.ValueOf(dst).Elem()
		dstType := dstValue.Type()

		// Verify it's actually a slice type
		if dstType.Kind() != reflect.Slice {
			return types.NewError("unmarshal", "csv", fmt.Sprintf("destination is not a slice: %s", dstType), nil)
		}

		elemType := dstType.Elem()

		// Create slice
		slice := reflect.MakeSlice(dstType, len(dataRows), len(dataRows))

		// Parse based on element type
		for i, row := range dataRows {
			val, err := u.parseValue(row[0], elemType)
			if err != nil {
				return types.NewError("unmarshal", "csv", fmt.Sprintf("row %d: %v", i+startRow, err), err)
			}
			slice.Index(i).Set(reflect.ValueOf(val).Convert(elemType))
		}

		dstValue.Set(slice)
		return nil
	}

	// Default: Read from single row (horizontal)
	dataRow := records[startRow]
	if len(dataRow) == 0 {
		return types.NewError("unmarshal", "csv", "empty row", nil)
	}

	dstValue := reflect.ValueOf(dst).Elem()
	dstType := dstValue.Type()

	// Verify it's actually a slice type
	if dstType.Kind() != reflect.Slice {
		return types.NewError("unmarshal", "csv", fmt.Sprintf("destination is not a slice: %s", dstType), nil)
	}

	elemType := dstType.Elem()

	// Create slice
	slice := reflect.MakeSlice(dstType, len(dataRow), len(dataRow))

	// Parse based on element type
	for i, cell := range dataRow {
		val, err := u.parseValue(cell, elemType)
		if err != nil {
			return types.NewError("unmarshal", "csv", fmt.Sprintf("col %d: %v", i, err), err)
		}
		slice.Index(i).Set(reflect.ValueOf(val).Convert(elemType))
	}

	dstValue.Set(slice)
	return nil
}

// parseValue parses a string value to the target type.
func (u *Unmarshaller) parseValue(s string, targetType reflect.Type) (any, error) {
	switch targetType.Kind() {
	case reflect.Float32:
		return parseFloat32(s)
	case reflect.Float64:
		return parseFloat64(s)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		val, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, err
		}
		return reflect.ValueOf(val).Convert(targetType).Interface(), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		val, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return nil, err
		}
		return reflect.ValueOf(val).Convert(targetType).Interface(), nil
	case reflect.String:
		return s, nil
	default:
		return nil, fmt.Errorf("unsupported type: %s", targetType)
	}
}

// parseFloat32 parses a float32 from string.
func parseFloat32(s string) (float32, error) {
	val, err := strconv.ParseFloat(s, 32)
	if err != nil {
		return 0, err
	}
	return float32(val), nil
}

// parseFloat64 parses a float64 from string.
func parseFloat64(s string) (float64, error) {
	return strconv.ParseFloat(s, 64)
}

// assignToDst assigns value to destination.
func (u *Unmarshaller) assignToDst(dst any, value any) error {
	dstValue := reflect.ValueOf(dst)
	if dstValue.Kind() != reflect.Ptr {
		return fmt.Errorf("destination must be a pointer")
	}

	dstElem := dstValue.Elem()
	valueValue := reflect.ValueOf(value)

	// Check if types are compatible
	if !valueValue.Type().AssignableTo(dstElem.Type()) {
		// Try to convert
		if !valueValue.Type().ConvertibleTo(dstElem.Type()) {
			return fmt.Errorf("cannot assign %T to %T", value, dstElem.Type())
		}
		valueValue = valueValue.Convert(dstElem.Type())
	}

	dstElem.Set(valueValue)
	return nil
}

// readDataTypeComment reads and parses the data type comment from the beginning of the file.
// Returns DT_UNKNOWN if no comment is found.
func (u *Unmarshaller) readDataTypeComment(r *bufio.Reader) (types.DataType, error) {
	// Peek at first line
	line, err := r.Peek(256) // Peek at first 256 bytes
	if err != nil && err != io.EOF {
		return types.DT_UNKNOWN, err
	}

	// Find first newline
	firstLine := string(line)
	if idx := strings.IndexByte(firstLine, '\n'); idx >= 0 {
		firstLine = firstLine[:idx]
	}

	// Check if it's a dtype comment
	firstLine = strings.TrimSpace(firstLine)
	if strings.HasPrefix(firstLine, dataTypeCommentPrefix) {
		// Extract dtype string
		dtypeStr := strings.TrimSpace(strings.TrimPrefix(firstLine, dataTypeCommentPrefix))
		dtype := stringToDataType(dtypeStr)

		// Read the line to consume it
		_, err := r.ReadString('\n')
		if err != nil && err != io.EOF {
			return types.DT_UNKNOWN, err
		}

		return dtype, nil
	}

	return types.DT_UNKNOWN, nil
}

// stringToDataType converts string representation to DataType.
func stringToDataType(s string) types.DataType {
	s = strings.ToUpper(strings.TrimSpace(s))
	switch s {
	case "FP32":
		return types.FP32
	case "FP64":
		return types.FP64
	case "INT32":
		return types.INT32
	case "INT16":
		return types.INT16
	case "INT8":
		return types.INT8
	case "INT64":
		return types.INT64
	case "UINT8":
		return types.UINT8
	default:
		return types.DT_UNKNOWN
	}
}

// isAllNumeric checks if all values in a CSV row are numeric.
func (u *Unmarshaller) isAllNumeric(row []string) bool {
	if len(row) == 0 {
		return false
	}

	for _, cell := range row {
		cell = strings.TrimSpace(cell)
		if cell == "" {
			return false // Empty cells are not numeric
		}
		// Try to parse as float
		_, err := strconv.ParseFloat(cell, 64)
		if err != nil {
			return false
		}
	}
	return true
}

// fillTensor1D fills a 1D tensor with data from a single CSV row.
func (u *Unmarshaller) fillTensor1D(t types.Tensor, dataRows [][]string, dtype types.DataType, startRow int) error {
	data := t.Data()
	if data == nil {
		return types.NewError("unmarshal", "csv", "tensor data is nil", nil)
	}

	if len(dataRows) == 0 {
		return types.NewError("unmarshal", "csv", "no data rows", nil)
	}

	// 1D tensor is a single row
	dataRow := dataRows[0]

	switch dtype {
	case types.FP32:
		dataSlice, ok := data.([]float32)
		if !ok {
			return types.NewError("unmarshal", "csv", "tensor data is not []float32", nil)
		}
		for i, cell := range dataRow {
			val, err := parseFloat32(cell)
			if err != nil {
				return types.NewError("unmarshal", "csv", fmt.Sprintf("col %d: %v", i, err), err)
			}
			dataSlice[i] = val
		}
	case types.INT32:
		dataSlice, ok := data.([]int32)
		if !ok {
			return types.NewError("unmarshal", "csv", "tensor data is not []int32", nil)
		}
		for i, cell := range dataRow {
			val, err := strconv.ParseInt(cell, 10, 32)
			if err != nil {
				return types.NewError("unmarshal", "csv", fmt.Sprintf("col %d: %v", i, err), err)
			}
			dataSlice[i] = int32(val)
		}
	case types.INT16:
		dataSlice, ok := data.([]int16)
		if !ok {
			return types.NewError("unmarshal", "csv", "tensor data is not []int16", nil)
		}
		for i, cell := range dataRow {
			val, err := strconv.ParseInt(cell, 10, 16)
			if err != nil {
				return types.NewError("unmarshal", "csv", fmt.Sprintf("col %d: %v", i, err), err)
			}
			dataSlice[i] = int16(val)
		}
	case types.INT8:
		dataSlice, ok := data.([]int8)
		if !ok {
			return types.NewError("unmarshal", "csv", "tensor data is not []int8", nil)
		}
		for i, cell := range dataRow {
			val, err := strconv.ParseInt(cell, 10, 8)
			if err != nil {
				return types.NewError("unmarshal", "csv", fmt.Sprintf("col %d: %v", i, err), err)
			}
			dataSlice[i] = int8(val)
		}
	case types.UINT8:
		dataSlice, ok := data.([]uint8)
		if !ok {
			return types.NewError("unmarshal", "csv", "tensor data is not []uint8", nil)
		}
		for i, cell := range dataRow {
			val, err := strconv.ParseUint(cell, 10, 8)
			if err != nil {
				return types.NewError("unmarshal", "csv", fmt.Sprintf("col %d: %v", i, err), err)
			}
			dataSlice[i] = uint8(val)
		}
	default:
		return types.NewError("unmarshal", "csv", fmt.Sprintf("unsupported tensor dtype: %v", dtype), nil)
	}
	return nil
}

// fillTensor2D fills a 2D tensor with data from CSV rows.
func (u *Unmarshaller) fillTensor2D(t types.Tensor, dataRows [][]string, dtype types.DataType, numCols int, startRow int) error {
	data := t.Data()
	if data == nil {
		return types.NewError("unmarshal", "csv", "tensor data is nil", nil)
	}

	switch dtype {
	case types.FP32:
		dataSlice, ok := data.([]float32)
		if !ok {
			return types.NewError("unmarshal", "csv", "tensor data is not []float32", nil)
		}
		for i, row := range dataRows {
			for j, cell := range row {
				val, err := parseFloat32(cell)
				if err != nil {
					return types.NewError("unmarshal", "csv", fmt.Sprintf("row %d, col %d: %v", i+startRow, j, err), err)
				}
				dataSlice[i*numCols+j] = val
			}
		}
	case types.INT32:
		dataSlice, ok := data.([]int32)
		if !ok {
			return types.NewError("unmarshal", "csv", "tensor data is not []int32", nil)
		}
		for i, row := range dataRows {
			for j, cell := range row {
				val, err := strconv.ParseInt(cell, 10, 32)
				if err != nil {
					return types.NewError("unmarshal", "csv", fmt.Sprintf("row %d, col %d: %v", i+startRow, j, err), err)
				}
				dataSlice[i*numCols+j] = int32(val)
			}
		}
	case types.INT16:
		dataSlice, ok := data.([]int16)
		if !ok {
			return types.NewError("unmarshal", "csv", "tensor data is not []int16", nil)
		}
		for i, row := range dataRows {
			for j, cell := range row {
				val, err := strconv.ParseInt(cell, 10, 16)
				if err != nil {
					return types.NewError("unmarshal", "csv", fmt.Sprintf("row %d, col %d: %v", i+startRow, j, err), err)
				}
				dataSlice[i*numCols+j] = int16(val)
			}
		}
	case types.INT8:
		dataSlice, ok := data.([]int8)
		if !ok {
			return types.NewError("unmarshal", "csv", "tensor data is not []int8", nil)
		}
		for i, row := range dataRows {
			for j, cell := range row {
				val, err := strconv.ParseInt(cell, 10, 8)
				if err != nil {
					return types.NewError("unmarshal", "csv", fmt.Sprintf("row %d, col %d: %v", i+startRow, j, err), err)
				}
				dataSlice[i*numCols+j] = int8(val)
			}
		}
	case types.UINT8:
		dataSlice, ok := data.([]uint8)
		if !ok {
			return types.NewError("unmarshal", "csv", "tensor data is not []uint8", nil)
		}
		for i, row := range dataRows {
			for j, cell := range row {
				val, err := strconv.ParseUint(cell, 10, 8)
				if err != nil {
					return types.NewError("unmarshal", "csv", fmt.Sprintf("row %d, col %d: %v", i+startRow, j, err), err)
				}
				dataSlice[i*numCols+j] = uint8(val)
			}
		}
	default:
		return types.NewError("unmarshal", "csv", fmt.Sprintf("unsupported tensor dtype: %v", dtype), nil)
	}
	return nil
}
