package csv

import (
	"bytes"
	_ "embed"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

//go:embed testdata/vector_no_header.csv
var vectorNoHeaderCSV string

//go:embed testdata/vector_with_header.csv
var vectorWithHeaderCSV string

//go:embed testdata/matrix_no_header.csv
var matrixNoHeaderCSV string

//go:embed testdata/matrix_with_header.csv
var matrixWithHeaderCSV string

//go:embed testdata/tensor_1d_fp32.csv
var tensor1DFP32CSV string

//go:embed testdata/tensor_2d_fp32.csv
var tensor2DFP32CSV string

//go:embed testdata/tensor_1d_int32.csv
var tensor1DInt32CSV string

//go:embed testdata/tensor_2d_int32.csv
var tensor2DInt32CSV string

//go:embed testdata/slice_string.csv
var sliceStringCSV string

//go:embed testdata/vector_column.csv
var vectorColumnCSV string

//go:embed testdata/vector_column_with_header.csv
var vectorColumnWithHeaderCSV string

//go:embed testdata/slice_float32_column.csv
var sliceFloat32ColumnCSV string

func TestMarshaller_Format(t *testing.T) {
	m := NewMarshaller()
	assert.Equal(t, "csv", m.Format())
}

func TestMarshaller_MarshalVector(t *testing.T) {
	m := NewMarshaller()
	v := vec.NewFrom(1.0, 2.0, 3.0, 4.0, 5.0)

	var buf bytes.Buffer
	err := m.Marshal(&buf, v)
	require.NoError(t, err)

	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	// Account for dtype comment line + 1 data row
	assert.GreaterOrEqual(t, len(lines), 2)
	assert.Contains(t, output, "# dtype: FP32")
	// Format uses 'g' so 1.0 becomes "1", but values should be present
	assert.Contains(t, output, "1")
	assert.Contains(t, output, "5")
}

func TestMarshaller_MarshalVectorWithHeader(t *testing.T) {
	m := NewMarshaller(WithHeader(true))
	v := vec.NewFrom(1.0, 2.0, 3.0)

	var buf bytes.Buffer
	err := m.Marshal(&buf, v)
	require.NoError(t, err)

	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	// Account for dtype comment line + header + 1 data row
	assert.GreaterOrEqual(t, len(lines), 3) // 1 dtype comment + 1 header + 1 data row
	assert.Contains(t, output, "# dtype: FP32")
	assert.Contains(t, output, "col_0")
	assert.Contains(t, output, "col_2")
	// Format uses 'g' so 1.0 becomes "1", but values should be present
	assert.Contains(t, output, "1")
	assert.Contains(t, output, "3")
}

func TestMarshaller_MarshalMatrix(t *testing.T) {
	m := NewMarshaller()
	mat := mat.New(2, 3)
	mat.SetRow(0, vec.NewFrom(1.0, 2.0, 3.0))
	mat.SetRow(1, vec.NewFrom(4.0, 5.0, 6.0))

	var buf bytes.Buffer
	err := m.Marshal(&buf, mat)
	require.NoError(t, err)

	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	// Account for dtype comment line
	assert.GreaterOrEqual(t, len(lines), 3) // 1 dtype comment + 2 data rows
	assert.Contains(t, output, "# dtype: FP32")
	assert.Contains(t, output, "1")
	assert.Contains(t, output, "6")
}

func TestMarshaller_MarshalMatrixWithHeader(t *testing.T) {
	m := NewMarshaller(WithHeader(true))
	mat := mat.New(2, 3)
	mat.SetRow(0, vec.NewFrom(1.0, 2.0, 3.0))
	mat.SetRow(1, vec.NewFrom(4.0, 5.0, 6.0))

	var buf bytes.Buffer
	err := m.Marshal(&buf, mat)
	require.NoError(t, err)

	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	// Account for dtype comment line + header + data rows
	assert.GreaterOrEqual(t, len(lines), 4) // 1 dtype comment + 1 header + 2 data rows
	assert.Contains(t, output, "# dtype: FP32")
	assert.Contains(t, output, "col_0")
	assert.Contains(t, output, "col_2")
}

func TestMarshaller_MarshalTensor1D(t *testing.T) {
	m := NewMarshaller()
	tens := tensor.FromArray(tensor.NewShape(5), []float32{1.0, 2.0, 3.0, 4.0, 5.0})

	var buf bytes.Buffer
	err := m.Marshal(&buf, tens)
	require.NoError(t, err)

	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	// Account for dtype comment line + 1 data row
	assert.GreaterOrEqual(t, len(lines), 2) // 1 dtype comment + 1 data row
	assert.Contains(t, output, "# dtype: FP32")
	// Format uses 'g' so 1.0 becomes "1", but values should be present
	assert.Contains(t, output, "1")
	assert.Contains(t, output, "5")
}

func TestMarshaller_MarshalTensor2D(t *testing.T) {
	m := NewMarshaller()
	tens := tensor.FromArray(tensor.NewShape(2, 3), []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})

	var buf bytes.Buffer
	err := m.Marshal(&buf, tens)
	require.NoError(t, err)

	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	// Account for dtype comment line
	assert.GreaterOrEqual(t, len(lines), 3) // 1 dtype comment + 2 data rows
	assert.Contains(t, output, "# dtype: FP32")
}

func TestMarshaller_MarshalSliceFloat32(t *testing.T) {
	m := NewMarshaller()
	slice := []float32{1.0, 2.0, 3.0}

	var buf bytes.Buffer
	err := m.Marshal(&buf, slice)
	require.NoError(t, err)

	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	// Default is row format: dtype comment + 1 data row
	assert.GreaterOrEqual(t, len(lines), 2)
	assert.Contains(t, output, "# dtype: FP32")
	// Should be a single row with comma-separated values (formatFloat32 uses 'g' format)
	assert.Contains(t, output, "1")
	assert.Contains(t, output, "2")
	assert.Contains(t, output, "3")
}

func TestMarshaller_MarshalSliceString(t *testing.T) {
	m := NewMarshaller()
	slice := []string{"a", "b", "c"}

	var buf bytes.Buffer
	err := m.Marshal(&buf, slice)
	require.NoError(t, err)

	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	// Default is row format: 1 data row (no dtype comment for strings)
	assert.GreaterOrEqual(t, len(lines), 1)
	assert.Contains(t, output, "a")
	assert.Contains(t, output, "c")
	// Should be a single row with comma-separated values
	assert.Contains(t, output, "a,b,c")
}

func TestUnmarshaller_UnmarshalVector(t *testing.T) {
	u := NewUnmarshaller()
	csvData := "1.0,2.0,3.0\n"

	var result vec.Vector
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	assert.Equal(t, 3, result.Len())
	assert.Equal(t, float32(1.0), result.View().(vec.Vector)[0])
	assert.Equal(t, float32(3.0), result.View().(vec.Vector)[2])
}

func TestUnmarshaller_UnmarshalVectorWithHeader(t *testing.T) {
	u := NewUnmarshaller(WithHeader(true))
	csvData := "col_0,col_1,col_2\n1.0,2.0,3.0\n"

	var result vec.Vector
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	assert.Equal(t, 3, result.Len())
	assert.Equal(t, float32(1.0), result.View().(vec.Vector)[0])
	assert.Equal(t, float32(3.0), result.View().(vec.Vector)[2])
}

func TestUnmarshaller_UnmarshalMatrix(t *testing.T) {
	u := NewUnmarshaller()
	csvData := "1.0,2.0,3.0\n4.0,5.0,6.0\n"

	var result mat.Matrix
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	assert.Equal(t, 2, result.Rows())
	assert.Equal(t, 3, result.Cols())
	row0 := result.Row(0).View().(vec.Vector)
	assert.Equal(t, float32(1.0), row0[0])
	assert.Equal(t, float32(3.0), row0[2])
}

func TestUnmarshaller_UnmarshalTensor1D(t *testing.T) {
	u := NewUnmarshaller(types.WithHint("tensor"))
	csvData := "1.0,2.0,3.0\n"

	var result types.Tensor
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	assert.Equal(t, tensor.NewShape(3), result.Shape())
	data := result.Data().([]float32)
	assert.Equal(t, float32(1.0), data[0])
	assert.Equal(t, float32(3.0), data[2])
}

func TestUnmarshaller_UnmarshalTensor2D(t *testing.T) {
	u := NewUnmarshaller()
	csvData := "1.0,2.0,3.0\n4.0,5.0,6.0\n"

	var result types.Tensor
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	assert.Equal(t, tensor.NewShape(2, 3), result.Shape())
	data := result.Data().([]float32)
	assert.Equal(t, float32(1.0), data[0])
	assert.Equal(t, float32(6.0), data[5])
}

func TestUnmarshaller_UnmarshalSliceFloat32(t *testing.T) {
	u := NewUnmarshaller()
	csvData := "1.0,2.0,3.0\n"

	var result []float32
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	assert.Equal(t, 3, len(result))
	assert.Equal(t, float32(1.0), result[0])
	assert.Equal(t, float32(3.0), result[2])
}

func TestUnmarshaller_UnmarshalSliceString(t *testing.T) {
	u := NewUnmarshaller()
	csvData := "a,b,c\n"

	var result []string
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	assert.Equal(t, 3, len(result))
	assert.Equal(t, "a", result[0])
	assert.Equal(t, "c", result[2])
}

func TestRoundTrip_Vector(t *testing.T) {
	m := NewMarshaller()
	u := NewUnmarshaller()

	original := vec.NewFrom(1.0, 2.0, 3.0, 4.0, 5.0)

	var buf bytes.Buffer
	err := m.Marshal(&buf, original)
	require.NoError(t, err)

	var result vec.Vector
	err = u.Unmarshal(&buf, &result)
	require.NoError(t, err)

	assert.Equal(t, original.Len(), result.Len())
	origView := original.View().(vec.Vector)
	resultView := result.View().(vec.Vector)
	for i := 0; i < original.Len(); i++ {
		assert.InDelta(t, origView[i], resultView[i], 0.0001)
	}
}

func TestRoundTrip_Matrix(t *testing.T) {
	m := NewMarshaller()
	u := NewUnmarshaller()

	original := mat.New(2, 3)
	original.SetRow(0, vec.NewFrom(1.0, 2.0, 3.0))
	original.SetRow(1, vec.NewFrom(4.0, 5.0, 6.0))

	var buf bytes.Buffer
	err := m.Marshal(&buf, original)
	require.NoError(t, err)

	var result mat.Matrix
	err = u.Unmarshal(&buf, &result)
	require.NoError(t, err)

	assert.Equal(t, original.Rows(), result.Rows())
	assert.Equal(t, original.Cols(), result.Cols())
	for i := 0; i < original.Rows(); i++ {
		origRow := original.Row(i).View().(vec.Vector)
		resultRow := result.Row(i).View().(vec.Vector)
		for j := 0; j < original.Cols(); j++ {
			assert.InDelta(t, origRow[j], resultRow[j], 0.0001)
		}
	}
}

func TestRoundTrip_Tensor2D(t *testing.T) {
	m := NewMarshaller()
	u := NewUnmarshaller()

	original := tensor.FromArray(tensor.NewShape(2, 3), []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})

	var buf bytes.Buffer
	err := m.Marshal(&buf, original)
	require.NoError(t, err)

	var result types.Tensor
	err = u.Unmarshal(&buf, &result)
	require.NoError(t, err)

	assert.Equal(t, original.Shape(), result.Shape())
	origData := original.Data().([]float32)
	resultData := result.Data().([]float32)
	for i := 0; i < len(origData); i++ {
		assert.InDelta(t, origData[i], resultData[i], 0.0001)
	}
}

// TestDataFile tests using embedded testdata files

func TestUnmarshaller_FromTestData_VectorNoHeader(t *testing.T) {
	u := NewUnmarshaller()

	var result vec.Vector
	err := u.Unmarshal(strings.NewReader(vectorNoHeaderCSV), &result)
	require.NoError(t, err)

	assert.Equal(t, 5, result.Len())
	resultView := result.View().(vec.Vector)
	assert.InDelta(t, 1.0, resultView[0], 0.0001)
	assert.InDelta(t, 5.0, resultView[4], 0.0001)
}

func TestUnmarshaller_FromTestData_VectorWithHeader(t *testing.T) {
	u := NewUnmarshaller(WithHeader(true))

	var result vec.Vector
	err := u.Unmarshal(strings.NewReader(vectorWithHeaderCSV), &result)
	require.NoError(t, err)

	assert.Equal(t, 5, result.Len())
	resultView := result.View().(vec.Vector)
	assert.InDelta(t, 1.0, resultView[0], 0.0001)
	assert.InDelta(t, 5.0, resultView[4], 0.0001)
}

func TestUnmarshaller_FromTestData_MatrixNoHeader(t *testing.T) {
	u := NewUnmarshaller()

	var result mat.Matrix
	err := u.Unmarshal(strings.NewReader(matrixNoHeaderCSV), &result)
	require.NoError(t, err)

	assert.Equal(t, 3, result.Rows())
	assert.Equal(t, 3, result.Cols())
	row0 := result.Row(0).View().(vec.Vector)
	assert.InDelta(t, 1.0, row0[0], 0.0001)
	assert.InDelta(t, 3.0, row0[2], 0.0001)
}

func TestUnmarshaller_FromTestData_MatrixWithHeader(t *testing.T) {
	u := NewUnmarshaller(WithHeader(true))

	var result mat.Matrix
	err := u.Unmarshal(strings.NewReader(matrixWithHeaderCSV), &result)
	require.NoError(t, err)

	assert.Equal(t, 3, result.Rows())
	assert.Equal(t, 3, result.Cols())
	row0 := result.Row(0).View().(vec.Vector)
	assert.InDelta(t, 1.0, row0[0], 0.0001)
	assert.InDelta(t, 3.0, row0[2], 0.0001)
}

func TestUnmarshaller_FromTestData_Tensor1DFP32(t *testing.T) {
	u := NewUnmarshaller(types.WithHint("tensor"))

	var result types.Tensor
	err := u.Unmarshal(strings.NewReader(tensor1DFP32CSV), &result)
	require.NoError(t, err)

	assert.Equal(t, tensor.NewShape(5), result.Shape())
	assert.Equal(t, types.FP32, result.DataType())
	data := result.Data().([]float32)
	assert.InDelta(t, 1.5, data[0], 0.0001)
	assert.InDelta(t, 5.5, data[4], 0.0001)
}

func TestUnmarshaller_FromTestData_Tensor2DFP32(t *testing.T) {
	u := NewUnmarshaller()

	var result types.Tensor
	err := u.Unmarshal(strings.NewReader(tensor2DFP32CSV), &result)
	require.NoError(t, err)

	assert.Equal(t, tensor.NewShape(2, 3), result.Shape())
	assert.Equal(t, types.FP32, result.DataType())
	data := result.Data().([]float32)
	assert.InDelta(t, 1.0, data[0], 0.0001)
	assert.InDelta(t, 6.0, data[5], 0.0001)
}

func TestUnmarshaller_FromTestData_Tensor1DInt32(t *testing.T) {
	u := NewUnmarshaller(types.WithHint("tensor"))

	var result types.Tensor
	err := u.Unmarshal(strings.NewReader(tensor1DInt32CSV), &result)
	require.NoError(t, err)

	assert.Equal(t, tensor.NewShape(5), result.Shape())
	assert.Equal(t, types.INT32, result.DataType())
	data := result.Data().([]int32)
	assert.Equal(t, int32(1), data[0])
	assert.Equal(t, int32(5), data[4])
}

func TestUnmarshaller_FromTestData_Tensor2DInt32(t *testing.T) {
	u := NewUnmarshaller()

	var result types.Tensor
	err := u.Unmarshal(strings.NewReader(tensor2DInt32CSV), &result)
	require.NoError(t, err)

	assert.Equal(t, tensor.NewShape(2, 3), result.Shape())
	assert.Equal(t, types.INT32, result.DataType())
	data := result.Data().([]int32)
	assert.Equal(t, int32(1), data[0])
	assert.Equal(t, int32(6), data[5])
}

func TestUnmarshaller_FromTestData_SliceString(t *testing.T) {
	u := NewUnmarshaller()

	var result []string
	err := u.Unmarshal(strings.NewReader(sliceStringCSV), &result)
	require.NoError(t, err)

	assert.Equal(t, 5, len(result))
	assert.Equal(t, "apple", result[0])
	assert.Equal(t, "elderberry", result[4])
}

func TestMarshaller_ToTestDataFormat(t *testing.T) {
	// Test that marshalled output matches expected format
	m := NewMarshaller()
	v := vec.NewFrom(1.0, 2.0, 3.0, 4.0, 5.0)

	var buf bytes.Buffer
	err := m.Marshal(&buf, v)
	require.NoError(t, err)

	output := buf.String()
	// Should start with dtype comment
	assert.Contains(t, output, "# dtype: FP32")
	// Should contain the values
	assert.Contains(t, output, "1")
	assert.Contains(t, output, "5")
}

func TestUnmarshaller_FromFile(t *testing.T) {
	// Test loading from actual file (if testdata exists)
	testdataDir := "testdata"
	if _, err := os.Stat(testdataDir); os.IsNotExist(err) {
		t.Skip("testdata directory not found")
	}

	u := NewUnmarshaller()

	// Test loading vector_no_header.csv
	filePath := filepath.Join(testdataDir, "vector_no_header.csv")
	file, err := os.Open(filePath)
	require.NoError(t, err)
	defer file.Close()

	var result vec.Vector
	err = u.Unmarshal(file, &result)
	require.NoError(t, err)

	assert.Equal(t, 5, result.Len())
}

// TestWithDataType tests the WithDataType option
func TestUnmarshaller_WithDataType(t *testing.T) {
	// CSV with ambiguous integer data
	csvData := "1,2,3,4,5\n"

	// Test unmarshalling as INT32
	u := NewUnmarshaller(WithDataType(types.INT32), types.WithHint("tensor"))
	var result types.Tensor
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	assert.Equal(t, tensor.NewShape(5), result.Shape())
	assert.Equal(t, types.INT32, result.DataType())
	data := result.Data().([]int32)
	assert.Equal(t, int32(1), data[0])
	assert.Equal(t, int32(5), data[4])
}

// TestHeaderAutoDetection tests that numeric headers are treated as data
func TestUnmarshaller_HeaderAutoDetection(t *testing.T) {
	// CSV where first row looks like it could be a header but is all numeric
	csvData := "1.0,2.0,3.0\n4.0,5.0,6.0\n"

	u := NewUnmarshaller()
	var result mat.Matrix
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	// Should read both rows as data (not skip first row)
	assert.Equal(t, 2, result.Rows())
	assert.Equal(t, 3, result.Cols())
	row0 := result.Row(0).View().(vec.Vector)
	assert.InDelta(t, 1.0, row0[0], 0.0001)
}

// TestHeaderExplicit tests that explicit header option works
func TestUnmarshaller_HeaderExplicit(t *testing.T) {
	// CSV with explicit header
	csvData := "col_0,col_1,col_2\n1.0,2.0,3.0\n4.0,5.0,6.0\n"

	u := NewUnmarshaller(WithHeader(true))
	var result mat.Matrix
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	// Should skip header row
	assert.Equal(t, 2, result.Rows())
	assert.Equal(t, 3, result.Cols())
	row0 := result.Row(0).View().(vec.Vector)
	assert.InDelta(t, 1.0, row0[0], 0.0001)
}

// TestDtypeComment tests that dtype comments are read correctly
func TestUnmarshaller_DtypeComment(t *testing.T) {
	csvData := "# dtype: INT32\n1,2,3,4,5\n"

	u := NewUnmarshaller(types.WithHint("tensor"))
	var result types.Tensor
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	assert.Equal(t, types.INT32, result.DataType())
	assert.Equal(t, tensor.NewShape(5), result.Shape())
}

// TestRoundTrip_WithHeader tests round-trip with header
func TestRoundTrip_WithHeader(t *testing.T) {
	m := NewMarshaller(WithHeader(true))
	u := NewUnmarshaller(WithHeader(true))

	original := mat.New(2, 3)
	original.SetRow(0, vec.NewFrom(1.0, 2.0, 3.0))
	original.SetRow(1, vec.NewFrom(4.0, 5.0, 6.0))

	var buf bytes.Buffer
	err := m.Marshal(&buf, original)
	require.NoError(t, err)

	var result mat.Matrix
	err = u.Unmarshal(&buf, &result)
	require.NoError(t, err)

	assert.Equal(t, original.Rows(), result.Rows())
	assert.Equal(t, original.Cols(), result.Cols())
	for i := 0; i < original.Rows(); i++ {
		origRow := original.Row(i).View().(vec.Vector)
		resultRow := result.Row(i).View().(vec.Vector)
		for j := 0; j < original.Cols(); j++ {
			assert.InDelta(t, origRow[j], resultRow[j], 0.0001)
		}
	}
}

// TestRoundTrip_TensorWithDtype tests round-trip with dtype comment
func TestRoundTrip_TensorWithDtype(t *testing.T) {
	m := NewMarshaller()
	u := NewUnmarshaller(types.WithHint("tensor"))

	original := tensor.FromArray(tensor.NewShape(5), []float32{1.5, 2.5, 3.5, 4.5, 5.5})

	var buf bytes.Buffer
	err := m.Marshal(&buf, original)
	require.NoError(t, err)

	// Verify dtype comment is present
	output := buf.String()
	assert.Contains(t, output, "# dtype: FP32")

	var result types.Tensor
	err = u.Unmarshal(&buf, &result)
	require.NoError(t, err)

	assert.Equal(t, original.Shape(), result.Shape())
	assert.Equal(t, original.DataType(), result.DataType())
	origData := original.Data().([]float32)
	resultData := result.Data().([]float32)
	for i := 0; i < len(origData); i++ {
		assert.InDelta(t, origData[i], resultData[i], 0.0001)
	}
}

// TestRoundTrip_VectorWithHeader tests round-trip vector with header
func TestRoundTrip_VectorWithHeader(t *testing.T) {
	m := NewMarshaller(WithHeader(true))
	u := NewUnmarshaller(WithHeader(true))

	original := vec.NewFrom(1.0, 2.0, 3.0, 4.0, 5.0)

	var buf bytes.Buffer
	err := m.Marshal(&buf, original)
	require.NoError(t, err)

	var result vec.Vector
	err = u.Unmarshal(&buf, &result)
	require.NoError(t, err)

	assert.Equal(t, original.Len(), result.Len())
	origView := original.View().(vec.Vector)
	resultView := result.View().(vec.Vector)
	for i := 0; i < original.Len(); i++ {
		assert.InDelta(t, origView[i], resultView[i], 0.0001)
	}
}

// TestVectorColumnOrientation tests reading/writing vectors as columns

func TestMarshaller_MarshalVectorAsColumn(t *testing.T) {
	m := NewMarshaller(WithOrientation("column"))
	v := vec.NewFrom(1.0, 2.0, 3.0, 4.0, 5.0)

	var buf bytes.Buffer
	err := m.Marshal(&buf, v)
	require.NoError(t, err)

	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	// Account for dtype comment line + 5 data rows
	assert.GreaterOrEqual(t, len(lines), 6)
	assert.Contains(t, output, "# dtype: FP32")
	assert.Contains(t, output, "1")
	assert.Contains(t, output, "5")
}

func TestMarshaller_MarshalVectorAsColumnWithHeader(t *testing.T) {
	m := NewMarshaller(WithOrientation("column"), WithHeader(true))
	v := vec.NewFrom(1.0, 2.0, 3.0)

	var buf bytes.Buffer
	err := m.Marshal(&buf, v)
	require.NoError(t, err)

	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	// Account for dtype comment line + header + 3 data rows
	assert.GreaterOrEqual(t, len(lines), 5)
	assert.Contains(t, output, "# dtype: FP32")
	assert.Contains(t, output, "value")
}

func TestUnmarshaller_UnmarshalVectorAsColumn(t *testing.T) {
	u := NewUnmarshaller(WithOrientation("column"))
	csvData := "1.0\n2.0\n3.0\n"

	var result vec.Vector
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	assert.Equal(t, 3, result.Len())
	assert.Equal(t, float32(1.0), result.View().(vec.Vector)[0])
	assert.Equal(t, float32(3.0), result.View().(vec.Vector)[2])
}

func TestUnmarshaller_UnmarshalVectorAsColumnWithHeader(t *testing.T) {
	u := NewUnmarshaller(WithOrientation("column"), WithHeader(true))
	csvData := "value\n1.0\n2.0\n3.0\n"

	var result vec.Vector
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	assert.Equal(t, 3, result.Len())
	assert.Equal(t, float32(1.0), result.View().(vec.Vector)[0])
	assert.Equal(t, float32(3.0), result.View().(vec.Vector)[2])
}

func TestUnmarshaller_FromTestData_VectorColumn(t *testing.T) {
	u := NewUnmarshaller(WithOrientation("column"))

	var result vec.Vector
	err := u.Unmarshal(strings.NewReader(vectorColumnCSV), &result)
	require.NoError(t, err)

	assert.Equal(t, 5, result.Len())
	resultView := result.View().(vec.Vector)
	assert.InDelta(t, 1.0, resultView[0], 0.0001)
	assert.InDelta(t, 5.0, resultView[4], 0.0001)
}

func TestUnmarshaller_FromTestData_VectorColumnWithHeader(t *testing.T) {
	u := NewUnmarshaller(WithOrientation("column"), WithHeader(true))

	var result vec.Vector
	err := u.Unmarshal(strings.NewReader(vectorColumnWithHeaderCSV), &result)
	require.NoError(t, err)

	assert.Equal(t, 5, result.Len())
	resultView := result.View().(vec.Vector)
	assert.InDelta(t, 1.0, resultView[0], 0.0001)
	assert.InDelta(t, 5.0, resultView[4], 0.0001)
}

// TestSliceColumnOrientation tests reading/writing slices as columns

func TestMarshaller_MarshalSliceAsColumn(t *testing.T) {
	m := NewMarshaller(WithOrientation("column"))
	slice := []float32{1.0, 2.0, 3.0}

	var buf bytes.Buffer
	err := m.Marshal(&buf, slice)
	require.NoError(t, err)

	output := buf.String()
	lines := strings.Split(strings.TrimSpace(output), "\n")
	// Account for dtype comment line + 3 data rows
	assert.GreaterOrEqual(t, len(lines), 4)
	assert.Contains(t, output, "# dtype: FP32")
}

func TestUnmarshaller_UnmarshalSliceAsColumn(t *testing.T) {
	u := NewUnmarshaller(WithOrientation("column"))
	csvData := "1.0\n2.0\n3.0\n"

	var result []float32
	err := u.Unmarshal(strings.NewReader(csvData), &result)
	require.NoError(t, err)

	assert.Equal(t, 3, len(result))
	assert.Equal(t, float32(1.0), result[0])
	assert.Equal(t, float32(3.0), result[2])
}

func TestUnmarshaller_FromTestData_SliceFloat32Column(t *testing.T) {
	u := NewUnmarshaller(WithOrientation("column"))

	var result []float32
	err := u.Unmarshal(strings.NewReader(sliceFloat32ColumnCSV), &result)
	require.NoError(t, err)

	assert.Equal(t, 3, len(result))
	assert.Equal(t, float32(1.0), result[0])
	assert.Equal(t, float32(3.0), result[2])
}

// TestRoundTrip_VectorAsColumn tests round-trip with column orientation
func TestRoundTrip_VectorAsColumn(t *testing.T) {
	m := NewMarshaller(WithOrientation("column"))
	u := NewUnmarshaller(WithOrientation("column"))

	original := vec.NewFrom(1.0, 2.0, 3.0, 4.0, 5.0)

	var buf bytes.Buffer
	err := m.Marshal(&buf, original)
	require.NoError(t, err)

	var result vec.Vector
	err = u.Unmarshal(&buf, &result)
	require.NoError(t, err)

	assert.Equal(t, original.Len(), result.Len())
	origView := original.View().(vec.Vector)
	resultView := result.View().(vec.Vector)
	for i := 0; i < original.Len(); i++ {
		assert.InDelta(t, origView[i], resultView[i], 0.0001)
	}
}

// TestRoundTrip_SliceAsColumn tests round-trip slice with column orientation
func TestRoundTrip_SliceAsColumn(t *testing.T) {
	m := NewMarshaller(WithOrientation("column"))
	u := NewUnmarshaller(WithOrientation("column"))

	original := []float32{1.0, 2.0, 3.0, 4.0, 5.0}

	var buf bytes.Buffer
	err := m.Marshal(&buf, original)
	require.NoError(t, err)

	var result []float32
	err = u.Unmarshal(&buf, &result)
	require.NoError(t, err)

	assert.Equal(t, len(original), len(result))
	for i := 0; i < len(original); i++ {
		assert.InDelta(t, original[i], result[i], 0.0001)
	}
}
