package layers

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/nn"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewBase(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "basic",
			input:    "test_layer",
			expected: "test_layer",
		},
		{
			name:     "empty_name",
			input:    "",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := NewBase(WithName(tt.input))
			require.NotNil(t, base, "Base should not be nil")
			assert.Equal(t, tt.expected, base.Name(), "Name should match")
			assert.False(t, base.CanLearn(), "Default CanLearn should be false")
		})
	}
}

func TestBase_Name(t *testing.T) {
	base := NewBase(WithName("test"))
	assert.Equal(t, "test", base.Name(), "Name should match")

	// Test nil receiver
	var nilBase *Base
	assert.Equal(t, "", nilBase.Name(), "Nil base should return empty name")
}

func TestBase_SetName(t *testing.T) {
	base := NewBase(WithName("old_name"))
	base.SetName("new_name")
	assert.Equal(t, "new_name", base.Name(), "Name should be updated")

	// Test nil receiver
	var nilBase *Base
	nilBase.SetName("should_not_crash")
}

func TestBase_SetDefaultName(t *testing.T) {
	tests := []struct {
		name         string
		layerType    string
		shape        []int
		existingName string
		expected     string
	}{
		{
			name:         "with_shape",
			layerType:    "Dense",
			shape:        []int{2, 4},
			existingName: "",
			expected:     "Dense_[2 4]",
		},
		{
			name:         "no_shape",
			layerType:    "Dense",
			shape:        []int{},
			existingName: "",
			expected:     "Dense",
		},
		{
			name:         "existing_name_preserved",
			layerType:    "Dense",
			shape:        []int{2, 4},
			existingName: "custom_name",
			expected:     "custom_name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := NewBase(WithName(tt.existingName))
			base.SetDefaultName(tt.layerType, tt.shape)
			assert.Equal(t, tt.expected, base.Name(), "Name should match expected")

			// Test nil receiver
			var nilBase *Base
			nilBase.SetDefaultName("Dense", []int{2, 4})
		})
	}
}

func TestBase_CanLearn(t *testing.T) {
	base := NewBase()
	assert.False(t, base.CanLearn(), "Default CanLearn should be false")

	base.SetCanLearn(true)
	assert.True(t, base.CanLearn(), "CanLearn should be true")

	// Test nil receiver
	var nilBase *Base
	assert.False(t, nilBase.CanLearn(), "Nil base should return false")
}

func TestBase_SetCanLearn(t *testing.T) {
	base := NewBase()
	base.SetCanLearn(true)
	assert.True(t, base.CanLearn(), "CanLearn should be true")

	base.SetCanLearn(false)
	assert.False(t, base.CanLearn(), "CanLearn should be false")

	// Test nil receiver
	var nilBase *Base
	nilBase.SetCanLearn(true)
}

func TestBase_Input(t *testing.T) {
	base := NewBase()
	input := tensor.Tensor{
		Dim:  []int{2, 3},
		Data: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
	}

	base.StoreInput(input)
	retrieved := base.Input()
	assert.Equal(t, input.Dim, retrieved.Dim, "Input dimensions should match")
	assert.Equal(t, input.Data, retrieved.Data, "Input data should match")

	// Test nil receiver
	var nilBase *Base
	retrieved = nilBase.Input()
	assert.Len(t, retrieved.Dim, 0, "Nil base should return empty tensor")
}

func TestBase_Output(t *testing.T) {
	base := NewBase()
	output := tensor.Tensor{
		Dim:  []int{2, 3},
		Data: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
	}

	base.StoreOutput(output)
	retrieved := base.Output()
	assert.Equal(t, output.Dim, retrieved.Dim, "Output dimensions should match")
	assert.Equal(t, output.Data, retrieved.Data, "Output data should match")

	// Test nil receiver
	var nilBase *Base
	retrieved = nilBase.Output()
	assert.Len(t, retrieved.Dim, 0, "Nil base should return empty tensor")
}

func TestBase_Grad(t *testing.T) {
	base := NewBase()
	grad := tensor.Tensor{
		Dim:  []int{2, 3},
		Data: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
	}

	base.StoreGrad(grad)
	retrieved := base.Grad()
	assert.Equal(t, grad.Dim, retrieved.Dim, "Grad dimensions should match")
	assert.Equal(t, grad.Data, retrieved.Data, "Grad data should match")

	// Test nil receiver
	var nilBase *Base
	retrieved = nilBase.Grad()
	assert.Len(t, retrieved.Dim, 0, "Nil base should return empty tensor")
}

func TestBase_AllocOutput(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		size     int
		expected []int
	}{
		{
			name:     "2d_tensor",
			shape:    []int{2, 3},
			size:     6,
			expected: []int{2, 3},
		},
		{
			name:     "3d_tensor",
			shape:    []int{1, 2, 3},
			size:     6,
			expected: []int{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := NewBase()
			base.AllocOutput(tt.shape, tt.size)
			output := base.Output()
			assert.Equal(t, tt.expected, output.Dim, "Output dimensions should match")
			assert.Len(t, output.Data, tt.size, "Output data size should match")

			// Test nil receiver
			var nilBase *Base
			nilBase.AllocOutput(tt.shape, tt.size)
		})
	}
}

func TestBase_AllocGrad(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		size     int
		expected []int
	}{
		{
			name:     "2d_tensor",
			shape:    []int{2, 3},
			size:     6,
			expected: []int{2, 3},
		},
		{
			name:     "3d_tensor",
			shape:    []int{1, 2, 3},
			size:     6,
			expected: []int{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := NewBase()
			base.AllocGrad(tt.shape, tt.size)
			grad := base.Grad()
			assert.Equal(t, tt.expected, grad.Dim, "Grad dimensions should match")
			assert.Len(t, grad.Data, tt.size, "Grad data size should match")

			// Test nil receiver
			var nilBase *Base
			nilBase.AllocGrad(tt.shape, tt.size)
		})
	}
}

func TestBase_InitParams(t *testing.T) {
	tests := []struct {
		name      string
		numParams int
	}{
		{
			name:      "zero_params",
			numParams: 0,
		},
		{
			name:      "one_param",
			numParams: 1,
		},
		{
			name:      "multiple_params",
			numParams: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := NewBase()
			base.InitParams(tt.numParams)
			params := base.Parameters()
			if tt.numParams == 0 {
				assert.Nil(t, params, "Parameters should be nil for 0 params")
			} else {
				assert.Len(t, params, tt.numParams, "Parameters length should match")
			}

			// Test nil receiver
			var nilBase *Base
			nilBase.InitParams(tt.numParams)
		})
	}
}

func TestBase_Parameter(t *testing.T) {
	base := NewBase()
	base.InitParams(3)

	// Test valid index
	param := base.Parameter(0)
	assert.NotNil(t, param, "Parameter should not be nil")

	param = base.Parameter(2)
	assert.NotNil(t, param, "Parameter should not be nil")

	// Test invalid indices
	param = base.Parameter(-1)
	assert.Nil(t, param, "Parameter should be nil for negative index")

	param = base.Parameter(3)
	assert.Nil(t, param, "Parameter should be nil for out of bounds index")

	// Test nil receiver
	var nilBase *Base
	param = nilBase.Parameter(0)
	assert.Nil(t, param, "Parameter should be nil for nil receiver")
}

func TestBase_Parameters(t *testing.T) {
	base := NewBase()

	// Test with no params
	params := base.Parameters()
	assert.Nil(t, params, "Parameters should be nil when not initialized")

	// Test with params
	base.InitParams(2)
	params = base.Parameters()
	assert.Len(t, params, 2, "Parameters should have length 2")
	assert.NotNil(t, params[0], "First parameter should not be nil")
	assert.NotNil(t, params[1], "Second parameter should not be nil")

	// Test nil receiver
	var nilBase *Base
	params = nilBase.Parameters()
	assert.Nil(t, params, "Parameters should be nil for nil receiver")
}

func TestBase_SetParameters(t *testing.T) {
	base := NewBase()
	base.InitParams(2)

	newParams := []nn.Parameter{
		{
			Data: tensor.Tensor{
				Dim:  []int{2},
				Data: []float32{1.0, 2.0},
			},
			RequiresGrad: true,
		},
		{
			Data: tensor.Tensor{
				Dim:  []int{3},
				Data: []float32{3.0, 4.0, 5.0},
			},
			RequiresGrad: false,
		},
	}

	err := base.SetParameters(newParams)
	require.NoError(t, err, "SetParameters should succeed")

	// Verify parameters were set
	params := base.Parameters()
	require.Len(t, params, 2, "Parameters should have length 2")
	assert.Equal(t, newParams[0].Data.Data, params[0].Data.Data, "First parameter data should match")
	assert.Equal(t, newParams[1].Data.Data, params[1].Data.Data, "Second parameter data should match")

	// Test wrong number of parameters
	err = base.SetParameters([]nn.Parameter{{}})
	assert.Error(t, err, "SetParameters should error for wrong number of params")

	// Test nil receiver
	var nilBase *Base
	err = nilBase.SetParameters(newParams)
	assert.Error(t, err, "SetParameters should error for nil receiver")
}

func TestBase_ZeroGrad(t *testing.T) {
	base := NewBase()
	base.InitParams(2)

	// Set parameter data and enable gradients
	param0 := base.Parameter(0)
	param0.Data = tensor.Tensor{
		Dim:  []int{2},
		Data: []float32{1.0, 2.0},
	}
	param0.RequiresGrad = true
	param0.Grad = tensor.Tensor{
		Dim:  []int{2},
		Data: []float32{1.0, 2.0},
	}

	param1 := base.Parameter(1)
	param1.Data = tensor.Tensor{
		Dim:  []int{3},
		Data: []float32{3.0, 4.0, 5.0},
	}
	param1.RequiresGrad = true
	param1.Grad = tensor.Tensor{
		Dim:  []int{3},
		Data: []float32{3.0, 4.0, 5.0},
	}

	// Zero gradients
	base.ZeroGrad()

	// Verify gradients are zeroed
	for i := 0; i < len(param0.Grad.Data); i++ {
		assert.Equal(t, float32(0.0), param0.Grad.Data[i], "Param0 grad should be zero")
	}
	for i := 0; i < len(param1.Grad.Data); i++ {
		assert.Equal(t, float32(0.0), param1.Grad.Data[i], "Param1 grad should be zero")
	}

	// Test nil receiver
	var nilBase *Base
	nilBase.ZeroGrad()
}

func TestBase_StoreInput(t *testing.T) {
	base := NewBase()
	input := tensor.Tensor{
		Dim:  []int{2, 3},
		Data: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
	}

	base.StoreInput(input)
	retrieved := base.Input()
	assert.Equal(t, input.Dim, retrieved.Dim, "Input dimensions should match")
	assert.Equal(t, input.Data, retrieved.Data, "Input data should match")
}

func TestBase_StoreOutput(t *testing.T) {
	base := NewBase()
	output := tensor.Tensor{
		Dim:  []int{2, 3},
		Data: []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
	}

	base.StoreOutput(output)
	retrieved := base.Output()
	assert.Equal(t, output.Dim, retrieved.Dim, "Output dimensions should match")
	assert.Equal(t, output.Data, retrieved.Data, "Output data should match")
}

func TestBase_StoreGrad(t *testing.T) {
	base := NewBase()
	grad := tensor.Tensor{
		Dim:  []int{2, 3},
		Data: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
	}

	base.StoreGrad(grad)
	retrieved := base.Grad()
	assert.Equal(t, grad.Dim, retrieved.Dim, "Grad dimensions should match")
	assert.Equal(t, grad.Data, retrieved.Data, "Grad data should match")
}
