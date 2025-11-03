package layers

import (
	"testing"

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
			base := NewBase("", WithName(tt.input))
			require.NotNil(t, base, "Base should not be nil")
			assert.Equal(t, tt.expected, base.Name(), "Name should match")
			assert.False(t, base.CanLearn(), "Default CanLearn should be false")
		})
	}
}

func TestBase_Name(t *testing.T) {
	base := NewBase("", WithName("test"))
	assert.Equal(t, "test", base.Name(), "Name should match")

	// Test nil receiver
	var nilBase *Base
	assert.Equal(t, "", nilBase.Name(), "Nil base should return empty name")
}

func TestBase_SetName(t *testing.T) {
	base := NewBase("", WithName("old_name"))
	base.SetName("new_name")
	assert.Equal(t, "new_name", base.Name(), "Name should be updated")

	// Test nil receiver
	var nilBase *Base
	nilBase.SetName("should_not_crash")
}

func TestBase_Name_Default(t *testing.T) {
	tests := []struct {
		name         string
		prefix       string
		shape        []int
		existingName string
		hasShape     bool
	}{
		{
			name:         "with_prefix_and_shape",
			prefix:       "test",
			shape:        []int{2, 4},
			existingName: "",
			hasShape:     true,
		},
		{
			name:         "no_prefix_with_shape",
			prefix:       "",
			shape:        []int{2, 4},
			existingName: "",
			hasShape:     true,
		},
		{
			name:         "existing_name_preserved",
			prefix:       "test",
			shape:        []int{2, 4},
			existingName: "custom_name",
			hasShape:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := NewBase(tt.prefix, WithName(tt.existingName))
			if tt.hasShape {
				base.AllocOutput(tt.shape, 8)
			}
			name := base.Name()
			if tt.existingName != "" {
				assert.Equal(t, tt.existingName, name, "Existing name should be preserved")
			} else {
				assert.NotEmpty(t, name, "Default name should be generated")
			}
		})
	}
}

func TestBase_CanLearn(t *testing.T) {
	base := NewBase("")
	assert.False(t, base.CanLearn(), "Default CanLearn should be false")

	base.SetCanLearn(true)
	assert.True(t, base.CanLearn(), "CanLearn should be true")

	// Test nil receiver
	var nilBase *Base
	assert.False(t, nilBase.CanLearn(), "Nil base should return false")
}

func TestBase_SetCanLearn(t *testing.T) {
	base := NewBase("")
	base.SetCanLearn(true)
	assert.True(t, base.CanLearn(), "CanLearn should be true")

	base.SetCanLearn(false)
	assert.False(t, base.CanLearn(), "CanLearn should be false")

	// Test nil receiver
	var nilBase *Base
	nilBase.SetCanLearn(true)
}

func TestBase_Input(t *testing.T) {
	base := NewBase("")
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
	base := NewBase("")
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
	base := NewBase("")
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
			base := NewBase("")
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
			base := NewBase("")
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

func TestBase_SetParam(t *testing.T) {
	tests := []struct {
		name      string
		numParams int
	}{
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
			base := NewBase("")
			for i := 0; i < tt.numParams; i++ {
				base.SetParam(ParamIndex(i), Parameter{
					Data: tensor.Tensor{
						Dim:  []int{2},
						Data: []float32{1.0, 2.0},
					},
					RequiresGrad: true,
				})
			}
			params := base.Parameters()
			if tt.numParams == 0 {
				assert.Nil(t, params, "Parameters should be nil for 0 params")
			} else {
				assert.Len(t, params, tt.numParams, "Parameters length should match")
			}
		})
	}
}

func TestBase_Parameter(t *testing.T) {
	base := NewBase("")
	base.SetParam(ParamWeights, Parameter{
		Data: tensor.Tensor{
			Dim:  []int{2},
			Data: []float32{1.0, 2.0},
		},
		RequiresGrad: true,
	})
	base.SetParam(ParamBiases, Parameter{
		Data: tensor.Tensor{
			Dim:  []int{2},
			Data: []float32{3.0, 4.0},
		},
		RequiresGrad: true,
	})

	// Test valid index
	param, ok := base.Parameter(ParamWeights)
	assert.True(t, ok, "Parameter should exist")
	assert.NotNil(t, param.Data.Data, "Parameter data should not be nil")

	param, ok = base.Parameter(ParamBiases)
	assert.True(t, ok, "Parameter should exist")
	assert.NotNil(t, param.Data.Data, "Parameter data should not be nil")

	// Test invalid index
	_, ok = base.Parameter(ParamCustom)
	assert.False(t, ok, "Parameter should not exist for unused index")

	// Test nil receiver
	var nilBase *Base
	_, ok = nilBase.Parameter(ParamWeights)
	assert.False(t, ok, "Parameter should not exist for nil receiver")
}

func TestBase_Parameters(t *testing.T) {
	base := NewBase("")

	// Test with no params
	params := base.Parameters()
	assert.Nil(t, params, "Parameters should be nil when empty")

	// Test with params
	base.SetParam(ParamWeights, Parameter{
		Data: tensor.Tensor{
			Dim:  []int{2},
			Data: []float32{1.0, 2.0},
		},
		RequiresGrad: true,
	})
	base.SetParam(ParamBiases, Parameter{
		Data: tensor.Tensor{
			Dim:  []int{2},
			Data: []float32{3.0, 4.0},
		},
		RequiresGrad: false,
	})
	params = base.Parameters()
	assert.Len(t, params, 2, "Parameters should have length 2")
	assert.NotNil(t, params[ParamWeights], "Weights parameter should exist")
	assert.NotNil(t, params[ParamBiases], "Biases parameter should exist")

	// Test nil receiver
	var nilBase *Base
	params = nilBase.Parameters()
	assert.Nil(t, params, "Parameters should be nil for nil receiver")
}

func TestBase_SetParameters(t *testing.T) {
	base := NewBase("")

	newParams := map[ParamIndex]Parameter{
		ParamWeights: {
			Data: tensor.Tensor{
				Dim:  []int{2},
				Data: []float32{1.0, 2.0},
			},
			RequiresGrad: true,
		},
		ParamBiases: {
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
	assert.Equal(t, newParams[ParamWeights].Data.Data, params[ParamWeights].Data.Data, "Weights parameter data should match")
	assert.Equal(t, newParams[ParamBiases].Data.Data, params[ParamBiases].Data.Data, "Biases parameter data should match")

	// Test nil receiver
	var nilBase *Base
	err = nilBase.SetParameters(newParams)
	assert.Error(t, err, "SetParameters should error for nil receiver")
}

func TestBase_ZeroGrad(t *testing.T) {
	base := NewBase("")
	base.SetParam(ParamWeights, Parameter{
		Data: tensor.Tensor{
			Dim:  []int{2},
			Data: []float32{1.0, 2.0},
		},
		RequiresGrad: true,
		Grad: tensor.Tensor{
			Dim:  []int{2},
			Data: []float32{1.0, 2.0},
		},
	})
	base.SetParam(ParamBiases, Parameter{
		Data: tensor.Tensor{
			Dim:  []int{3},
			Data: []float32{3.0, 4.0, 5.0},
		},
		RequiresGrad: true,
		Grad: tensor.Tensor{
			Dim:  []int{3},
			Data: []float32{3.0, 4.0, 5.0},
		},
	})

	// Zero gradients
	base.ZeroGrad()

	// Verify gradients are zeroed
	param0, _ := base.Parameter(ParamWeights)
	for i := 0; i < len(param0.Grad.Data); i++ {
		assert.Equal(t, float32(0.0), param0.Grad.Data[i], "Param0 grad should be zero")
	}
	param1, _ := base.Parameter(ParamBiases)
	for i := 0; i < len(param1.Grad.Data); i++ {
		assert.Equal(t, float32(0.0), param1.Grad.Data[i], "Param1 grad should be zero")
	}

	// Test nil receiver
	var nilBase *Base
	nilBase.ZeroGrad()
}

func TestBase_StoreInput(t *testing.T) {
	base := NewBase("")
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
	base := NewBase("")
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
	base := NewBase("")
	grad := tensor.Tensor{
		Dim:  []int{2, 3},
		Data: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
	}

	base.StoreGrad(grad)
	retrieved := base.Grad()
	assert.Equal(t, grad.Dim, retrieved.Dim, "Grad dimensions should match")
	assert.Equal(t, grad.Data, retrieved.Data, "Grad data should match")
}
