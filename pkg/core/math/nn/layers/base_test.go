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
			base := NewBase("")
			base.ParseOptions(WithName(tt.input))
			assert.Equal(t, tt.expected, base.Name(), "Name should match")
			assert.False(t, base.CanLearn(), "Default CanLearn should be false")
		})
	}
}

func TestBase_Name(t *testing.T) {
	base := NewBase("")
	base.ParseOptions(WithName("test"))
	assert.Equal(t, "test", base.Name(), "Name should match")

	// Test nil receiver
	var nilBase *Base
	assert.Equal(t, "", nilBase.Name(), "Nil base should return empty name")
}

func TestBase_SetName(t *testing.T) {
	base := NewBase("")
	base.ParseOptions(WithName("old_name"))
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
			var opts []Option
			if tt.existingName != "" {
				opts = append(opts, WithName(tt.existingName))
			}
			base := NewBase(tt.prefix)
			base.ParseOptions(opts...)
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
	input := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})

	base.StoreInput(input)
	retrieved := base.Input()
	assert.Equal(t, input.Shape().ToSlice(), retrieved.Shape().ToSlice(), "Input dimensions should match")
	assert.Equal(t, input.Data(), retrieved.Data(), "Input data should match")

	// Test nil receiver
	var nilBase *Base
	retrieved = nilBase.Input()
	assert.Nil(t, retrieved, "Nil base should return nil tensor")
}

func TestBase_Output(t *testing.T) {
	base := NewBase("")
	output := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})

	base.StoreOutput(output)
	retrieved := base.Output()
	assert.Equal(t, output.Shape().ToSlice(), retrieved.Shape().ToSlice(), "Output dimensions should match")
	assert.Equal(t, output.Data(), retrieved.Data(), "Output data should match")

	// Test nil receiver
	var nilBase *Base
	retrieved = nilBase.Output()
	assert.Nil(t, retrieved, "Nil base should return nil tensor")
}

func TestBase_Grad(t *testing.T) {
	base := NewBase("")
	grad := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})

	base.StoreGrad(grad)
	retrieved := base.Grad()
	assert.Equal(t, grad.Shape().ToSlice(), retrieved.Shape().ToSlice(), "Grad dimensions should match")
	assert.Equal(t, grad.Data(), retrieved.Data(), "Grad data should match")

	// Test nil receiver
	var nilBase *Base
	retrieved = nilBase.Grad()
	assert.Nil(t, retrieved, "Nil base should return nil tensor")
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
			assert.Equal(t, tt.expected, output.Shape().ToSlice(), "Output dimensions should match")
			assert.Equal(t, tt.size, output.Size(), "Output data size should match")

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
			assert.Equal(t, tt.expected, grad.Shape().ToSlice(), "Grad dimensions should match")
			assert.Equal(t, tt.size, grad.Size(), "Grad data size should match")

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
					Data:         tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 2.0}),
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
		Data:         tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 2.0}),
		RequiresGrad: true,
	})
	base.SetParam(ParamBiases, Parameter{
		Data:         tensor.FromFloat32(tensor.NewShape(2), []float32{3.0, 4.0}),
		RequiresGrad: true,
	})

	// Test valid index
	param, ok := base.Parameter(ParamWeights)
	assert.True(t, ok, "Parameter should exist")
	assert.True(t, len(param.Data.Shape().ToSlice()) > 0, "Parameter should have shape")

	param, ok = base.Parameter(ParamBiases)
	assert.True(t, ok, "Parameter should exist")
	assert.True(t, len(param.Data.Shape().ToSlice()) > 0, "Parameter should have shape")

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
		Data:         tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 2.0}),
		RequiresGrad: true,
	})
	base.SetParam(ParamBiases, Parameter{
		Data:         tensor.FromFloat32(tensor.NewShape(2), []float32{3.0, 4.0}),
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
			Data:         tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 2.0}),
			RequiresGrad: true,
		},
		ParamBiases: {
			Data:         tensor.FromFloat32(tensor.NewShape(3), []float32{3.0, 4.0, 5.0}),
			RequiresGrad: false,
		},
	}

	err := base.SetParameters(newParams)
	require.NoError(t, err, "SetParameters should succeed")

	// Verify parameters were set
	params := base.Parameters()
	require.Len(t, params, 2, "Parameters should have length 2")
	weights := params[ParamWeights]
	newWeights := newParams[ParamWeights]
	assert.Equal(t, newWeights.Data.Shape().ToSlice(), weights.Data.Shape().ToSlice(), "Weights parameter shape should match")

	biases := params[ParamBiases]
	newBiases := newParams[ParamBiases]
	assert.Equal(t, newBiases.Data.Shape().ToSlice(), biases.Data.Shape().ToSlice(), "Biases parameter shape should match")

	// Test nil receiver
	var nilBase *Base
	err = nilBase.SetParameters(newParams)
	assert.Error(t, err, "SetParameters should error for nil receiver")
}

func TestBase_ZeroGrad(t *testing.T) {
	base := NewBase("")
	base.SetParam(ParamWeights, Parameter{
		Data:         tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 2.0}),
		RequiresGrad: true,
		Grad:         tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 2.0}),
	})
	base.SetParam(ParamBiases, Parameter{
		Data:         tensor.FromFloat32(tensor.NewShape(3), []float32{3.0, 4.0, 5.0}),
		RequiresGrad: true,
		Grad:         tensor.FromFloat32(tensor.NewShape(3), []float32{3.0, 4.0, 5.0}),
	})

	// Zero gradients
	base.ZeroGrad()

	// Verify gradients are zeroed
	param0, _ := base.Parameter(ParamWeights)
	param0GradData := param0.Grad.Data().([]float32)
	for i := 0; i < len(param0GradData); i++ {
		assert.Equal(t, float32(0.0), param0GradData[i], "Param0 grad should be zero")
	}
	param1, _ := base.Parameter(ParamBiases)
	param1GradData := param1.Grad.Data().([]float32)
	for i := 0; i < len(param1GradData); i++ {
		assert.Equal(t, float32(0.0), param1GradData[i], "Param1 grad should be zero")
	}

	// Test nil receiver
	var nilBase *Base
	nilBase.ZeroGrad()
}

func TestBase_StoreInput(t *testing.T) {
	base := NewBase("")
	input := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})

	base.StoreInput(input)
	retrieved := base.Input()
	assert.Equal(t, input.Shape().ToSlice(), retrieved.Shape().ToSlice(), "Input dimensions should match")
	assert.Equal(t, input.Data(), retrieved.Data(), "Input data should match")
}

func TestBase_StoreOutput(t *testing.T) {
	base := NewBase("")
	output := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})

	base.StoreOutput(output)
	retrieved := base.Output()
	assert.Equal(t, output.Shape().ToSlice(), retrieved.Shape().ToSlice(), "Output dimensions should match")
	assert.Equal(t, output.Data(), retrieved.Data(), "Output data should match")
}

func TestBase_StoreGrad(t *testing.T) {
	base := NewBase("")
	grad := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6})

	base.StoreGrad(grad)
	retrieved := base.Grad()
	assert.Equal(t, grad.Shape().ToSlice(), retrieved.Shape().ToSlice(), "Grad dimensions should match")
	assert.Equal(t, grad.Data(), retrieved.Data(), "Grad data should match")
}

func TestBase_ParseOptions(t *testing.T) {
	base := NewBase("test")

	// Test parsing multiple options
	base.ParseOptions(
		WithName("parsed_name"),
		WithCanLearn(true),
	)

	assert.Equal(t, "parsed_name", base.Name(), "Name should be set via option")
	assert.True(t, base.CanLearn(), "CanLearn should be set via option")

	// Test nil receiver
	var nilBase *Base
	nilBase.ParseOptions(WithName("should_not_crash"))
}

func TestBase_BiasHint(t *testing.T) {
	base := NewBase("")

	// Test without bias hint
	assert.Nil(t, base.BiasHint(), "BiasHint should be nil initially")

	// Test with bias hint
	base.ParseOptions(UseBias(true))
	hint := base.BiasHint()
	require.NotNil(t, hint, "BiasHint should not be nil after setting")
	assert.True(t, *hint, "BiasHint should be true")

	// Test with false bias hint
	base2 := NewBase("")
	base2.ParseOptions(UseBias(false))
	hint2 := base2.BiasHint()
	require.NotNil(t, hint2, "BiasHint should not be nil after setting")
	assert.False(t, *hint2, "BiasHint should be false")

	// Test nil receiver
	var nilBase *Base
	assert.Nil(t, nilBase.BiasHint(), "BiasHint should be nil for nil receiver")
}

func TestBase_Weights(t *testing.T) {
	base := NewBase("")

	// Test without weights
	weights := base.Weights()
	assert.True(t, tensor.IsNil(weights.Data), "Weights should be empty initially")

	// Test with weights
	weightTensor := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
	base.SetParam(ParamWeights, Parameter{
		Data:         weightTensor,
		RequiresGrad: true,
	})
	weights = base.Weights()
	assert.False(t, tensor.IsNil(weights.Data), "Weights should exist after setting")
	assert.Equal(t, weightTensor.Shape().ToSlice(), weights.Data.Shape().ToSlice(), "Weights shape should match")

	// Test nil receiver
	var nilBase *Base
	weights = nilBase.Weights()
	assert.True(t, tensor.IsNil(weights.Data), "Weights should be empty for nil receiver")
}

func TestBase_Biases(t *testing.T) {
	base := NewBase("")

	// Test without biases
	biases := base.Biases()
	assert.True(t, tensor.IsNil(biases.Data), "Biases should be empty initially")

	// Test with biases
	biasTensor := tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0})
	base.SetParam(ParamBiases, Parameter{
		Data:         biasTensor,
		RequiresGrad: true,
	})
	biases = base.Biases()
	assert.False(t, tensor.IsNil(biases.Data), "Biases should exist after setting")
	assert.Equal(t, biasTensor.Shape().ToSlice(), biases.Data.Shape().ToSlice(), "Biases shape should match")

	// Test nil receiver
	var nilBase *Base
	biases = nilBase.Biases()
	assert.True(t, tensor.IsNil(biases.Data), "Biases should be empty for nil receiver")
}

func TestBase_Kernels(t *testing.T) {
	base := NewBase("")

	// Test without kernels
	kernels := base.Kernels()
	assert.True(t, tensor.IsNil(kernels.Data), "Kernels should be empty initially")

	// Test with kernels
	kernelTensor := tensor.FromFloat32(tensor.NewShape(16, 3, 3, 3), make([]float32, 16*3*3*3))
	base.SetParam(ParamKernels, Parameter{
		Data:         kernelTensor,
		RequiresGrad: true,
	})
	kernels = base.Kernels()
	assert.False(t, tensor.IsNil(kernels.Data), "Kernels should exist after setting")
	assert.Equal(t, kernelTensor.Shape().ToSlice(), kernels.Data.Shape().ToSlice(), "Kernels shape should match")

	// Test nil receiver
	var nilBase *Base
	kernels = nilBase.Kernels()
	assert.True(t, tensor.IsNil(kernels.Data), "Kernels should be empty for nil receiver")
}
