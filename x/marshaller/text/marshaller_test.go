package text

import (
	"bytes"
	"strings"
	"testing"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

// Text marshaller is output-only for diagnostics, so we only test display output

func TestTensorDisplay(t *testing.T) {
	tests := []struct {
		name        string
		shape       []int
		dtype       types.DataType
		wantStrings []string
	}{
		{
			name:        "Vector",
			shape:       []int{5},
			dtype:       types.FP32,
			wantStrings: []string{"Tensor", "fp32", "shape", "5"},
		},
		{
			name:        "Matrix",
			shape:       []int{3, 4},
			dtype:       types.FP32,
			wantStrings: []string{"Tensor", "fp32", "shape", "3", "4"},
		},
		{
			name:        "3DTensor",
			shape:       []int{2, 3, 4},
			dtype:       types.FP64,
			wantStrings: []string{"Tensor", "fp64", "shape", "2", "3", "4"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create tensor
			shape := tensor.NewShape(tt.shape...)
			tens := tensor.New(tt.dtype, shape)

			// Marshal to text
			var buf bytes.Buffer
			m := NewMarshaller()
			if err := m.Marshal(&buf, tens); err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}

			output := buf.String()
			t.Logf("Output:\n%s", output)

			// Verify expected strings are present
			for _, want := range tt.wantStrings {
				if !strings.Contains(strings.ToLower(output), strings.ToLower(want)) {
					t.Errorf("Output missing expected string: %q", want)
				}
			}
		})
	}
}

func TestArrayDisplay(t *testing.T) {
	tests := []struct {
		name        string
		data        any
		wantStrings []string
	}{
		{
			name:        "Float32Array",
			data:        []float32{1.0, 2.5, 3.7, 4.2, 5.9},
			wantStrings: []string{"float32", "5"},
		},
		{
			name:        "Float64Array",
			data:        []float64{1.1, 2.2, 3.3},
			wantStrings: []string{"float64", "3"},
		},
		{
			name:        "IntArray",
			data:        []int{100, 200, 300},
			wantStrings: []string{"int", "3"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			m := NewMarshaller()
			if err := m.Marshal(&buf, tt.data); err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}

			output := buf.String()
			t.Logf("Array output:\n%s", output)

			// Verify array information
			for _, want := range tt.wantStrings {
				if !strings.Contains(strings.ToLower(output), strings.ToLower(want)) {
					t.Errorf("Output missing expected string: %q", want)
				}
			}
		})
	}
}
