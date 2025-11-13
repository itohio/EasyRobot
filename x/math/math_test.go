package math

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
)

func TestSQR(t *testing.T) {
	type args struct {
		a float32
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 float32
	}{
		{"2^2", func(t *testing.T) args { return args{2} }, 4},
		{"3^2", func(t *testing.T) args { return args{3} }, 9},
		{"-2^2", func(t *testing.T) args { return args{-2} }, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := SQR(tArgs.a)

			assert.Equal(t, tt.want1, got1)
		})
	}
}

func TestClamp(t *testing.T) {
	type args struct {
		a   float32
		min float32
		max float32
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 float32
	}{
		{"inside", func(t *testing.T) args { return args{1, -1, 1} }, 1},
		{"min", func(t *testing.T) args { return args{-2, -1, 1} }, -1},
		{"max", func(t *testing.T) args { return args{2, -1, 1} }, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := Clamp(tArgs.a, tArgs.min, tArgs.max)

			assert.Equal(t, tt.want1, got1)
		})
	}
}

func TestPytag(t *testing.T) {
	type args struct {
		a float32
		b float32
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 float32
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := Pytag(tArgs.a, tArgs.b)

			assert.Equal(t, tt.want1, got1)
		})
	}
}

func TestFastISqrt(t *testing.T) {
	tests := []struct {
		name string
		args float32

		precision float32
	}{
		{"1", 1, 2e-2},
		{"2", 2, 1e-3},
		{"3", 3, 1e-3},
		{"4", 4, 1e-3},
		{"70", 70, 1e-4},
		{"500", 500, 1e-4},
		{"700000", 700000, 1e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got1 := FastISqrt(tt.args)

			want1 := 1 / math32.Sqrt(tt.args)
			assert.InDelta(t, want1, got1, float64(tt.precision), "FastISqrt")
		})
	}
}
