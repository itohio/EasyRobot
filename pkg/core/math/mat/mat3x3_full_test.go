package mat

import (
	"reflect"
	"testing"

	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

func TestMatrix3x3_Flat(t *testing.T) {
	type args struct {
		v vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 vec.Vector
	}{
		{
			"new",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			nil,
			func(t *testing.T) args {
				return args{vec.New(9)}
			},
			vec.Vector{1, 2, 3, 4, 5, 6, 7, 8, 9},
		},
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			func(r *Matrix3x3, t *testing.T) {
				r[1][1] = 123
			},
			func(t *testing.T) args {
				return args{vec.New(9)}
			},
			vec.Vector{1, 2, 3, 4, 5, 6, 7, 8, 9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Flat(tArgs.v)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Flat got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Matrix(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		want1 Matrix
	}{
		{
			"new",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			nil,
			New(3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9),
		},
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			func(r *Matrix3x3, t *testing.T) {
				r[1][1] = 123
			},
			New(3, 3, 1, 2, 3, 4, 123, 6, 7, 8, 9),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Matrix()

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Matrix got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}
func TestNew3x3(t *testing.T) {
	type args struct {
		rows int
		cols int
		arr  []float32
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 Matrix3x3
	}{
		{
			"new",
			func(t *testing.T) args {
				return args{
					3, 3,
					[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
				}
			},
			Matrix3x3{
				[3]float32{1, 2, 3},
				[3]float32{4, 5, 6},
				[3]float32{7, 8, 9},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := New3x3(tArgs.arr...)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("New3x3 got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_RotationX(t *testing.T) {
	type args struct {
		a float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Matrix3x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.RotationX(tArgs.a)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.RotationX got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_RotationY(t *testing.T) {
	type args struct {
		a float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Matrix3x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.RotationY(tArgs.a)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.RotationY got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_RotationZ(t *testing.T) {
	type args struct {
		a float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Matrix3x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.RotationZ(tArgs.a)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.RotationZ got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Orientation(t *testing.T) {
	type args struct {
		q vec.Quaternion
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Matrix3x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Orientation(tArgs.q)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Orientation got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Eye(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		want1 *Matrix3x3
	}{
		{
			"new",
			func(t *testing.T) Matrix3x3 {
				return New3x3()
			},
			nil,
			&Matrix3x3{
				[3]float32{1, 0, 0},
				[3]float32{0, 1, 0},
				[3]float32{0, 0, 1},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Eye()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Eye got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Row(t *testing.T) {
	type args struct {
		row int
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 vec.Vector
	}{
		{
			"new",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			nil,
			func(t *testing.T) args {
				return args{1}
			},
			vec.Vector{4, 5, 6},
		},
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			func(r *Matrix3x3, t *testing.T) { r[1][1] = 123 },
			func(t *testing.T) args {
				return args{1}
			},
			vec.Vector{4, 123, 6},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Row(tArgs.row)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Row got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Col(t *testing.T) {
	type args struct {
		col int
		v   vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 vec.Vector
	}{
		{
			"new",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			nil,
			func(t *testing.T) args {
				return args{1, vec.New(3)}
			},
			vec.Vector{2, 5, 8},
		},
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			func(r Matrix3x3, t *testing.T) {
				r[1][1] = 123
			},
			func(t *testing.T) args {
				return args{1, vec.New(3)}
			},
			vec.Vector{2, 5, 8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Col(tArgs.col, tArgs.v)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Col got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_SetRow(t *testing.T) {
	type args struct {
		row int
		v   vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix3x3
	}{
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
			},
			func(r *Matrix3x3, t *testing.T) {
				r[0][0] = 123
			},
			func(t *testing.T) args {
				return args{1, vec.NewFrom(11, 22, 33)}
			},
			&Matrix3x3{{123, 2, 3}, {11, 22, 33}, {7, 8, 9}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.SetRow(tArgs.row, tArgs.v)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.SetRow got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_SetCol(t *testing.T) {
	type args struct {
		col int
		v   vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix3x3
	}{
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
			},
			func(r *Matrix3x3, t *testing.T) {
				r[0][0] = 123
			},
			func(t *testing.T) args {
				return args{1, vec.NewFrom(11, 22, 33)}
			},
			&Matrix3x3{{123, 11, 3}, {4, 22, 6}, {7, 33, 9}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.SetCol(tArgs.col, tArgs.v)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.SetCol got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Diagonal(t *testing.T) {
	type args struct {
		v vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 vec.Vector
	}{
		{
			"new",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			nil,
			func(t *testing.T) args {
				return args{vec.New(3)}
			},
			vec.Vector{1, 5, 9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Diagonal(tArgs.v)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Diagonal got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_SetDiagonal(t *testing.T) {
	type args struct {
		v vec.Vector3D
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix3x3
	}{
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
			},
			func(r *Matrix3x3, t *testing.T) {
				r[0][1] = 123
			},
			func(t *testing.T) args {
				return args{vec.Vector3D{11, 22, 33}}
			},
			&Matrix3x3{{11, 123, 3}, {4, 22, 6}, {7, 8, 33}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.SetDiagonal(tArgs.v)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.SetDiagonal got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Submatrix(t *testing.T) {
	type args struct {
		row int
		col int
		m1  Matrix
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Matrix
	}{
		{
			"new 0x0",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			nil,
			func(t *testing.T) args {
				return args{0, 0, New(2, 2)}
			},
			New(2, 2, 1, 2, 4, 5),
		},
		{
			"new 1x1",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			nil,
			func(t *testing.T) args {
				return args{1, 1, New(2, 2)}
			},
			New(2, 2, 5, 6, 8, 9),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Submatrix(tArgs.row, tArgs.col, tArgs.m1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Submatrix got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_SetSubmatrix(t *testing.T) {
	type args struct {
		row int
		col int
		m1  Matrix
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix3x3
	}{
		{
			"new 0x0",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			nil,
			func(t *testing.T) args {
				return args{0, 0, New(2, 2, 11, 22, 33, 44)}
			},
			&Matrix3x3{{11, 22, 3}, {33, 44, 6}, {7, 8, 9}},
		},
		{
			"new 1x1",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			nil,
			func(t *testing.T) args {
				return args{1, 1, New(2, 2, 11, 22, 33, 44)}
			},
			&Matrix3x3{{1, 2, 3}, {4, 11, 22}, {7, 33, 44}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.SetSubmatrix(tArgs.row, tArgs.col, tArgs.m1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.SetSubmatrix got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_SetSubmatrixRaw(t *testing.T) {
	type args struct {
		row   int
		col   int
		rows1 int
		cols1 int
		m1    []float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix3x3
	}{
		{
			"new 0x0",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			nil,
			func(t *testing.T) args {
				return args{0, 0, 2, 2, []float32{11, 22, 33, 44}}
			},
			&Matrix3x3{{11, 22, 3}, {33, 44, 6}, {7, 8, 9}},
		},
		{
			"new 1x1",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			nil,
			func(t *testing.T) args {
				return args{1, 1, 2, 2, []float32{11, 22, 33, 44}}
			},
			&Matrix3x3{{1, 2, 3}, {4, 11, 22}, {7, 33, 44}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.SetSubmatrixRaw(tArgs.row, tArgs.col, tArgs.rows1, tArgs.cols1, tArgs.m1...)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.SetSubmatrixRaw got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Clone(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		want1 *Matrix3x3
	}{
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			func(r Matrix3x3, t *testing.T) {
				r[0][0] = 123
			},
			&Matrix3x3{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Clone()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Clone got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Transpose(t *testing.T) {
	type args struct {
		m1 Matrix3x3
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix3x3
	}{
		{
			"new",
			func(t *testing.T) Matrix3x3 {
				return New3x3()
			},
			nil,
			func(t *testing.T) args {
				return args{New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)}
			},
			&Matrix3x3{
				[3]float32{1, 4, 7},
				[3]float32{2, 5, 8},
				[3]float32{3, 6, 9},
			},
		},
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3()
			},
			func(r *Matrix3x3, t *testing.T) {
				r[0][0] = 123
			},
			func(t *testing.T) args {
				return args{New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)}
			},
			&Matrix3x3{
				[3]float32{123, 4, 7},
				[3]float32{2, 5, 8},
				[3]float32{3, 6, 9},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Transpose(tArgs.m1)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Transpose got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Add(t *testing.T) {
	type args struct {
		m1 Matrix3x3
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix3x3
	}{
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			func(r *Matrix3x3, t *testing.T) {
				r[0][0] = 123
			},
			func(t *testing.T) args {
				return args{New3x3(1, 1, 1, 1, 1, 1, 1, 1, 1)}
			},
			&Matrix3x3{
				[3]float32{123, 2 + 1, 3 + 1},
				[3]float32{4 + 1, 5 + 1, 6 + 1},
				[3]float32{7 + 1, 8 + 1, 9 + 1},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Add(tArgs.m1)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Add got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Sub(t *testing.T) {
	type args struct {
		m1 Matrix3x3
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix3x3
	}{
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			func(r *Matrix3x3, t *testing.T) {
				r[0][0] = 123
			},
			func(t *testing.T) args {
				return args{New3x3(1, 1, 1, 1, 1, 1, 1, 1, 1)}
			},
			&Matrix3x3{
				[3]float32{123, 2 - 1, 3 - 1},
				[3]float32{4 - 1, 5 - 1, 6 - 1},
				[3]float32{7 - 1, 8 - 1, 9 - 1},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Sub(tArgs.m1)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Sub got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_MulC(t *testing.T) {
	type args struct {
		c float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix3x3
	}{
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			func(r *Matrix3x3, t *testing.T) {
				r[0][0] = 123
			},
			func(t *testing.T) args {
				return args{2}
			},
			&Matrix3x3{
				[3]float32{123, 2 * 2, 3 * 2},
				[3]float32{4 * 2, 5 * 2, 6 * 2},
				[3]float32{7 * 2, 8 * 2, 9 * 2},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.MulC(tArgs.c)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.MulC got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_DivC(t *testing.T) {
	type args struct {
		c float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix3x3
	}{
		{
			"new modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3(
					1, 2, 3, 4, 5, 6, 7, 8, 9,
				)
			},
			func(r *Matrix3x3, t *testing.T) {
				r[0][0] = 123
			},
			func(t *testing.T) args {
				return args{2}
			},
			&Matrix3x3{
				[3]float32{123, 2.0 / 2.0, 3 / 2.0},
				[3]float32{4 / 2.0, 5 / 2.0, 6 / 2.0},
				[3]float32{7 / 2.0, 8 / 2.0, 9 / 2.0},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.DivC(tArgs.c)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.DivC got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Mul(t *testing.T) {
	type args struct {
		a Matrix3x3
		b Matrix3x3
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix3x3
	}{
		{
			"3x3 vs 3x3 modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3()
			},
			func(r *Matrix3x3, t *testing.T) {
				r[0][0] = 123
			},
			func(t *testing.T) args {
				return args{
					New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9),
					New3x3(0, 1, 1, 0, 2, 2, 3, 3, 3),
				}
			},
			&Matrix3x3{{123, 14, 14}, {18, 32, 32}, {27, 50, 50}},
		},
		{
			"3x3 vs 3x3",
			func(t *testing.T) Matrix3x3 {
				return New3x3()
			},
			nil,
			func(t *testing.T) args {
				return args{
					New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9),
					New3x3(0, 1, 1, 0, 2, 2, 3, 3, 3),
				}
			},
			&Matrix3x3{{9, 14, 14}, {18, 32, 32}, {27, 50, 50}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Mul(tArgs.a, tArgs.b)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Mul got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_MulDiag(t *testing.T) {
	type args struct {
		a Matrix3x3
		b vec.Vector3D
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix3x3
	}{
		{
			"3x3 vs 3vec",
			func(t *testing.T) Matrix3x3 {
				return New3x3()
			},
			nil,
			func(t *testing.T) args {
				return args{
					New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9),
					vec.Vector3D{1, 2, 3},
				}
			},
			&Matrix3x3{{1, 4, 9}, {4, 10, 18}, {7, 16, 27}},
		},
		{
			"3x3 vs 3vec modify",
			func(t *testing.T) Matrix3x3 {
				return New3x3()
			},
			func(r *Matrix3x3, t *testing.T) {
				r[0][0] = 123
			},
			func(t *testing.T) args {
				return args{
					New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9),
					vec.Vector3D{1, 2, 3},
				}
			},
			&Matrix3x3{{123, 4, 9}, {4, 10, 18}, {7, 16, 27}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.MulDiag(tArgs.a, tArgs.b)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.MulDiag got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_MulVec(t *testing.T) {
	type args struct {
		v   vec.Vector3D
		dst vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 vec.Vector
	}{
		{
			"3x3 vs 3vec",
			func(t *testing.T) Matrix3x3 {
				return New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
			},
			nil,
			func(t *testing.T) args {
				return args{
					vec.Vector3D{1, 2, 3},
					vec.New(3),
				}
			},
			vec.NewFrom(14, 32, 50),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.MulVec(tArgs.v, tArgs.dst)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.MulVec got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_MulVecT(t *testing.T) {
	type args struct {
		v   vec.Vector3D
		dst vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r *Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 vec.Vector
	}{
		{
			"3x3 vs 3vec",
			func(t *testing.T) Matrix3x3 {
				return New3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
			},
			nil,
			func(t *testing.T) args {
				return args{
					vec.Vector3D{1, 2, 3},
					vec.New(3),
				}
			},
			vec.NewFrom(30, 36, 42),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.MulVecT(tArgs.v, tArgs.dst)

			if tt.inspect != nil {
				tt.inspect(&receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.MulVecT got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_Det(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		want1 float32
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Det()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Det got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix3x3_LU(t *testing.T) {
	type args struct {
		L Matrix3x3
		U Matrix3x3
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			receiver.LU(&tArgs.L, &tArgs.U)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

		})
	}
}

func TestMatrix3x3_Quaternion(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix3x3
		inspect func(r Matrix3x3, t *testing.T) //inspects receiver after test run

		want1 *vec.Quaternion
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Quaternion()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix3x3.Quaternion got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}
