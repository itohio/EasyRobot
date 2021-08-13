package mat

import (
	"reflect"
	"testing"

	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

func TestNew4x3(t *testing.T) {
	type args struct {
		arr []float32
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 Matrix4x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := New4x3(tArgs.arr...)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("New4x3 got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_Flat(t *testing.T) {
	type args struct {
		v vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 vec.Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Flat(tArgs.v)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.Flat got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_Matrix(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		want1 Matrix
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Matrix()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.Matrix got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_RotationX(t *testing.T) {
	type args struct {
		a float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
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
				t.Errorf("Matrix4x3.RotationX got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_RotationY(t *testing.T) {
	type args struct {
		a float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
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
				t.Errorf("Matrix4x3.RotationY got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_RotationZ(t *testing.T) {
	type args struct {
		a float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
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
				t.Errorf("Matrix4x3.RotationZ got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_Orientation(t *testing.T) {
	type args struct {
		q vec.Quaternion
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
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
				t.Errorf("Matrix4x3.Orientation got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_Row(t *testing.T) {
	type args struct {
		row int
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 vec.Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Row(tArgs.row)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.Row got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_Col(t *testing.T) {
	type args struct {
		col int
		v   vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 vec.Vector
	}{
		//TODO: Add test cases
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
				t.Errorf("Matrix4x3.Col got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_SetRow(t *testing.T) {
	type args struct {
		row int
		v   vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.SetRow(tArgs.row, tArgs.v)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.SetRow got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_SetCol(t *testing.T) {
	type args struct {
		col int
		v   vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.SetCol(tArgs.col, tArgs.v)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.SetCol got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_Submatrix(t *testing.T) {
	type args struct {
		row int
		col int
		m1  Matrix
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Matrix
	}{
		//TODO: Add test cases
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
				t.Errorf("Matrix4x3.Submatrix got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_SetSubmatrix(t *testing.T) {
	type args struct {
		row int
		col int
		m1  Matrix
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
	}{
		//TODO: Add test cases
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
				t.Errorf("Matrix4x3.SetSubmatrix got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_SetSubmatrixRaw(t *testing.T) {
	type args struct {
		row   int
		col   int
		rows1 int
		cols1 int
		m1    []float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
	}{
		//TODO: Add test cases
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
				t.Errorf("Matrix4x3.SetSubmatrixRaw got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_Clone(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		want1 *Matrix4x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Clone()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.Clone got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_Transpose(t *testing.T) {
	type args struct {
		m1 Matrix3x4
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Transpose(tArgs.m1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.Transpose got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_Add(t *testing.T) {
	type args struct {
		m1 Matrix4x3
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Add(tArgs.m1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.Add got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_Sub(t *testing.T) {
	type args struct {
		m1 Matrix4x3
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Sub(tArgs.m1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.Sub got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_MulC(t *testing.T) {
	type args struct {
		c float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.MulC(tArgs.c)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.MulC got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_DivC(t *testing.T) {
	type args struct {
		c float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.DivC(tArgs.c)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.DivC got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_Mul(t *testing.T) {
	type args struct {
		a Matrix
		b Matrix3x4
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x3
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Mul(tArgs.a, tArgs.b)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.Mul got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_MulVec(t *testing.T) {
	type args struct {
		v   vec.Vector3D
		dst vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 vec.Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.MulVec(tArgs.v, tArgs.dst)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.MulVec got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_MulVecT(t *testing.T) {
	type args struct {
		v   vec.Vector4D
		dst vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 vec.Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.MulVecT(tArgs.v, tArgs.dst)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x3.MulVecT got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x3_Quaternion(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x3
		inspect func(r *Matrix4x3, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix4x3.Quaternion got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}
