package mat

import (
	"reflect"
	"testing"

	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

func TestNew4x4(t *testing.T) {
	type args struct {
		arr []float32
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 Matrix4x4
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := New4x4(tArgs.arr...)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("New4x4 got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Flat(t *testing.T) {
	type args struct {
		v vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix4x4.Flat got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Matrix(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix4x4.Matrix got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_RotationX(t *testing.T) {
	type args struct {
		a float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.RotationX got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_RotationY(t *testing.T) {
	type args struct {
		a float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.RotationY got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_RotationZ(t *testing.T) {
	type args struct {
		a float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.RotationZ got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Orientation(t *testing.T) {
	type args struct {
		q vec.Quaternion
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.Orientation got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Eye(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		want1 *Matrix4x4
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Eye()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x4.Eye got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Row(t *testing.T) {
	type args struct {
		row int
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix4x4.Row got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Col(t *testing.T) {
	type args struct {
		col int
		v   vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix4x4.Col got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_SetRow(t *testing.T) {
	type args struct {
		row int
		v   vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.SetRow got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_SetCol(t *testing.T) {
	type args struct {
		col int
		v   vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.SetCol got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Diagonal(t *testing.T) {
	type args struct {
		dst vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 vec.Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Diagonal(tArgs.dst)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x4.Diagonal got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_SetDiagonal(t *testing.T) {
	type args struct {
		v vec.Vector4D
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.SetDiagonal(tArgs.v)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x4.SetDiagonal got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Submatrix(t *testing.T) {
	type args struct {
		row int
		col int
		m1  Matrix
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix4x4.Submatrix got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_SetSubmatrix(t *testing.T) {
	type args struct {
		row int
		col int
		m1  Matrix
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.SetSubmatrix got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_SetSubmatrixRaw(t *testing.T) {
	type args struct {
		row   int
		col   int
		rows1 int
		cols1 int
		m1    []float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.SetSubmatrixRaw got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Clone(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.Clone got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Transpose(t *testing.T) {
	type args struct {
		m1 Matrix4x4
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.Transpose got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Add(t *testing.T) {
	type args struct {
		m1 Matrix4x4
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.Add got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Sub(t *testing.T) {
	type args struct {
		m1 Matrix4x4
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.Sub got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_MulC(t *testing.T) {
	type args struct {
		c float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.MulC got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_DivC(t *testing.T) {
	type args struct {
		c float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.DivC got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Mul(t *testing.T) {
	type args struct {
		a Matrix4x4
		b Matrix4x4
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
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
				t.Errorf("Matrix4x4.Mul got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_MulDiag(t *testing.T) {
	type args struct {
		a Matrix4x4
		b vec.Vector4D
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix4x4
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.MulDiag(tArgs.a, tArgs.b)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix4x4.MulDiag got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_MulVec(t *testing.T) {
	type args struct {
		v   vec.Vector4D
		dst vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix4x4.MulVec got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_MulVecT(t *testing.T) {
	type args struct {
		v   vec.Vector4D
		dst vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix4x4.MulVecT got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_Det(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix4x4.Det got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix4x4_LU(t *testing.T) {
	type args struct {
		L *Matrix4x4
		U *Matrix4x4
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			receiver.LU(tArgs.L, tArgs.U)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

		})
	}
}

func TestMatrix4x4_Quaternion(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		inspect func(r *Matrix4x4, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix4x4.Quaternion got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}
