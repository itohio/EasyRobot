package mat

import (
	"reflect"
	"testing"

	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

func TestNew2x2(t *testing.T) {
	type args struct {
		arr []float32
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 Matrix2x2
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := New2x2(tArgs.arr...)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("New2x2 got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Flat(t *testing.T) {
	type args struct {
		v vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix2x2.Flat got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Matrix(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix2x2.Matrix got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Rotation2D(t *testing.T) {
	type args struct {
		a float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Rotation2D(tArgs.a)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Matrix2x2.Rotation2D got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Eye(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.Eye got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Row(t *testing.T) {
	type args struct {
		row int
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix2x2.Row got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Col(t *testing.T) {
	type args struct {
		col int
		v   vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix2x2.Col got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_SetRow(t *testing.T) {
	type args struct {
		row int
		v   vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.SetRow got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_SetCol(t *testing.T) {
	type args struct {
		col int
		v   vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.SetCol got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Diagonal(t *testing.T) {
	type args struct {
		dst vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix2x2.Diagonal got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_SetDiagonal(t *testing.T) {
	type args struct {
		v vec.Vector2D
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.SetDiagonal got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Submatrix(t *testing.T) {
	type args struct {
		row int
		col int
		m1  Matrix
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix2x2.Submatrix got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_SetSubmatrix(t *testing.T) {
	type args struct {
		row int
		col int
		m1  Matrix
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.SetSubmatrix got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_SetSubmatrixRaw(t *testing.T) {
	type args struct {
		row   int
		col   int
		rows1 int
		cols1 int
		m1    []float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.SetSubmatrixRaw got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Clone(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.Clone got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Transpose(t *testing.T) {
	type args struct {
		m1 Matrix2x2
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.Transpose got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Add(t *testing.T) {
	type args struct {
		m1 Matrix2x2
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.Add got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Sub(t *testing.T) {
	type args struct {
		m1 Matrix2x2
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.Sub got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_MulC(t *testing.T) {
	type args struct {
		c float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.MulC got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_DivC(t *testing.T) {
	type args struct {
		c float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.DivC got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Mul(t *testing.T) {
	type args struct {
		a Matrix2x2
		b Matrix2x2
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.Mul got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_MulDiag(t *testing.T) {
	type args struct {
		a Matrix2x2
		b vec.Vector2D
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 *Matrix2x2
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
				t.Errorf("Matrix2x2.MulDiag got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_MulVec(t *testing.T) {
	type args struct {
		v   vec.Vector2D
		dst vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix2x2.MulVec got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_MulVecT(t *testing.T) {
	type args struct {
		v   vec.Vector2D
		dst vec.Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix2x2.MulVecT got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_Det(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

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
				t.Errorf("Matrix2x2.Det got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestMatrix2x2_LU(t *testing.T) {
	type args struct {
		L *Matrix2x2
		U *Matrix2x2
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		inspect func(r *Matrix2x2, t *testing.T) //inspects receiver after test run

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
