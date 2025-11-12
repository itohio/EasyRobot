package vec

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	type args struct {
		size int
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 Vector
	}{
		{"new", func(t *testing.T) args { return args{15} }, make([]float32, 15)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := New(tArgs.size)

			assert.Equal(t, tt.want1, got1, "New")
		})
	}
}

func TestNewFrom(t *testing.T) {
	type args struct {
		v []float32
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 Vector
	}{
		{"new", func(t *testing.T) args { return args{[]float32{1, 2, 3}} }, []float32{1, 2, 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := NewFrom(tArgs.v...)

			assert.Equal(t, tt.want1, got1, "NewFrom")
		})
	}
}

func TestVector_Sum(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 float32
	}{
		{"sum", func(t *testing.T) Vector { return NewFrom(1, 2, 3) }, nil, 1 + 2 + 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Sum()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Sum")
		})
	}
}

func TestVector_Slice(t *testing.T) {
	type args struct {
		start int
		end   int
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		{
			"modify",
			func(t *testing.T) Vector { return NewFrom(1, 2, 3) },
			func(r Vector, t *testing.T) { r[0] = 123 },
			func(t *testing.T) args { return args{1, -1} },
			NewFrom(2, 3),
		},
		{
			"modify1",
			func(t *testing.T) Vector { return NewFrom(1, 2, 3) },
			func(r Vector, t *testing.T) { r[0] = 123 },
			func(t *testing.T) args { return args{0, 2} },
			NewFrom(123, 2),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Slice(tArgs.start, tArgs.end)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Slice")
		})
	}
}

func TestVector_XY(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 float32
		want2 float32
	}{
		{"new", func(t *testing.T) Vector { return NewFrom(1, 2, 3) }, nil, 1, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1, got2 := receiver.XY()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.XY got1")
			assert.Equal(t, tt.want2, got2, "Vector.XY got2")
		})
	}
}

func TestVector_XYZ(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 float32
		want2 float32
		want3 float32
	}{
		{"new", func(t *testing.T) Vector { return NewFrom(1, 2, 3) }, nil, 1, 2, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1, got2, got3 := receiver.XYZ()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.XYZ got1")
			assert.Equal(t, tt.want2, got2, "Vector.XYZ got2")
			assert.Equal(t, tt.want3, got3, "Vector.XYZ got3")
		})
	}
}

func TestVector_XYZW(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 float32
		want2 float32
		want3 float32
		want4 float32
	}{
		{"new", func(t *testing.T) Vector { return NewFrom(1, 2, 3, 4) }, nil, 1, 2, 3, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1, got2, got3, got4 := receiver.XYZW()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.XYZW got1")
			assert.Equal(t, tt.want2, got2, "Vector.XYZW got2")
			assert.Equal(t, tt.want3, got3, "Vector.XYZW got3")
			assert.Equal(t, tt.want4, got4, "Vector.XYZW got4")
		})
	}
}
func TestVector_SumSqr(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 float32
	}{
		{"sum", func(t *testing.T) Vector { return NewFrom(1, 2, 3) }, nil, 1 + 2*2 + 3*3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.SumSqr()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.SumSqr got1 = %v, want1: %v")
		})
	}
}

func TestVector_Magnitude(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 float32
	}{
		{"sum", func(t *testing.T) Vector { return NewFrom(1, 2, 2) }, nil, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Magnitude()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Magnitude got1 = %v, want1: %v")
		})
	}
}

func TestVector_DistanceSqr(t *testing.T) {
	type args struct {
		v1 Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 float32
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.DistanceSqr(tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.DistanceSqr got1 = %v, want1: %v")
		})
	}
}

func TestVector_Distance(t *testing.T) {
	type args struct {
		v1 Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 float32
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Distance(tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Distance got1 = %v, want1: %v")
		})
	}
}

func TestVector_Clone(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 Vector
	}{
		{
			"modify",
			func(t *testing.T) Vector { return NewFrom(1, 2, 3) },
			func(r Vector, t *testing.T) { r[0] = 123 },
			NewFrom(1, 2, 3),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Clone()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Clone got1 = %v, want1: %v")
		})
	}
}

func TestVector_CopyFrom(t *testing.T) {
	type args struct {
		start int
		v1    Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.CopyFrom(tArgs.start, tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.CopyFrom got1 = %v, want1: %v")
		})
	}
}

func TestVector_CopyTo(t *testing.T) {
	type args struct {
		start int
		v1    Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.CopyTo(tArgs.start, tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.CopyTo got1 = %v, want1: %v")
		})
	}
}

func TestVector_Clamp(t *testing.T) {
	type args struct {
		min Vector
		max Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Clamp(tArgs.min, tArgs.max)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Clamp got1 = %v, want1: %v")
		})
	}
}

func TestVector_FillC(t *testing.T) {
	type args struct {
		c float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.FillC(tArgs.c)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.FillC got1 = %v, want1: %v")
		})
	}
}

func TestVector_Neg(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Neg()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Neg got1 = %v, want1: %v")
		})
	}
}

func TestVector_Add(t *testing.T) {
	type args struct {
		v1 Vector
	}
	var backing []float32
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		{
			name: "adds elements in place",
			init: func(t *testing.T) Vector {
				backing = []float32{1, 2, 3}
				return Vector(backing)
			},
			inspect: func(r Vector, t *testing.T) {
				assert.Equal(t, Vector{4, 6, 8}, r)
				assert.Equal(t, []float32{4, 6, 8}, backing)
			},
			args: func(t *testing.T) args {
				return args{NewFrom(3, 4, 5)}
			},
			want1: NewFrom(4, 6, 8),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			var basePtr *float32
			if len(receiver) > 0 {
				basePtr = &receiver[0]
			}
			got1 := receiver.Add(tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Add got1 = %v, want1: %v")
			if basePtr != nil {
				if res, ok := got1.(Vector); ok && len(res) > 0 {
					assert.True(t, basePtr == &res[0], "Vector.Add should mutate receiver in place")
				}
			}
		})
	}
}

func TestVector_AddDoesNotMutateValueOperands(t *testing.T) {
	receiverBacking := []float32{1, 2}
	receiver := Vector(receiverBacking)
	operand := Vector2D{3, 4}

	got := receiver.Add(operand)

	assert.Equal(t, Vector{4, 6}, got)
	assert.Equal(t, []float32{4, 6}, receiverBacking)
	assert.Equal(t, Vector2D{3, 4}, operand)
}

func TestVector_AddC(t *testing.T) {
	type args struct {
		c float32
	}
	var backing []float32
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		{
			name: "adds constant in place",
			init: func(t *testing.T) Vector {
				backing = []float32{1, 2, 3}
				return Vector(backing)
			},
			inspect: func(r Vector, t *testing.T) {
				assert.Equal(t, Vector{3, 4, 5}, r)
				assert.Equal(t, []float32{3, 4, 5}, backing)
			},
			args: func(t *testing.T) args {
				return args{2}
			},
			want1: NewFrom(3, 4, 5),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			var basePtr *float32
			if len(receiver) > 0 {
				basePtr = &receiver[0]
			}
			got1 := receiver.AddC(tArgs.c)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.AddC got1 = %v, want1: %v")
			if basePtr != nil {
				if res, ok := got1.(Vector); ok && len(res) > 0 {
					assert.True(t, basePtr == &res[0], "Vector.AddC should mutate receiver in place")
				}
			}
		})
	}
}

func TestVector_Sub(t *testing.T) {
	type args struct {
		v1 Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Sub(tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Sub got1 = %v, want1: %v")
		})
	}
}

func TestVector_SubC(t *testing.T) {
	type args struct {
		c float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.SubC(tArgs.c)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.SubC got1 = %v, want1: %v")
		})
	}
}

func TestVector_MulC(t *testing.T) {
	type args struct {
		c float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
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

			assert.Equal(t, tt.want1, got1, "Vector.MulC got1 = %v, want1: %v")
		})
	}
}

func TestVector_MulCAdd(t *testing.T) {
	type args struct {
		c  float32
		v1 Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.MulCAdd(tArgs.c, tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.MulCAdd got1 = %v, want1: %v")
		})
	}
}

func TestVector_MulCSub(t *testing.T) {
	type args struct {
		c  float32
		v1 Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.MulCSub(tArgs.c, tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.MulCSub got1 = %v, want1: %v")
		})
	}
}

func TestVector_DivC(t *testing.T) {
	type args struct {
		c float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
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

			assert.Equal(t, tt.want1, got1, "Vector.DivC got1 = %v, want1: %v")
		})
	}
}

func TestVector_DivCAdd(t *testing.T) {
	type args struct {
		c  float32
		v1 Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.DivCAdd(tArgs.c, tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.DivCAdd got1 = %v, want1: %v")
		})
	}
}

func TestVector_DivCSub(t *testing.T) {
	type args struct {
		c  float32
		v1 Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.DivCSub(tArgs.c, tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.DivCSub got1 = %v, want1: %v")
		})
	}
}

func TestVector_Normal(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Normal()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Normal got1 = %v, want1: %v")
		})
	}
}

func TestVector_NormalFast(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.NormalFast()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.NormalFast got1 = %v, want1: %v")
		})
	}
}

func TestVector_Axis(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Axis()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Axis got1 = %v, want1: %v")
		})
	}
}

func TestVector_Theta(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 float32
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Theta()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Theta got1 = %v, want1: %v")
		})
	}
}

func TestVector_Conjugate(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Conjugate()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Conjugate got1 = %v, want1: %v")
		})
	}
}

func TestVector_Roll(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 float32
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Roll()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Roll got1 = %v, want1: %v")
		})
	}
}

func TestVector_Pitch(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 float32
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Pitch()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Pitch got1 = %v, want1: %v")
		})
	}
}

func TestVector_Yaw(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		want1 float32
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Yaw()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Yaw got1 = %v, want1: %v")
		})
	}
}

func TestVector_Product(t *testing.T) {
	type args struct {
		b *Quaternion
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Product(tArgs.b)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Product got1 = %v, want1: %v")
		})
	}
}

func TestVector_Slerp(t *testing.T) {
	type args struct {
		v1   Vector
		time float32
		spin float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Slerp(tArgs.v1, tArgs.time, tArgs.spin)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Slerp got1 = %v, want1: %v")
		})
	}
}

func TestVector_SlerpLong(t *testing.T) {
	type args struct {
		v1   Vector
		time float32
		spin float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.SlerpLong(tArgs.v1, tArgs.time, tArgs.spin)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.SlerpLong got1 = %v, want1: %v")
		})
	}
}

func TestVector_Multiply(t *testing.T) {
	type args struct {
		v1 Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Multiply(tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Multiply got1 = %v, want1: %v")
		})
	}
}

func TestVector_Dot(t *testing.T) {
	type args struct {
		v1 Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 float32
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Dot(tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Dot got1 = %v, want1: %v")
		})
	}
}

func TestVector_Cross(t *testing.T) {
	type args struct {
		v1 Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Cross(tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Cross got1 = %v, want1: %v")
		})
	}
}

func TestVector_Refract2D(t *testing.T) {
	type args struct {
		n  Vector
		ni float32
		nt float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
		want2 bool
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1, got2 := receiver.Refract2D(tArgs.n, tArgs.ni, tArgs.nt)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Refract2D got1 = %v, want1: %v")

			assert.Equal(t, tt.want2, got2, "Vector.Refract2D got2 = %v, want2: %v")
		})
	}
}

func TestVector_Refract3D(t *testing.T) {
	type args struct {
		n  Vector
		ni float32
		nt float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
		want2 bool
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1, got2 := receiver.Refract3D(tArgs.n, tArgs.ni, tArgs.nt)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Refract3D got1 = %v, want1: %v")

			assert.Equal(t, tt.want2, got2, "Vector.Refract3D got2 = %v, want2: %v")
		})
	}
}

func TestVector_Reflect(t *testing.T) {
	type args struct {
		n Vector
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Reflect(tArgs.n)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Reflect got1 = %v, want1: %v")
		})
	}
}

func TestVector_Interpolate(t *testing.T) {
	type args struct {
		v1 Vector
		t  float32
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) Vector
		inspect func(r Vector, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Interpolate(tArgs.v1, tArgs.t)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			assert.Equal(t, tt.want1, got1, "Vector.Interpolate got1 = %v, want1: %v")
		})
	}
}
