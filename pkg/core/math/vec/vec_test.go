package vec

import (
	"reflect"
	"testing"
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("New got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("NewFrom got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Sum got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Slice got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.XY got1 = %v, want1: %v", got1, tt.want1)
			}

			if !reflect.DeepEqual(got2, tt.want2) {
				t.Errorf("Vector.XY got2 = %v, want2: %v", got2, tt.want2)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.XYZ got1 = %v, want1: %v", got1, tt.want1)
			}

			if !reflect.DeepEqual(got2, tt.want2) {
				t.Errorf("Vector.XYZ got2 = %v, want2: %v", got2, tt.want2)
			}

			if !reflect.DeepEqual(got3, tt.want3) {
				t.Errorf("Vector.XYZ got3 = %v, want3: %v", got3, tt.want3)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.XYZW got1 = %v, want1: %v", got1, tt.want1)
			}

			if !reflect.DeepEqual(got2, tt.want2) {
				t.Errorf("Vector.XYZW got2 = %v, want2: %v", got2, tt.want2)
			}

			if !reflect.DeepEqual(got3, tt.want3) {
				t.Errorf("Vector.XYZW got3 = %v, want3: %v", got3, tt.want3)
			}

			if !reflect.DeepEqual(got4, tt.want4) {
				t.Errorf("Vector.XYZW got4 = %v, want4: %v", got4, tt.want4)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.SumSqr got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Magnitude got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.DistanceSqr got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Distance got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Clone got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.CopyFrom got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.CopyTo got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Clamp got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.FillC got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Neg got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestVector_Add(t *testing.T) {
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
			got1 := receiver.Add(tArgs.v1)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Add got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestVector_AddC(t *testing.T) {
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
			got1 := receiver.AddC(tArgs.c)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.AddC got1 = %v, want1: %v", got1, tt.want1)
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Sub got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.SubC got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.MulC got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.MulCAdd got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.MulCSub got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.DivC got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.DivCAdd got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.DivCSub got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Normal got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.NormalFast got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Axis got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Theta got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Conjugate got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Roll got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Pitch got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Yaw got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestVector_Product(t *testing.T) {
	type args struct {
		b Quaternion
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Product got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Slerp got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.SlerpLong got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Multiply got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Dot got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Cross got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Refract2D got1 = %v, want1: %v", got1, tt.want1)
			}

			if !reflect.DeepEqual(got2, tt.want2) {
				t.Errorf("Vector.Refract2D got2 = %v, want2: %v", got2, tt.want2)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Refract3D got1 = %v, want1: %v", got1, tt.want1)
			}

			if !reflect.DeepEqual(got2, tt.want2) {
				t.Errorf("Vector.Refract3D got2 = %v, want2: %v", got2, tt.want2)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Reflect got1 = %v, want1: %v", got1, tt.want1)
			}
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

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector.Interpolate got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}
