package vec

import (
	"reflect"
	"testing"
)

func TestVector2D_Vector(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Vector2D
		inspect func(r *Vector2D, t *testing.T) //inspects receiver after test run

		want1 Vector
	}{
		{
			"modify",
			func(t *testing.T) *Vector2D { return &Vector2D{1, 2} },
			func(r *Vector2D, t *testing.T) { r[0] = 123 },
			NewFrom(123, 2),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Vector()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector2D.Vector got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestVector2D_Slice(t *testing.T) {
	type args struct {
		start int
		end   int
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Vector2D
		inspect func(r *Vector2D, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Vector
	}{
		{
			"modify",
			func(t *testing.T) *Vector2D { return &Vector2D{1, 2} },
			func(r *Vector2D, t *testing.T) { r[0] = 123 },
			func(t *testing.T) args { return args{1, -1} },
			NewFrom(2),
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
				t.Errorf("Vector2D.Slice got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestVector2D_Clone(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Vector2D
		inspect func(r *Vector2D, t *testing.T) //inspects receiver after test run

		want1 *Vector2D
	}{
		{
			"modify",
			func(t *testing.T) *Vector2D { return &Vector2D{1, 2} },
			func(r *Vector2D, t *testing.T) { r[0] = 123 },
			&Vector2D{1, 2},
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
				t.Errorf("Vector2D.Clone got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestVector2D_Neg(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Vector2D
		inspect func(r *Vector2D, t *testing.T) //inspects receiver after test run

		want1 *Vector2D
	}{
		{
			"modify",
			func(t *testing.T) *Vector2D { return &Vector2D{1, 2} },
			func(r *Vector2D, t *testing.T) { r[0] = 123 },
			&Vector2D{123, -2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Neg()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Vector2D.Neg got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}
