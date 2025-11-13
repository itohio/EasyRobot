package vec

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestVector2D_Vector(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Vector2D
		inspect func(r *Vector2D, t *testing.T) //inspects receiver after test run
		want1   Vector
	}{
		{
			name: "modify",
			init: func(t *testing.T) *Vector2D { return &Vector2D{1, 2} },
			inspect: func(r *Vector2D, t *testing.T) {
				r[0] = 123
			},
			want1: NewFrom(123, 2),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}
			got1 := receiver.View()

			assert.Equal(t, tt.want1, got1, "Vector2D.Vector got1 = %v, want1: %v", got1, tt.want1)
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
		args    func(t *testing.T) args
		want1   Vector
	}{
		{
			name: "modify",
			init: func(t *testing.T) *Vector2D { return &Vector2D{1, 2} },
			inspect: func(r *Vector2D, t *testing.T) {
				r[0] = 123
			},
			args:  func(t *testing.T) args { return args{1, -1} },
			want1: NewFrom(2),
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

			assert.Equal(t, tt.want1, got1, "Vector2D.Slice got1 = %v, want1: %v", got1, tt.want1)
		})
	}
}

func TestVector2D_Clone(t *testing.T) {
	v := Vector2D{1, 2}

	cloned := v.Clone()
	v[0] = 123

	cloned2D, ok := cloned.(Vector2D)
	if assert.True(t, ok, "Clone should return a Vector2D copy") {
		assert.Equal(t, Vector2D{1, 2}, cloned2D)
	}
}

func TestVector2D_Neg(t *testing.T) {
	v := Vector2D{1, 2}

	neg := v.Neg()

	neg2D, ok := neg.(Vector2D)
	if assert.True(t, ok, "Neg should return a Vector2D") {
		assert.Equal(t, Vector2D{-1, -2}, neg2D)
	}
	assert.Equal(t, Vector2D{1, 2}, v)
}

func TestVector2D_AddDoesNotMutateReceiver(t *testing.T) {
	receiver := Vector2D{1, 2}
	operand := Vector2D{3, 4}

	got := receiver.Add(operand)

	result, ok := got.(Vector2D)
	if assert.True(t, ok, "Add should return a Vector2D") {
		assert.Equal(t, Vector2D{4, 6}, result)
	}
	assert.Equal(t, Vector2D{1, 2}, receiver)
}

func TestVector2D_AddDoesNotMutatePointerReceiver(t *testing.T) {
	receiver := &Vector2D{1, 2}
	operand := Vector2D{3, 4}
	before := *receiver

	got := receiver.Add(operand)

	result, ok := got.(Vector2D)
	if assert.True(t, ok, "Add should return a Vector2D") {
		assert.Equal(t, Vector2D{4, 6}, result)
	}
	assert.Equal(t, before, *receiver)
}
