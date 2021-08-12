package mat

import (
	"testing"
)

func Test_multo(t *testing.T) {
	type args struct {
		r1  int
		c1  int
		r2  int
		c2  int
		m1  []float32
		m2  []float32
		dst []float32
	}
	tests := []struct {
		name string
		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			multo(tArgs.r1, tArgs.c1, tArgs.r2, tArgs.c2, tArgs.m1, tArgs.m2, tArgs.dst)

		})
	}
}

func Test_muldiagto(t *testing.T) {
	type args struct {
		r1  int
		c1  int
		v   []float32
		m1  []float32
		dst []float32
	}
	tests := []struct {
		name string
		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			muldiagto(tArgs.r1, tArgs.c1, tArgs.v, tArgs.m1, tArgs.dst)

		})
	}
}

func Test_mulvto(t *testing.T) {
	type args struct {
		m1  []float32
		v   []float32
		dst []float32
	}
	tests := []struct {
		name string
		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			mulvto(tArgs.m1, tArgs.v, tArgs.dst)

		})
	}
}

func Test_mulvtto(t *testing.T) {
	type args struct {
		m1  []float32
		v   []float32
		dst []float32
	}
	tests := []struct {
		name string
		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			mulvtto(tArgs.m1, tArgs.v, tArgs.dst)

		})
	}
}
