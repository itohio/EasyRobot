package pipeline

import (
	"context"
	"reflect"
	"testing"

	"github.com/foxis/EasyRobot/pkg/core/plugin"
)

func TestStepReceive(t *testing.T) {
	type args struct {
		ctx context.Context
		o   plugin.Options
		in  <-chan Data
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1      Data
		wantErr    bool
		inspectErr func(err error, t *testing.T) //use for more precise error evaluation after test
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1, err := StepReceive(tArgs.ctx, tArgs.o, tArgs.in)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("StepReceive got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("StepReceive error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestStepSend(t *testing.T) {
	type args struct {
		ctx  context.Context
		o    plugin.Options
		out  chan Data
		data Data
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		wantErr    bool
		inspectErr func(err error, t *testing.T) //use for more precise error evaluation after test
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			err := StepSend(tArgs.ctx, tArgs.o, tArgs.out, tArgs.data)

			if (err != nil) != tt.wantErr {
				t.Fatalf("StepSend error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestStepMakeChan(t *testing.T) {
	type args struct {
		o plugin.Options
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 chan Data
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := StepMakeChan(tArgs.o)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("StepMakeChan got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}
