package steps

import (
	"context"
	"reflect"
	"testing"

	"github.com/foxis/EasyRobot/pkg/pipeline"
	"github.com/foxis/EasyRobot/pkg/plugin"
)

func TestNewFanOut(t *testing.T) {
	type args struct {
		opts []plugin.Option
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1      pipeline.Step
		wantErr    bool
		inspectErr func(err error, t *testing.T) //use for more precise error evaluation after test
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1, err := NewFanOut(tArgs.opts...)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("NewFanOut got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("NewFanOut error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func Testfanout_In(t *testing.T) {
	type args struct {
		ch <-chan pipeline.Data
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *fanout
		inspect func(r *fanout, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			receiver.In(tArgs.ch)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

		})
	}
}

func Testfanout_Out(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *fanout
		inspect func(r *fanout, t *testing.T) //inspects receiver after test run

		want1 <-chan pipeline.Data
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Out()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("fanout.Out got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func Testfanout_Run(t *testing.T) {
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *fanout
		inspect func(r *fanout, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			receiver.Run(tArgs.ctx)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

		})
	}
}

func Testfanout_Reset(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *fanout
		inspect func(r *fanout, t *testing.T) //inspects receiver after test run

	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			receiver.Reset()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

		})
	}
}
