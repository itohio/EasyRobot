package pipeline

import (
	"reflect"
	"testing"

	"github.com/foxis/EasyRobot/pkg/core/plugin"
)

func TestRegister(t *testing.T) {
	type args struct {
		name    string
		builder StepBuilder
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

			err := Register(tArgs.name, tArgs.builder)

			if (err != nil) != tt.wantErr {
				t.Fatalf("Register error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestUnregister(t *testing.T) {
	type args struct {
		name string
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

			err := Unregister(tArgs.name)

			if (err != nil) != tt.wantErr {
				t.Fatalf("Unregister error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestNewStep(t *testing.T) {
	type args struct {
		name string
		opts []plugin.Option
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1      Step
		wantErr    bool
		inspectErr func(err error, t *testing.T) //use for more precise error evaluation after test
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1, err := NewStep(tArgs.name, tArgs.opts...)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("NewStep got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("NewStep error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}
