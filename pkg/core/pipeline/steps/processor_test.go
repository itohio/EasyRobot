package steps

import (
	"context"
	"reflect"
	"testing"

	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

func TestDefaultProcessor_Init(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) DefaultProcessor
		inspect func(r DefaultProcessor, t *testing.T) //inspects receiver after test run

		wantErr    bool
		inspectErr func(err error, t *testing.T) //use for more precise error evaluation after test
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			err := receiver.Init()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("DefaultProcessor.Init error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestDefaultProcessor_Reset(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) DefaultProcessor
		inspect func(r DefaultProcessor, t *testing.T) //inspects receiver after test run

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

func TestDefaultProcessor_Close(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) DefaultProcessor
		inspect func(r DefaultProcessor, t *testing.T) //inspects receiver after test run

	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			receiver.Close()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

		})
	}
}

func TestDefaultProcessor_Process(t *testing.T) {
	type args struct {
		src pipeline.Data
		dst pipeline.Data
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) DefaultProcessor
		inspect func(r DefaultProcessor, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		wantErr    bool
		inspectErr func(err error, t *testing.T) //use for more precise error evaluation after test
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			err := receiver.Process(tArgs.src, tArgs.dst)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("DefaultProcessor.Process error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestWithFields(t *testing.T) {
	type args struct {
		fields store.Store
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 plugin.Option
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := WithFields(tArgs.fields)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithFields got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestWithProcessor(t *testing.T) {
	type args struct {
		pr Processor
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 plugin.Option
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := WithProcessor(tArgs.pr)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithProcessor got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestWithProcessorFunc(t *testing.T) {
	type args struct {
		f ProcessFunc
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 plugin.Option
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := WithProcessorFunc(tArgs.f)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithProcessorFunc got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestWithNamedProcessorFunc(t *testing.T) {
	type args struct {
		name string
		f    ProcessFunc
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 plugin.Option
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := WithNamedProcessorFunc(tArgs.name, tArgs.f)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithNamedProcessorFunc got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestWithInitFunc(t *testing.T) {
	type args struct {
		f func() error
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 plugin.Option
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := WithInitFunc(tArgs.f)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithInitFunc got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestWithResetFunc(t *testing.T) {
	type args struct {
		f func()
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 plugin.Option
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := WithResetFunc(tArgs.f)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithResetFunc got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestNewProcessor(t *testing.T) {
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

			got1, err := NewProcessor(tArgs.opts...)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("NewProcessor got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("NewProcessor error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func Testprocessor_In(t *testing.T) {
	type args struct {
		ch <-chan pipeline.Data
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *processor
		inspect func(r *processor, t *testing.T) //inspects receiver after test run

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

func Testprocessor_Out(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *processor
		inspect func(r *processor, t *testing.T) //inspects receiver after test run

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
				t.Errorf("processor.Out got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func Testprocessor_Run(t *testing.T) {
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *processor
		inspect func(r *processor, t *testing.T) //inspects receiver after test run

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

func Testprocessor_Reset(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *processor
		inspect func(r *processor, t *testing.T) //inspects receiver after test run

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
