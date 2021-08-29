package steps

import (
	"context"
	"reflect"
	"testing"

	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

type tmpsink struct{}

func (t tmpsink) Init() error              { return nil }
func (t tmpsink) Close()                   {}
func (t tmpsink) Reset()                   {}
func (t tmpsink) Sink(s store.Store) error { return nil }

func TestOptions(t *testing.T) {
	tmpSink := tmpsink{}
	tmpSinkFunc := func(s store.Store) error { return nil }
	tmpFunc := func() {}
	tmpErrFunc := func() error { return nil }

	tests := []struct {
		name    string
		args    []options.Option
		want1   SinkOptions
		wantErr bool
	}{
		{"WithSinkProcessor", []options.Option{plugin.WithName("name"), WithSinkProcessor(&tmpSink)}, SinkOptions{base: plugin.Options{"name", false, true, 0, false, false, true}, sink: &tmpSink}, false},
		{"WithSinkFunc", []options.Option{WithSinkFunc(tmpSinkFunc)}, SinkOptions{base: plugin.Options{"sink", false, true, 0, false, false, true}, sink: &DefaultSink{sink: tmpSinkFunc}}, false},
		{"WithNamedSinkFunc", []options.Option{WithNamedSinkFunc("name", tmpSinkFunc)}, SinkOptions{base: plugin.Options{"name", false, true, 0, false, false, true}, sink: &DefaultSink{sink: tmpSinkFunc}}, false},
		{"WithSinkInitFunc", []options.Option{WithSinkInitFunc(tmpErrFunc)}, SinkOptions{base: plugin.Options{"sink", false, true, 0, false, false, true}, sink: &DefaultSink{init: tmpErrFunc}}, false},
		{"WithSinkResetFunc", []options.Option{WithSinkResetFunc(tmpFunc)}, SinkOptions{base: plugin.Options{"sink", false, true, 0, false, false, true}, sink: &DefaultSink{reset: tmpFunc}}, false},
		{"WithSinkCloseFunc", []options.Option{WithSinkCloseFunc(tmpFunc)}, SinkOptions{base: plugin.Options{"sink", false, true, 0, false, false, true}, sink: &DefaultSink{close: tmpFunc}}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got1, err := NewSink(tt.args...)

			if (err != nil) != tt.wantErr {
				t.Fatalf("NewSink %s error = %v, wantErr: %t", tt.name, err, tt.wantErr)
			}

			sink, ok := got1.(*sink)
			if !ok {
				t.Fatalf("NewSink %s did not cast to sink", tt.name)
			}

			if !reflect.DeepEqual(sink.base, tt.want1.base) {
				t.Errorf("NewSink %s got1 = %v, want1: %v", tt.name, sink.SinkOptions, tt.want1)
			}

			if dpwant, ok := tt.want1.sink.(*DefaultSink); ok {
				if dp, ok := sink.sink.(*DefaultSink); ok {
					if !reflect.DeepEqual(*dp, *dpwant) {
						t.Errorf("NewSink %s sink f got1 = %v, want1: %v", tt.name, *dp, *dpwant)
					}
				} else {
					t.Errorf("NewSink %s sink f got1 = %v, want1: %v", tt.name, sink.sink, dpwant)
				}
			} else {
				if !reflect.DeepEqual(sink.sink, tt.want1.sink) {
					t.Errorf("NewSink %s sink got1 = %v, want1: %v", tt.name, sink.sink, tt.want1.sink)
				}
			}
		})
	}
}

func TestDefaultSink_Init(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) DefaultSink
		inspect func(r DefaultSink, t *testing.T) //inspects receiver after test run

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
				t.Fatalf("DefaultSink.Init error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestDefaultSink_Reset(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) DefaultSink
		inspect func(r DefaultSink, t *testing.T) //inspects receiver after test run

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

func TestDefaultSink_Close(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) DefaultSink
		inspect func(r DefaultSink, t *testing.T) //inspects receiver after test run

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

func TestDefaultSink_Sink(t *testing.T) {
	type args struct {
		data pipeline.Data
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) DefaultSink
		inspect func(r DefaultSink, t *testing.T) //inspects receiver after test run

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
			err := receiver.Sink(tArgs.data)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("DefaultSink.Sink error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestWithSinkFunc(t *testing.T) {
	type args struct {
		f SinkFunc
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 options.Option
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := WithSinkFunc(tArgs.f)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithSinkFunc got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestWithNamedSinkFunc(t *testing.T) {
	type args struct {
		name string
		f    SinkFunc
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 options.Option
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := WithNamedSinkFunc(tArgs.name, tArgs.f)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithNamedSinkFunc got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestWithSinkInitFunc(t *testing.T) {
	type args struct {
		f func() error
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 options.Option
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := WithSinkInitFunc(tArgs.f)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithSinkInitFunc got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestWithSinkResetFunc(t *testing.T) {
	type args struct {
		f func()
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 options.Option
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := WithSinkResetFunc(tArgs.f)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithSinkResetFunc got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestNewSink(t *testing.T) {
	type args struct {
		opts []options.Option
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

			got1, err := NewSink(tArgs.opts...)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("NewSink got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("NewSink error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func Testsink_In(t *testing.T) {
	type args struct {
		ch <-chan pipeline.Data
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *sink
		inspect func(r *sink, t *testing.T) //inspects receiver after test run

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

func Testsink_Out(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *sink
		inspect func(r *sink, t *testing.T) //inspects receiver after test run

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
				t.Errorf("sink.Out got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func Testsink_Run(t *testing.T) {
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *sink
		inspect func(r *sink, t *testing.T) //inspects receiver after test run

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

func Testsink_Reset(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *sink
		inspect func(r *sink, t *testing.T) //inspects receiver after test run

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
