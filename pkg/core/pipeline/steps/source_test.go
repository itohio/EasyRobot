package steps

import (
	"context"
	"reflect"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/options"
	"github.com/itohio/EasyRobot/pkg/core/pipeline"
	"github.com/itohio/EasyRobot/pkg/core/store"
)

func TestWithSourceReader(t *testing.T) {
	type args struct {
		reader SourceReader
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

			got1 := WithSourceReader(tArgs.reader)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithSourceReader got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestWithRepeat(t *testing.T) {
	tests := []struct {
		name string

		want1 options.Option
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got1 := WithRepeat()

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithRepeat got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestWithKey(t *testing.T) {
	type args struct {
		dst store.FQDNType
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

			got1 := WithKey(tArgs.dst)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("WithKey got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestNewReader(t *testing.T) {
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

			got1, err := NewReader(tArgs.opts...)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("NewReader got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("NewReader error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestreaderImpl_In(t *testing.T) {
	type args struct {
		ch <-chan pipeline.Data
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *readerImpl
		inspect func(r *readerImpl, t *testing.T) //inspects receiver after test run

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

func TestreaderImpl_Out(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *readerImpl
		inspect func(r *readerImpl, t *testing.T) //inspects receiver after test run

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
				t.Errorf("readerImpl.Out got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestreaderImpl_Run(t *testing.T) {
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *readerImpl
		inspect func(r *readerImpl, t *testing.T) //inspects receiver after test run

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

func TestreaderImpl_Reset(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *readerImpl
		inspect func(r *readerImpl, t *testing.T) //inspects receiver after test run

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
