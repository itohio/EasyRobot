package steps

import (
	"context"
	"reflect"
	"testing"

	"github.com/foxis/EasyRobot/pkg/pipeline"
	"github.com/foxis/EasyRobot/pkg/plugin"
	"github.com/foxis/EasyRobot/pkg/store"
)

func Testbuffer_Add(t *testing.T) {
	type args struct {
		data store.Store
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) buffer
		inspect func(r buffer, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 buffer
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Add(tArgs.data)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("buffer.Add got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func Testbuffer_Pop(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *buffer
		inspect func(r *buffer, t *testing.T) //inspects receiver after test run

		want1 store.Store
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Pop()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("buffer.Pop got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func Testbuffer_Prune(t *testing.T) {
	type args struct {
		n int
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) buffer
		inspect func(r buffer, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 buffer
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Prune(tArgs.n)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("buffer.Prune got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func Testbuffer_FindClosest(t *testing.T) {
	type args struct {
		timestamp int64
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) buffer
		inspect func(r buffer, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 int
		want2 int64
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1, got2 := receiver.FindClosest(tArgs.timestamp)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("buffer.FindClosest got1 = %v, want1: %v", got1, tt.want1)
			}

			if !reflect.DeepEqual(got2, tt.want2) {
				t.Errorf("buffer.FindClosest got2 = %v, want2: %v", got2, tt.want2)
			}
		})
	}
}

func TestNewSync(t *testing.T) {
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

			got1, err := NewSync(tArgs.opts...)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("NewSync got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("NewSync error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func Testsyncronize_In(t *testing.T) {
	type args struct {
		ch <-chan pipeline.Data
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *syncronize
		inspect func(r *syncronize, t *testing.T) //inspects receiver after test run

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

func Testsyncronize_Out(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *syncronize
		inspect func(r *syncronize, t *testing.T) //inspects receiver after test run

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
				t.Errorf("syncronize.Out got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func Testsyncronize_Run(t *testing.T) {
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *syncronize
		inspect func(r *syncronize, t *testing.T) //inspects receiver after test run

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

func Testsyncronize_Reset(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *syncronize
		inspect func(r *syncronize, t *testing.T) //inspects receiver after test run

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

func Testsyncronize_doSyncronize(t *testing.T) {
	type args struct {
		ctx  context.Context
		idx  int
		data store.Store
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *syncronize
		inspect func(r *syncronize, t *testing.T) //inspects receiver after test run

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
			err := receiver.doSyncronize(tArgs.ctx, tArgs.idx, tArgs.data)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("syncronize.doSyncronize error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}
