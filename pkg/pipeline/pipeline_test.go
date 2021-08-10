package pipeline

import (
	"context"
	"reflect"
	"testing"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name string

		want1 Pipeline
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got1 := New()

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("New got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestFromJSON(t *testing.T) {
	type args struct {
		json string
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 Pipeline
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := FromJSON(tArgs.json)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("FromJSON got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestPipeline_AddStep(t *testing.T) {
	type args struct {
		step Step
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Pipeline
		inspect func(r *Pipeline, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1      int
		wantErr    bool
		inspectErr func(err error, t *testing.T) //use for more precise error evaluation after test
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1, err := receiver.AddStep(tArgs.step)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Pipeline.AddStep got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("Pipeline.AddStep error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestPipeline_AddOrFindStep(t *testing.T) {
	type args struct {
		step Step
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Pipeline
		inspect func(r *Pipeline, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1      int
		wantErr    bool
		inspectErr func(err error, t *testing.T) //use for more precise error evaluation after test
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1, err := receiver.AddOrFindStep(tArgs.step)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Pipeline.AddOrFindStep got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("Pipeline.AddOrFindStep error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestPipeline_GetStep(t *testing.T) {
	type args struct {
		id int
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Pipeline
		inspect func(r *Pipeline, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Step
		want2 bool
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1, got2 := receiver.GetStep(tArgs.id)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Pipeline.GetStep got1 = %v, want1: %v", got1, tt.want1)
			}

			if !reflect.DeepEqual(got2, tt.want2) {
				t.Errorf("Pipeline.GetStep got2 = %v, want2: %v", got2, tt.want2)
			}
		})
	}
}

func TestPipeline_FindStep(t *testing.T) {
	type args struct {
		step Step
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Pipeline
		inspect func(r *Pipeline, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1      int
		wantErr    bool
		inspectErr func(err error, t *testing.T) //use for more precise error evaluation after test
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1, err := receiver.FindStep(tArgs.step)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Pipeline.FindStep got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("Pipeline.FindStep error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestPipeline_ConnectStepsById(t *testing.T) {
	type args struct {
		ids []int
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Pipeline
		inspect func(r *Pipeline, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1      <-chan Data
		wantErr    bool
		inspectErr func(err error, t *testing.T) //use for more precise error evaluation after test
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1, err := receiver.ConnectStepsById(tArgs.ids...)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Pipeline.ConnectStepsById got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("Pipeline.ConnectStepsById error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestPipeline_connectChain(t *testing.T) {
	type args struct {
		ids stepChain
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Pipeline
		inspect func(r *Pipeline, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 <-chan Data
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.connectChain(tArgs.ids)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Pipeline.connectChain got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestPipeline_ConnectSteps(t *testing.T) {
	type args struct {
		steps []Step
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Pipeline
		inspect func(r *Pipeline, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1      <-chan Data
		wantErr    bool
		inspectErr func(err error, t *testing.T) //use for more precise error evaluation after test
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1, err := receiver.ConnectSteps(tArgs.steps...)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Pipeline.ConnectSteps got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("Pipeline.ConnectSteps error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func TestPipeline_Run(t *testing.T) {
	type args struct {
		ctx context.Context
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *Pipeline
		inspect func(r *Pipeline, t *testing.T) //inspects receiver after test run

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

func TestPipeline_Reset(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Pipeline
		inspect func(r *Pipeline, t *testing.T) //inspects receiver after test run

		want1      []<-chan Data
		wantErr    bool
		inspectErr func(err error, t *testing.T) //use for more precise error evaluation after test
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1, err := receiver.Reset()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Pipeline.Reset got1 = %v, want1: %v", got1, tt.want1)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("Pipeline.Reset error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}
