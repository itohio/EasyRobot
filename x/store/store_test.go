package store

import (
	"reflect"
	"testing"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name string

		want1 Store
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

func TestNewWithName(t *testing.T) {
	type args struct {
		name string
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 Store
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := NewWithName(tArgs.name)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("NewWithName got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func Teststore_Name(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *store
		inspect func(r *store, t *testing.T) //inspects receiver after test run

		want1 string
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			receiver := tt.init(t)
			got1 := receiver.Name()

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("store.Name got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func Teststore_SetName(t *testing.T) {
	type args struct {
		name string
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *store
		inspect func(r *store, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			receiver.SetName(tArgs.name)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

		})
	}
}

func Teststore_Set(t *testing.T) {
	type args struct {
		fqdn FQDNType
		val  Value
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *store
		inspect func(r *store, t *testing.T) //inspects receiver after test run

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
			err := receiver.Set(tArgs.fqdn, tArgs.val)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("store.Set error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func Teststore_Del(t *testing.T) {
	type args struct {
		fqdn FQDNType
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *store
		inspect func(r *store, t *testing.T) //inspects receiver after test run

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
			err := receiver.Del(tArgs.fqdn)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if (err != nil) != tt.wantErr {
				t.Fatalf("store.Del error = %v, wantErr: %t", err, tt.wantErr)
			}

			if tt.inspectErr != nil {
				tt.inspectErr(err, t)
			}
		})
	}
}

func Teststore_Get(t *testing.T) {
	type args struct {
		fqdn FQDNType
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *store
		inspect func(r *store, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Value
		want2 bool
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1, got2 := receiver.Get(tArgs.fqdn)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("store.Get got1 = %v, want1: %v", got1, tt.want1)
			}

			if !reflect.DeepEqual(got2, tt.want2) {
				t.Errorf("store.Get got2 = %v, want2: %v", got2, tt.want2)
			}
		})
	}
}

func Teststore_CopyFrom(t *testing.T) {
	type args struct {
		src Store
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *store
		inspect func(r *store, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			receiver.CopyFrom(tArgs.src)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

		})
	}
}

func Teststore_Clone(t *testing.T) {
	type args struct {
		fqdnPattern FQDNType
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *store
		inspect func(r *store, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args

		want1 Store
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			got1 := receiver.Clone(tArgs.fqdnPattern)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("store.Clone got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func Teststore_Close(t *testing.T) {
	type args struct {
		fqdnPattern FQDNType
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *store
		inspect func(r *store, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			receiver.Close(tArgs.fqdnPattern)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

		})
	}
}

func Teststore_ForEach(t *testing.T) {
	type args struct {
		fqdnPattern FQDNType
		callback    func(name FQDNType, val Value)
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *store
		inspect func(r *store, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			receiver.ForEach(tArgs.fqdnPattern, tArgs.callback)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

		})
	}
}

func Teststore_Dump(t *testing.T) {
	type args struct {
		fqdnPattern FQDNType
	}
	tests := []struct {
		name    string
		init    func(t *testing.T) *store
		inspect func(r *store, t *testing.T) //inspects receiver after test run

		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			receiver := tt.init(t)
			receiver.Dump(tArgs.fqdnPattern)

			if tt.inspect != nil {
				tt.inspect(receiver, t)
			}

		})
	}
}
