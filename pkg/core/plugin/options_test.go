package plugin

import (
	"reflect"
	"testing"
)

func TestDefaultOptions(t *testing.T) {
	want1 := Options{
		Enabled: true,
		Close:   true,
	}
	got1 := DefaultOptions()

	if !reflect.DeepEqual(got1, want1) {
		t.Errorf("DefaultOptions got1 = %v, want1: %v", got1, want1)
	}
}

func TestApplyOptions(t *testing.T) {
	tests := []struct {
		name  string
		opt   []Option
		want1 Options
	}{
		{"WithBufferSize", []Option{WithBufferSize(123)}, Options{BufferSize: 123}},
		{"WithName", []Option{WithName("test")}, Options{Name: "test"}},
		{"WithName1", []Option{WithName("test1")}, Options{Name: "test1"}},
		{"WithBlocking", []Option{WithBlocking(true)}, Options{Blocking: true}},
		{"!WithBlocking", []Option{WithBlocking(false)}, Options{}},
		{"WithEnable", []Option{WithEnable(true)}, Options{Enabled: true}},
		{"!WithEnable", []Option{WithEnable(false)}, Options{}},
		{"WithEOSExit", []Option{WithEOSExit(true)}, Options{NoEOS: false}},
		{"!WithEOSExit", []Option{WithEOSExit(false)}, Options{NoEOS: true}},
		{"WithIgnoreErrors", []Option{WithIgnoreErrors(true)}, Options{IgnoreErrors: true}},
		{"!WithIgnoreErrors", []Option{WithIgnoreErrors(false)}, Options{}},
		{"WithClose", []Option{WithClose(true)}, Options{Close: true}},
		{"!WithClose", []Option{WithClose(false)}, Options{}},
		{"WithMapping", []Option{WithMapping(map[string]interface{}{"buf_size": 123, "name": "321"})}, Options{Name: "321", BufferSize: 123}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			opt := Options{}
			ApplyOptions(&opt, tt.opt...)

			if !reflect.DeepEqual(opt, tt.want1) {
				t.Errorf("%s got1 = %v, want1: %v", tt.name, opt, tt.want1)
			}
		})
	}
}
