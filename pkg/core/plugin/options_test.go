package plugin

import (
	"reflect"
	"testing"

	"github.com/foxis/EasyRobot/pkg/core/options"
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
		opt   []options.Option
		want1 Options
	}{
		{"WithBufferSize", []options.Option{WithBufferSize(123)}, Options{BufferSize: 123}},
		{"WithName", []options.Option{WithName("test")}, Options{Name: "test"}},
		{"WithName1", []options.Option{WithName("test1")}, Options{Name: "test1"}},
		{"WithBlocking", []options.Option{WithBlocking(true)}, Options{Blocking: true}},
		{"!WithBlocking", []options.Option{WithBlocking(false)}, Options{}},
		{"WithEnable", []options.Option{WithEnable(true)}, Options{Enabled: true}},
		{"!WithEnable", []options.Option{WithEnable(false)}, Options{}},
		{"WithEOSExit", []options.Option{WithEOSExit(true)}, Options{NoEOS: false}},
		{"!WithEOSExit", []options.Option{WithEOSExit(false)}, Options{NoEOS: true}},
		{"WithIgnoreErrors", []options.Option{WithIgnoreErrors(true)}, Options{IgnoreErrors: true}},
		{"!WithIgnoreErrors", []options.Option{WithIgnoreErrors(false)}, Options{}},
		{"WithClose", []options.Option{WithClose(true)}, Options{Close: true}},
		{"!WithClose", []options.Option{WithClose(false)}, Options{}},
		{"WithMapping", []options.Option{WithMapping(map[string]interface{}{"buf_size": 123, "name": "321"})}, Options{Name: "321", BufferSize: 123}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			opt := Options{}
			options.ApplyOptions(&opt, tt.opt...)

			if !reflect.DeepEqual(opt, tt.want1) {
				t.Errorf("%s got1 = %v, want1: %v", tt.name, opt, tt.want1)
			}
		})
	}
}
