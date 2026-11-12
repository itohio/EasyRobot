package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRunRejectsFlags (review R-K210-K9same S5): a negative -cut other
// than -1 (no cut) and -list combined with -cut or -input are refused
// before the model is read, instead of being ignored.
func TestRunRejectsFlags(t *testing.T) {
	model := filepath.Join(t.TempDir(), "m.tflite") // never read
	for _, c := range []struct {
		name  string
		input string
		cut   int
		list  bool
		args  []string
		want  string
	}{
		{"-cut -2", "", -2, false, []string{model, "out.kmodel"}, "-cut"},
		{"-list -cut 3", "", 3, true, []string{model}, "-list"},
		{"-list -input", "64x64", -1, true, []string{model}, "-list"},
	} {
		err := run(c.input, c.cut, c.list, c.args)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want one naming %s", c.name, err, c.want)
		}
	}
}

// TestWriteFileAtomic (review S4): the output appears whole under its name
// (mode 0644) or not at all; a failed write leaves an existing file as it
// was and no temporary file behind.
func TestWriteFileAtomic(t *testing.T) {
	dir := t.TempDir()
	out := filepath.Join(dir, "out.kmodel")
	if err := os.WriteFile(out, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writeFile(out, []byte("new kmodel")); err != nil {
		t.Fatal(err)
	}
	if b, err := os.ReadFile(out); err != nil || string(b) != "new kmodel" {
		t.Fatalf("out = %q, %v", b, err)
	}
	if fi, _ := os.Stat(out); fi.Mode().Perm() != 0o644 {
		t.Errorf("mode %v, want 0644", fi.Mode().Perm())
	}
	// The rename fails: the target is a non-empty directory.
	sub := filepath.Join(dir, "sub")
	if err := os.MkdirAll(filepath.Join(sub, "x"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeFile(sub, []byte("lost")); err == nil {
		t.Error("writing over a directory: no error")
	}
	es, _ := os.ReadDir(dir)
	if len(es) != 2 {
		var names []string
		for _, e := range es {
			names = append(names, e.Name())
		}
		t.Errorf("dir holds %v, want only out.kmodel and sub", names)
	}
}
