//go:build unix

package main

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// TestWriteFileUmask (review recheck of ec2caa2): the output is created
// 0644 under the process umask, as os.WriteFile would, not forced to 0644.
func TestWriteFileUmask(t *testing.T) {
	old := syscall.Umask(0o077)
	defer syscall.Umask(old)
	out := filepath.Join(t.TempDir(), "out.kmodel")
	if err := writeFile(out, []byte("kmodel")); err != nil {
		t.Fatal(err)
	}
	if fi, err := os.Stat(out); err != nil || fi.Mode().Perm() != 0o600 {
		t.Errorf("mode %v (%v), want 0600 under umask 077", fi.Mode().Perm(), err)
	}
}
