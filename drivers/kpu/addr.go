package kpu

import (
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
)

// addrOf is the address of b's first byte.
func addrOf(b []byte) uintptr { return uintptr(unsafe.Pointer(unsafe.SliceData(b))) }

// New prepares model m to run on the KPU with main memory main (at least
// m.MainMemUsage bytes). m.Data must be 128-byte aligned (see AlignedCopy)
// and must stay alive and unchanged while the Runner is used: the KPU reads
// the weights from it directly. It returns ErrBusy once LendMemory has lent
// the KPU memory to the CPU.
func New(m *kmodel.Model, main []byte) (*Runner, error) {
	var r *Runner
	err := owner.use(func() error {
		e, err := open(m.EightBitMode())
		if err != nil {
			return err
		}
		r, err = newRunner(m, main, addrOf(m.Data), e)
		return err
	})
	return r, err
}
