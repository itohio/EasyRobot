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
// the weights from it directly.
func New(m *kmodel.Model, main []byte) (*Runner, error) {
	e, err := open(m.EightBitMode())
	if err != nil {
		return nil, err
	}
	return newRunner(m, main, addrOf(m.Data), e)
}
