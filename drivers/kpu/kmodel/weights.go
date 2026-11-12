package kmodel

import (
	"encoding/binary"
	"fmt"
)

// EncodeWeights writes conv weights as the KPU's weight blob (review
// R-K210-K8b B2). w holds the quantised weights, as unsigned values, in
// TFLite/qnet order: Conv2D [OC][KH][KW][IC], depthwise [KH][KW][C] (with
// C = IC = OC). The blob is OIHW, (oc, ic, ky, kx), or for depthwise
// (c, ky, kx): one byte per weight in 8-bit mode (values ≤ 255), otherwise
// little-endian uint16. The KPU's loads (SetLoad) read the blob in order;
// there is no padding between loads.
//
// Reproduces ncc's weight blobs byte for byte: the weights of all 61 convs
// of the K8 evidence's five kmodels, quantised from the float TFLite models
// as ncc rc5 does (ER-K210-K9 evidence, K9b).
func EncodeWeights(w []uint16, k, ic, oc int, depthwise, eightBit bool) ([]byte, error) {
	if k != 1 && k != 3 {
		return nil, rangeErr("kernel size", k)
	}
	if ic < 1 || ic > maxChannels || oc < 1 || oc > maxChannels || (depthwise && ic != oc) {
		return nil, rangeErr("channels (in, out)", [2]int{ic, oc})
	}
	n := k * k * ic
	if !depthwise {
		n *= oc
	}
	if len(w) != n {
		return nil, fmt.Errorf("%w: %d weights, want %d", ErrFormat, len(w), n)
	}
	size := 2
	if eightBit {
		size = 1
	}
	out := make([]byte, n*size)
	put := func(i int, v uint16) error {
		if eightBit {
			if v > 255 {
				return rangeErr("8-bit weight", v)
			}
			out[i] = byte(v)
			return nil
		}
		binary.LittleEndian.PutUint16(out[2*i:], v)
		return nil
	}
	i := 0
	if depthwise {
		for c := range oc {
			for ky := range k {
				for kx := range k {
					if err := put(i, w[(ky*k+kx)*oc+c]); err != nil {
						return nil, err
					}
					i++
				}
			}
		}
		return out, nil
	}
	for o := range oc {
		for c := range ic {
			for ky := range k {
				for kx := range k {
					if err := put(i, w[((o*k+ky)*k+kx)*ic+c]); err != nil {
						return nil, err
					}
					i++
				}
			}
		}
	}
	return out, nil
}
