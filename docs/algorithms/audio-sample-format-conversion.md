---
type: algorithm
status: implemented
tags: [easyrobot, audio]
workstream: W7
module: "[[audio-io]]"
summary: "none direct"
source: [x/marshaller/audio/tensor_convert.go, x/marshaller/audio/device_convert.go]
tests: []
test_result: none
complete: yes
correct: unknown
evidence: "no direct tests; only reached via device playback/capture (not run, needs hardware)"
---
# Audio sample-format conversion

**What it does:** converts audio samples between INT16/INT32/FP32/UINT8 tensors, and between raw device PCM bytes (`S16`, `S32`, `F32`, `U8` formats) and typed sample slices.

**Where:**
- Tensor dtype: `convertTensorDtype` tensor_convert.go:15-67; `convertFromFloat32` :70, `convertFromInt32` :96, `convertFromInt16` :120, `convertFromUint8` :144. Called only from `playbackSingleDevice` (marshaller.go:453).
- PCM ↔ samples: `pcmToTensor` device_convert.go:14, `tensorToPCM` :95, `sampleSizeFromFormat` :172, `byteRingToSampleRing` :189, `sampleRingToByteRing*` :234-518. Called from device paths and `byteRing.WriteFrames` (sample_ring.go:1633).
- Channel select/pad: `extractChannelsAndPad` tensor_convert.go:176.

**Inputs → outputs:** tensor `[frames,ch]` of one dtype → new tensor of target dtype (allocates). Float→int clamps to [-1,1] and multiplies by 32767 / 2147483647; int→float divides by 32767 / 2147483647; int16↔int32 by shift 16; uint8 offset 128.

## Completeness review

- Complete: yes for 4 dtypes; FP32 → UINT8 and INT32/INT16 → UINT8 return "unsupported" (tensor_convert.go:89-90,113-114,137-138).
- Tested: no unit tests; package tests never reach these functions (WAV path does its own conversion in `fillTensorFromPCM`).
- Correct: unknown. Notes from reading: no dithering on down-conversion (CONVERSION_ANALYSIS.md claim confirmed); int32→int16 is a truncating shift; scale constants differ from the WAV reader (÷32768) — see [[wav-codec]]. CONVERSION_ANALYSIS.md claim of "duplicated scaling across files" is accurate (tensor_convert.go, device_convert.go, marshaller.go, unmarshaller.go each carry their own constants); its proposed `x/types/st/convert.go` refactor was not done.

## TODOs

- [ ] Unit tests for each dtype pair incl. clamp and extreme values — x/marshaller/audio/tensor_convert.go:70-171 #p2
- [ ] Single shared conversion table (one set of constants) — tensor_convert.go, device_convert.go, unmarshaller.go:383-401 #p3
- [ ] Missing →UINT8 targets — tensor_convert.go:89 #p3

## Links

- Module: [[audio-io]]; [[audio-naive-resampling]]; [[audio-sample-ring-buffer]]
