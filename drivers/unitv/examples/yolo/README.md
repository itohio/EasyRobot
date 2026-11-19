# yolo — YOLOX-nano on the UnitV's CPU

Runs YOLOX-nano (int8, 416×416, the 80 COCO classes) on the K210's RISC-V
cores with qnet's int8 kernels (`x/math/nn/qnet`, qi8 `generic` backend) on
one built-in test image, decodes the heads (`x/vision/detect/yolox`, score ≥
0.3, NMS IoU 0.45) and prints on UARTHS (USB serial, 115200 baud), every run:

```
yolo: backend generic layers 1204 segments 1204 MACs 522553408 weights 896512 model_fnv df940dfa input_fnv 6da6b3bb arena 1827135 setup_ms …
yolo: run 1 frame_ms … decode_us … fnv_scores a8beb4f2 fnv_boxes caca354e dets 2 heap_inuse … heap_sys … total_alloc … mallocs …
yolo: det 3478 laptop 0.5680 72.34 17.21 399.54 373.84
yolo: det 3464 tv 0.4970 76.10 28.83 385.05 375.37
yolo: kind pointwise layers 339 MACs 429103168 ms …
…
```

`fnv_*` are FNV-1a hashes of the two output tensors, a `det` line is the anchor, label, score and box
corners in input pixels, and `kind` lines split the network time by layer kind (`pointwise` is a 1×1
Conv2D). The setup allocates everything once; the run loop allocates nothing, so `total_alloc` and
`mallocs` stay constant from run to run.

The same source builds for the host (`host.go`), which prints the same lines from files. Its hashes
and detections are the reference: the board must print the same ones, with only times and heap
figures differing.

## Model and test image (not in the repository)

- **Model:** `perchannel.tflite` from `huggingface.co/rts-ieper/yolox-nano-coco` (RTS's TFLite export
  of Megvii's YOLOX-nano COCO weights; Apache-2.0),
  SHA-256 `ab25924e106bfc5d137c8fb98404a57269381ec8db9eb8a79aecb99f8aae36af`.
- **Test image:** any 416×416 RGB image as raw bytes (416·416·3 = 519 168 bytes, row-major, RGB,
  uint8). For example: `ffmpeg -i photo.jpg -vf scale=416:416 -f rawvideo -pix_fmt rgb24 input.u8`.

Copy both into this directory as `model.tflite` and `input.u8`. Both are gitignored, and so is
`data_unitv.go`, which the build generates from them.

## Build and run

From the repository root (the target's linker-script path is relative to it), with TinyGo 0.43:

```sh
# host reference (reads the two files)
go run ./drivers/unitv/examples/yolo

# board: write data_unitv.go (model and int8 input as string constants),
# then build build/unitv-yolo.bin, or build and flash (asks first)
go run ./drivers/unitv/examples/yolo/gen.go
make unitv UNITV_PROG=yolo
make flash-unitv UNITV_PROG=yolo UNITV_PORT=/dev/cu.usbserial-…
```

The image is 2.25 MB: code plus the model (1.58 MB) and the input (0.52 MB). It loads into the 6 MiB
general SRAM and leaves 4.0 MB of heap. That is enough for qnet's graph and prepared layers, but not
also for the 1.83 MB activation arena in one piece: the loader's garbage is spread over the heap,
TinyGo's collector never moves objects, and the arena runs out of memory. So the board build takes the
arena from the KPU's 2 MiB memory, lent to the CPU with `kpu.LendMemory` (the KPU is idle in this
example). The host build allocates it. A KPU version of this example (ER-070) can't do the same: the
KPU needs that memory itself, so it needs its own memory plan, for example a smaller general-SRAM
arena for the CPU layers or a lighter loader.

## TinyGo 0.43 on the K210

- `//go:embed` does not link: ld.lld reports "cannot link object files with different floating-point
  ABI" for the embed objects. `gen.go` writes the files as string constants instead, which end up in the program's
  own rodata, one read-only copy used in place.
- libm's `exp` does not link: TinyGo's picolibc for the K210 uses the medlow code model, and its
  absolute references to `__exp_data` cannot reach RAM at 0x80000000 ("relocation R_RISCV_HI20 out
  of range"). qi8's logistic table and the YOLOX decoder call `math.Exp` during setup, so
  `exp_unitv.go` defines `exp` (Go's `math.exp` algorithm). Rounded to float32, as both callers use
  it, it equals the host's `math.Exp` for every float32 input in [−40, 40].
