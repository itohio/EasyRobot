# yolo — YOLOX-nano on the UnitV's CPU

Runs YOLOX-nano (int8, 416×416, the 80 COCO classes) on the K210's RISC-V
cores with qnet's int8 kernels (`x/math/nn/qnet`, qi8 `generic` backend),
decodes the heads (`x/vision/detect/yolox`, score ≥ 0.3, NMS IoU 0.45) and
writes the results on UARTHS (USB serial, 115200 baud). The build-time option
`source` picks the input.

## Camera (default): image + detections to a web page

Each frame:
1. The OV2640 captures QVGA (320×240) RGB565. Colour, because the detector needs it and the page shows the image the
   detector saw.
2. The frame is letterboxed to 416×416 the way YOLOX preprocesses: scaled ×1.3 with bilinear interpolation, at the top
   left, the rest grey 114.
3. The detector runs on it.
4. The board writes the frame as a frame v1 image (`drivers/camera/frame`), then a detection record for it (boxes in
   the camera image's pixels, scores, the stage times), then log lines:

```
yolo: frame 3 dets 0 capture_ms 57.5 letterbox_ms 162.4 detect_ms 32674.5 send_prev_ms 13429.8
yolo: det 3 laptop 0.3662 54 7 298 213        (one per detection)
yolo: check input_fnv … out0_fnv … out1_fnv … total_alloc … mallocs … heap_inuse …
```

The `check` line hashes the model's input and its two outputs (YOLOX: scores, boxes), so a host run on the same image
can be compared bit for bit. The loop allocates nothing: `total_alloc` and `mallocs` stay constant.

Steps 2–3 sit behind the `Detector` interface (`objects.go`): a frame goes in, detections in frame pixels come out,
and class names come from the detector's table. Another model, for example an SSD on the KPU, replaces
`yoloxObjects` without touching the camera loop, the stream format or the page.

Timing on the UnitV: capture ≈ 60 ms, letterbox 162 ms, detection 32.7 s, sending 13.4 s. The image is 154,528
bytes on the wire at 115200 baud, which is loss-free without chunk resend. That is one frame every 46.3 s.

**Viewer:** open `web/index.html` in Chrome or Edge (from disk; no server, no network).
- "Connect board" picks the UnitV's serial port through the browser's serial API.
- "Open replay" reads a capture file, for example `web/replay.bin`.

The page checks every CRC and draws each image with its boxes, class names and confidences. It lists the frames
(click one to show it again), shows the stage times and the link figures, and puts the board's text in a log pane. A
frame with a bad chunk is listed as dropped and never shown with another frame's boxes.

**Replay file:** `web/replay.bin` is a synthetic 160×120 scene with made-up detections, three frames, the second with a
corrupted chunk. `go run ./drivers/unitv/examples/yolo/replay` regenerates it, and its test checks that the committed
file is what the generator writes. A capture of the board (e.g. `cmd/camframe -raw`) replays the same way.

## Still image (`-ldflags "-X main.source=image"`): the board reference

The board runs on one built-in test image and prints, every run:

```
yolo: backend generic layers 1204 segments 1204 MACs 522553408 weights 896512 model_fnv df940dfa input_fnv 6da6b3bb arena 1827135 setup_ms …
yolo: run 1 frame_ms … decode_us … fnv_scores a8beb4f2 fnv_boxes caca354e dets 2 heap_inuse … heap_sys … total_alloc … mallocs …
yolo: det 3478 laptop 0.5680 72.34 17.21 399.54 373.84
yolo: det 3464 tv 0.4970 76.10 28.83 385.05 375.37
yolo: kind pointwise layers 339 MACs 429103168 ms …
…
```

- `fnv_*` are FNV-1a hashes of the two output tensors.
- A `det` line is the anchor, label, score and box corners in input pixels.
- `kind` lines split the network time by layer kind (`pointwise` is a 1×1 Conv2D).

The same source builds for the host (`host.go`), which prints the same lines from files. Its hashes and detections
are the reference: the board must print the same ones, with only times and heap figures differing.

## Model and test image (not in the repository)

- **Model:** `perchannel.tflite` from `huggingface.co/rts-ieper/yolox-nano-coco` (RTS's TFLite export
  of Megvii's YOLOX-nano COCO weights; Apache-2.0),
  SHA-256 `ab25924e106bfc5d137c8fb98404a57269381ec8db9eb8a79aecb99f8aae36af`.
- **Test image** (still-image mode and the host reference): any 416×416 RGB image as raw bytes (416·416·3 = 519 168
  bytes, row-major, RGB, uint8). For example: `ffmpeg -i photo.jpg -vf scale=416:416 -f rawvideo -pix_fmt rgb24 input.u8`.

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
# still-image mode instead of the camera
tinygo build -target=drivers/unitv/target/unitv.json -ldflags "-X main.source=image" -o build/unitv-yolo.bin ./drivers/unitv/examples/yolo
```

The still-image build is 2.25 MB: code plus the model (1.58 MB) and the input (0.52 MB). The camera build is 1.75 MB
(no test image); it allocates the camera frame (150 KB) and the detector input (507 KB) first, before the loader's
garbage. Either loads into the 6 MiB general SRAM and leaves about 4 MB of heap. That is enough for qnet's graph and
prepared layers, but not also for the 1.83 MB activation arena in one piece: the loader's garbage is spread over the
heap, TinyGo's collector never moves objects, and the arena runs out of memory. So the board build takes the
arena from the KPU's 2 MiB memory, lent to the CPU with `kpu.LendMemory` (the KPU is idle in this
example). The host build allocates it. A KPU version of this example (ER-070) can't do the same: the
KPU needs that memory itself, so it needs its own memory plan, for example a smaller general-SRAM
arena for the CPU layers or a lighter loader.

**Boot memory.** The camera build's setup fits the 4.5 MB heap only if the collector frees the loader's garbage. Two
things kept garbage alive, so a few resets after a long run, setup ended in "out of memory" before the setup line:

- **The conservative collector.** It treats any word in the loader's objects that looks like a pointer as one. The UnitV
  target now builds with TinyGo's precise collector (`"gc": "precise"` in `drivers/unitv/target/unitv.json`). In use
  after setup: 2.4 MB instead of 4.4 MB.
- **Stale words on the stack.** TinyGo runs `main` on the system stack (scheduler "none") and scans it
  conservatively, and RAM keeps the previous session's stack across a reset. `clearStack` (`stack_unitv.go`) zeroes the
  unused stack before anything is allocated. With the precise collector alone, the live set after loading still grew
  from boot to boot until a reset ran out of memory; with both fixes it stays at 2.37 MB on every boot.

The setup line prints `heap_sys`, `free_at_start` and `live_after_setup`.

## TinyGo 0.43 on the K210

- `//go:embed` does not link: ld.lld reports "cannot link object files with different floating-point
  ABI" for the embed objects. `gen.go` writes the files as string constants instead, which end up in the program's
  own rodata, one read-only copy used in place.
- libm's `exp` does not link: TinyGo's picolibc for the K210 uses the medlow code model, and its
  absolute references to `__exp_data` cannot reach RAM at 0x80000000 ("relocation R_RISCV_HI20 out
  of range"). qi8's logistic table and the YOLOX decoder call `math.Exp` during setup, so
  `exp_unitv.go` defines `exp` (Go's `math.exp` algorithm). Rounded to float32, as both callers use
  it, it equals the host's `math.Exp` for every float32 input in [−40, 40].
