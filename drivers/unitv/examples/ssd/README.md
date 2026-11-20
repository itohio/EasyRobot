# ssd — SSD-MobileNetV1 on the UnitV's KPU

Runs TensorFlow's quantised SSD-MobileNetV1 COCO detector (uint8, trained at
300×300) at 256×256 on the K210: the convolutions on the KPU, the small
layers, the logistic and the box decoding (`x/vision/detect/ssd`) on the
CPU. The program runs on a test image linked into it and reports, on UARTHS
(USB serial, 115200 baud), the hash of every segment's output, the time per
stage, the detections and the heap figures. The camera and the web page of
the `yolo` example are not wired to it yet.

## How the model gets on the board

The board cannot load the TFLite file and compile it: file plus kmodels
exceed the K210's 6 MiB. The host compiles, and the board runs one file in
place (`drivers/kpu/kgraph`):

1. `-gen` loads the detection model (`tfl.LoadDetection`), re-derives it for
   256×256 on its trained sampling grid (`compile.ResizeAligned`), compiles
   the KPU segments (`compile.Compile`), takes the trained anchors of the
   cells that remain (`ssd.CropAnchors`) and writes `model.kgraph`.
2. `data_unitv.c`, written by the same step, links `model.kgraph` and the
   test input into the program's read-only data with `.incbin`, the model
   after `.balign 128`: the KPU fetches weights from 128-byte-aligned
   addresses, and the K210 loads the whole image into RAM, so the file is in
   memory once and is never copied.
3. On the board `kgraph.Parse` checks the address and a CRC-32 over the
   file, `kgraph.NewBackend(file, kpu.New)` builds one KPU runner per
   segment over one shared upload buffer and one shared output buffer, and
   `qnet.NewRunner` runs the graph.

## Build and run

The model, the test image and the generated files are not in the repository
(`.gitignore`).

```sh
# model.kgraph, input.i8, data_unitv.c (the image: raw 256×256×3 uint8 RGB)
go run ./drivers/unitv/examples/ssd -gen -model detect.tflite -image image256.u8
# the reference: the same program on the emulated KPU
go run ./drivers/unitv/examples/ssd -runs 3
# the board image, then flash it (drivers/unitv/doc.go)
make unitv UNITV_PROG=ssd
```

`tinygo build … -ldflags "-X main.mode=dump"` also prints the first frame's
tensors as hex (about 1 MB, 100 s at 115200 baud), each as it is produced,
for comparing the board's values with the host's.

## Output

```
ssd: boot model_addr … model_addr_mod128 0 model_bytes 4401508 input_bytes 196608 heap_sys … heap_inuse …
ssd: parse ok ms 87
ssd: setup layers 68 segments 40 kpu 10 anchors 1284 arena 275983 upload 196608 output 98304 input_fnv … setup_ms 100 heap_inuse … total_alloc … mallocs …
ssd: seg 0 layers 0+24 kpu conv2d bytes 98304 us 38695 fnv …      (one per segment; fnv on frame 1)
ssd: frame 2 total_us 147825 convert_us 32579 upload_us 13681 kpu_us 43183 readback_us 7602 cpu_conv_us 37003 tail_us 10176 decode_us 3555 dets 1 heap_inuse … total_alloc … mallocs …
ssd: det anchor 1266 class 71 score 0.7656 0.1625 0.0505 0.9612 0.9287
```

- `seg`: a KPU segment (`kpu`, several layers as one kmodel) or one CPU
  layer, with its output size and time.
- `frame`: `convert` is the int8 NHWC ↔ uint8 CHW conversion around every
  KPU segment, `upload` and `readback` the copies into and out of KPU
  memory, `kpu` the KPU's own time, `cpu_conv` the convolutions left to the
  CPU (maps smaller than 4×4), `tail` the concatenations and the logistic,
  `decode` boxes and NMS. `total_alloc` and `mallocs` stay constant: a frame
  allocates nothing.
- `det`: anchor index, class (0-based, without the background), score, box
  corners x0 y0 x1 y1 as fractions of the input.

## Measured on the UnitV (TinyGo 0.43, `-opt=2`, CPU 390 MHz)

| | |
|---|---|
| image | 4,838,320 B (model 4,401,508, test input 196,608, program 240,204) |
| model address | `0x8003a780`, a multiple of 128 |
| `Parse` (CRC-32 over the file) | 87 ms |
| heap after setup | 759,008 B in use of 1,425,249 B |
| frame | **147.9 ms, 6.8 frames/s** (217 frames: 147.8–147.9 ms) |
| convert / upload / KPU / read-back | 32.6 / 13.7 / 43.2 / 7.6 ms |
| CPU convolutions / concatenations and logistic / decode | 37.0 / 10.2 / 3.6 ms |

Against the Go model of the KPU on the same input: the first seven KPU
layers are bit-equal; the eighth (a 1×1 convolution, 96 channels at 64×64)
is one step lower on 11 of 393,216 values, and the layers after it inherit
that: the first segment's output (24 KPU layers) is equal on 89.5 % of its
values and at most 4 steps off. Every other segment, run on the board's own
inputs, is bit-equal to the model. The detections on the test image are the
host's to four decimals.
