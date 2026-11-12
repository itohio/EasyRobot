# qi8bench — qi8 kernel benchmark (host and ESP32-S3)

Runs the qi8 prepared layers at MobileNetV1-0.25 96×96 layer shapes (stem
conv 3×3/2, depthwise 3×3 and pointwise 1×1 at 48×48×8, 24×24×32 and
6×6×128) in both rounding modes, plus `Requant.Apply`/`ApplyTFLite` alone.
Per kernel it prints the time per call, per MAC (×1000) and per output —
CCOUNT cycles on the S3, nanoseconds on the host — and an FNV of the output:
the FNVs must stay the same across builds and code changes (bit-exactness).
The host side of the same shapes is `go test -bench MobileNet
./x/math/primitive/qi8/`. Results go to the evidence notes, not here.

```sh
# host
go run ./cmd/qi8bench

# ESP32-S3, PIE backend
tinygo build -target ./cmd/nnbench/target/xiao-esp32s3-noradio.json \
  -o qi8bench-pie.bin ./cmd/qi8bench
esptool --port /dev/cu.usbmodem101 --chip esp32s3 write-flash 0x0 qi8bench-pie.bin

# ESP32-S3, generic backend
tinygo build -target ./cmd/nnbench/target/xiao-esp32s3-noradio.json -tags qi8generic \
  -o qi8bench-generic.bin ./cmd/qi8bench
```

Read the USB serial console at 115200 baud; the report repeats every few
seconds.
