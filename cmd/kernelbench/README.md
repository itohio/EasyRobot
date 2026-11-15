# kernelbench

Times the EasyRobot kernels the tufty2040 badge demo runs per IMU sample or
per frame, on the badge or the host:

| Kernel | Calls per pass | Input |
|---|---:|---|
| `ahrs.MahonyAHRS.Update` (6-axis, KI 0, `*mat.Matrix3x3`) | 1000 | the 100-sample stream below |
| `strapdown.Nav.Update` | 1000 | the stream, with Mahony's attitude per sample |
| `calib.Calibrator.Update` | 1000 | the stream |
| `pdr.Detector.Update` | 1000 | the stream's accelerometer |
| `flip.Liquid.Step` and `Raster` at 24×18/400 (the demo's Water) | 300 | gravity swinging ±40° every 120 steps, after 120 untimed settling steps |
| `boids.Flock.Step`, 64 boids, spinning box | 300 | |

The stream is one second of a badge in the hand at 100 Hz from a fixed seed:
resting, tilting about x, then pushed back and forth along y. It is the same
on every machine.

Per kernel it prints ns per call (from a µs timer over the whole loop), the
heap allocated during the timed loop (must be 0) and an FNV of the kernel's
outputs. An optimisation that keeps the FNVs keeps the outputs bit for bit,
on the chip as well as the host.

```sh
go run ./cmd/kernelbench                                   # host: one pass
tinygo flash -target=tufty2040 ./cmd/kernelbench           # badge: passes for ever over serial
```

Host numbers from this loop swing with machine load; the packages'
`Benchmark*` functions (minimum of several `-count` runs) are the host
reference. Flip's per-phase split on the badge comes from a phase-timed copy
of the engine.
