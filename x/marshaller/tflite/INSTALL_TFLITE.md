# Installing TensorFlow Lite C Library

This guide explains how to install the TensorFlow Lite C library to enable using `go-tflite` or `tflitego` libraries.

## Why This Approach?

**You're right!** Using existing TFLite Go libraries (like `go-tflite`) is much easier than:
- Manually compiling FlatBuffers schemas
- Writing a custom parser

The **only** requirement is installing the TensorFlow Lite C library.

## Installation Options

### Option A: Pre-built Binary (Easiest)

# ⚠️ Google never published a prebuilt `libtensorflowlite_c` archive for 2.2.0-rc3. You must build it yourself or carefully grab a community build. The steps below assume you build from source (see Option B).

```bash
# Extract to /usr/local
sudo tar -C /usr/local -xzf libtensorflowlite_c-2.2.0-rc3-linux-x86_64.tar.gz

# Update library cache
sudo ldconfig

# Verify installation
ls -la /usr/local/lib/libtensorflowlite*
```

> Google did not publish a macOS archive for 2.2.0-rc3 either. Build from source (Option B) or install via a package manager that ships the older headers/libraries.

### Option B: Build from Source (recommended)

If pre-built binaries don't work:

```bash
# Clone TensorFlow at the required tag
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout v2.2.0-rc3

# Build the TensorFlow Lite C library
bazel build --config=opt //tensorflow/lite/c:tensorflowlite_c

# Install the shared object and headers
sudo cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so /usr/local/lib/
sudo mkdir -p /usr/local/include/tensorflow/lite/c
sudo cp tensorflow/lite/c/*.h /usr/local/include/tensorflow/lite/c/
sudo ldconfig
```
If you need a static library instead, replace the Bazel target with `//tensorflow/lite/c:tensorflowlite_c` and copy the `.a` file.

### Option C: Use Docker

Create a Dockerfile with TFLite pre-installed:

```dockerfile
FROM golang:1.21

# Install build tooling (bazelisk works well for pinned versions)
RUN apt-get update && apt-get install -y bazelisk build-essential git python3 && rm -rf /var/lib/apt/lists/*

# Fetch TensorFlow source and build the 2.2.0-rc3 TFLite C library
RUN git clone https://github.com/tensorflow/tensorflow.git /tmp/tensorflow && \
    cd /tmp/tensorflow && \
    git checkout v2.2.0-rc3 && \
    bazelisk build --config=opt //tensorflow/lite/c:tensorflowlite_c && \
    cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so /usr/local/lib/ && \
    mkdir -p /usr/local/include/tensorflow/lite/c && \
    cp tensorflow/lite/c/*.h /usr/local/include/tensorflow/lite/c/ && \
    ldconfig && \
    rm -rf /tmp/tensorflow

WORKDIR /app
```

## After Installation

Once the C library is installed, you can use `go-tflite`:

```bash
# Install Go TFLite library
go get github.com/mattn/go-tflite

# Test it works
cd pkg/core/marshaller/tflite
go test -v
```

## Verify Installation

Check if the library is properly installed:

```bash
# Check library exists
ls -la /usr/local/lib/libtensorflowlite*

# Check headers
ls -la /usr/local/include/tensorflow/lite/c

# Check library is in cache
ldconfig -p | grep tensorflowlite
```

## Troubleshooting

### Error: `cannot find -ltensorflowlite_c`

The library is not in the search path or the version differs from `2.2.0-rc3` (required by go-tflite). Ensure you built from the correct tag and then try:

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export CGO_LDFLAGS="-L/usr/local/lib"
export CGO_CPPFLAGS="-I/usr/local/include"
```

Add these to your `~/.bashrc` or `~/.zshrc` to make permanent.

### Error: Missing headers

The C library package includes headers, but if missing:

```bash
# Download TensorFlow repository for headers
git clone --depth 1 https://github.com/tensorflow/tensorflow.git /tmp/tf
sudo cp -r /tmp/tf/tensorflow/lite/c/*.h /usr/local/include/tensorflow/lite/c/
```

### macOS: Library Not Loaded

On macOS, you may need to set `DYLD_LIBRARY_PATH`:

```bash
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

## Alternative: Use FlatBuffers Schema

If you **cannot** install the C library, fall back to FlatBuffers schema compilation (see `IMPLEMENTATION_PLAN.md`). But using the C library is **much easier**!

## Next Steps

Once installed:

1. Replace `stub.go` with actual implementation using `go-tflite`
2. Follow the example in the web search results for basic usage
3. Implement layer extraction and conversion to EasyRobot format
4. Test with MNIST model

See `IMPLEMENTATION_PLAN.md` for detailed steps.

