# GoCV / OpenCV Installation Guide

This repository uses the `gocv.io/x/gocv` Go bindings. To build those
bindings locally you need a matching OpenCV toolchain installed on your
development machine. The supplied `Makefile.cv` automates fetching,
configuring, building, and installing the required OpenCV version.

## Supported Versions

- GoCV module version: `v0.42.0`
- Required OpenCV version: `4.12.0`

The make targets pin to these versions. If you need a different version,
override the `OPENCV_VERSION` variable when invoking `make`, but keep the
GoCV/OpenCV compatibility matrix in mind.

## Prerequisites

```sh
go get -u gocv.io/x/gocv

cd $GOPATH/pkg/mod/gocv.io/x/gocv

make install
```


