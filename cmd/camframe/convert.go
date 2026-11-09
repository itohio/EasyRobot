package main

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"

	"github.com/itohio/EasyRobot/drivers/camera/frame"
)

// toImage converts one decoded frame to an image: Y8 → *image.Gray, the
// RGB565/YUYV/RGB888-planar formats → *image.RGBA, JPEG → image/jpeg's
// decoder output. Row padding beyond the visible width (Stride) is ignored.
func toImage(m frame.Meta, data []byte) (image.Image, error) {
	if m.Format == frame.JPEG {
		img, err := jpeg.Decode(bytes.NewReader(data))
		if err != nil {
			return nil, fmt.Errorf("decode JPEG: %w", err)
		}
		return img, nil
	}
	if !m.Format.Known() {
		return nil, fmt.Errorf("unsupported format %d", m.Format)
	}
	if err := m.Validate(); err != nil {
		return nil, err
	}
	w, h, row := int(m.Width), int(m.Height), m.RowBytes()
	if len(data) < h*row {
		return nil, errors.New("image data shorter than height × stride")
	}
	rect := image.Rect(0, 0, w, h)

	switch m.Format {
	case frame.Y8:
		g := image.NewGray(rect)
		for y := range h {
			copy(g.Pix[y*g.Stride:y*g.Stride+w], data[y*row:])
		}
		return g, nil

	case frame.RGB565LE, frame.RGB565BE:
		img := image.NewRGBA(rect)
		for y := range h {
			src := data[y*row:]
			dst := img.Pix[y*img.Stride:]
			for x := range w {
				var v uint16
				if m.Format == frame.RGB565LE {
					v = uint16(src[2*x]) | uint16(src[2*x+1])<<8
				} else {
					v = uint16(src[2*x])<<8 | uint16(src[2*x+1])
				}
				r5, g6, b5 := byte(v>>11), byte(v>>5)&0x3F, byte(v)&0x1F
				dst[4*x+0] = r5<<3 | r5>>2
				dst[4*x+1] = g6<<2 | g6>>4
				dst[4*x+2] = b5<<3 | b5>>2
				dst[4*x+3] = 0xFF
			}
		}
		return img, nil

	case frame.YUYV:
		img := image.NewRGBA(rect)
		for y := range h {
			src := data[y*row:]
			dst := img.Pix[y*img.Stride:]
			for x := range w {
				pair := src[(x/2)*4:]
				// color.YCbCrToRGB is JFIF, i.e. BT.601 full range.
				r, g, b := color.YCbCrToRGB(pair[(x&1)*2], pair[1], pair[3])
				dst[4*x+0], dst[4*x+1], dst[4*x+2], dst[4*x+3] = r, g, b, 0xFF
			}
		}
		return img, nil

	case frame.RGB888Planar:
		// Stride counts all three planes: each plane row is row/3 bytes.
		img := image.NewRGBA(rect)
		pr := row / 3
		plane := pr * h
		for y := range h {
			dst := img.Pix[y*img.Stride:]
			for x := range w {
				i := y*pr + x
				dst[4*x+0], dst[4*x+1], dst[4*x+2], dst[4*x+3] = data[i], data[plane+i], data[2*plane+i], 0xFF
			}
		}
		return img, nil
	}
	return nil, fmt.Errorf("unsupported format %d", m.Format)
}
