package gocv

import (
	"fmt"
	"image"
	"strings"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

func normalizeImageFormat(format string) string {
	format = strings.ToLower(strings.TrimSpace(format))
	switch format {
	case "", "png", ".png":
		return ".png"
	case "jpg", ".jpg", "jpeg", ".jpeg":
		return ".jpg"
	case "bmp", ".bmp":
		return ".bmp"
	default:
		if !strings.HasPrefix(format, ".") {
			format = "." + format
		}
		return format
	}
}

func encodeMatBytes(mat cv.Mat, format string) ([]byte, error) {
	if mat.Empty() {
		return nil, fmt.Errorf("gocv: cannot encode empty mat")
	}
	ext := normalizeImageFormat(format)
	buf, err := cv.IMEncode(cv.FileExt(ext), mat)
	if err != nil {
		return nil, fmt.Errorf("gocv: encode mat: %w", err)
	}
	defer buf.Close()

	bytes := append([]byte(nil), buf.GetBytes()...)
	return bytes, nil
}

func decodeMatBytes(data []byte, flag cv.IMReadFlag) (cv.Mat, error) {
	mat, err := cv.IMDecode(data, flag)
	if err != nil {
		return cv.Mat{}, fmt.Errorf("gocv: decode mat: %w", err)
	}
	if mat.Empty() {
		mat.Close()
		return cv.Mat{}, fmt.Errorf("gocv: decoded empty mat")
	}
	return mat, nil
}

func encodeImageBytes(img image.Image, format string) ([]byte, error) {
	if img == nil {
		return nil, fmt.Errorf("gocv: nil image")
	}

	matRGBA, err := cv.ImageToMatRGBA(img)
	if err != nil {
		return nil, fmt.Errorf("gocv: convert image to mat: %w", err)
	}
	defer matRGBA.Close()

	matBGR := cv.NewMat()
	defer matBGR.Close()

	cv.CvtColor(matRGBA, &matBGR, cv.ColorRGBAToBGR)

	return encodeMatBytes(matBGR, format)
}

func decodeImageBytes(data []byte) (image.Image, error) {
	mat, err := decodeMatBytes(data, cv.IMReadColor)
	if err != nil {
		return nil, err
	}
	defer mat.Close()

	img, err := mat.ToImage()
	if err != nil {
		return nil, fmt.Errorf("gocv: mat to image: %w", err)
	}
	return img, nil
}

func matToTensor(mat cv.Mat, cfg config, dtype tensortypes.DataType) (types.Tensor, error) {
	opts := append([]tensorgocv.Option{}, cfg.tensorOpts...)
	opts = append(opts, tensorgocv.WithAdoptedMat())
	tensor, err := tensorgocv.FromMat(mat, opts...)
	if err != nil {
		mat.Close()
		return nil, err
	}
	_ = dtype // reserved for future conversions
	return tensor, nil
}
