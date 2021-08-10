package backend

import (
	"image"
	"runtime"

	"gocv.io/x/gocv"
)

type gocvMatGetter struct {
	mat gocv.Mat
}

func FromGoCVMat(mat gocv.Mat) *gocv.Mat {
	matPtr := &mat
	runtime.SetFinalizer(matPtr, func(obj *gocv.Mat) {
		obj.Close()
	})
	return matPtr
}

func NewGoCVMat() *gocv.Mat {
	return FromGoCVMat(gocv.NewMat())
}

func (i *gocvMatGetter) Image() image.Image {
	img, _ := i.mat.ToImage()
	return img
}

func (i *gocvMatGetter) Value() interface{} {
	return i.mat
}
