package backend

import (
	"image"

	"github.com/foxis/EasyRobot/pkg/store"
)

type imageGetter struct {
	img image.Image
}

func FromImage(imt image.Image) store.ImageGetter {
	return &imageGetter{}
}

func (i *imageGetter) Image() image.Image {
	return i.img
}

func (i *imageGetter) Value() interface{} {
	return i.img
}
