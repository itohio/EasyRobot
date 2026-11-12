// Command kmodelc compiles an int8 TFLite model into a Kendryte kmodel v3
// for the K210's KPU (drivers/kpu), in pure Go:
//
//	kmodelc [-input HxW] [-cut N] in.tflite out.kmodel
//	kmodelc -list in.tflite
//
// The TFLite file is read with qnet/tfl, optionally cut after qnet layer N
// (-list shows the layers) and resized to an H×W input, then compiled by
// drivers/kpu/compile. Every layer up to the cut must run on the KPU
// (convolutions with a fused Pad, ReLU/ReLU6 or pool); otherwise kmodelc
// names the first one that does not and writes nothing. TF SAME stride 2,
// as in the official MobileNet exports, is supported.
//
// The kmodel's input and output are uint8 CHW (int8 value + 128) with the
// printed shapes and quantisation.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/kmodel"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/nn/qnet/tfl"
)

func main() {
	input := flag.String("input", "", "resize the input to HxW (e.g. 256x256)")
	cut := flag.Int("cut", -1, "keep qnet layers 0..N (see -list)")
	list := flag.Bool("list", false, "list the model's qnet layers and exit")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: kmodelc [-input HxW] [-cut N] in.tflite out.kmodel\n       kmodelc -list in.tflite")
		flag.PrintDefaults()
	}
	flag.Parse()
	if err := run(*input, *cut, *list, flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, "kmodelc:", err)
		os.Exit(1)
	}
}

func run(input string, cut int, list bool, args []string) error {
	if (list && len(args) != 1) || (!list && len(args) != 2) {
		flag.Usage()
		return fmt.Errorf("wrong arguments")
	}
	data, err := os.ReadFile(args[0])
	if err != nil {
		return err
	}
	if list {
		g, err := tfl.Load(data)
		if err != nil {
			return err
		}
		for i, l := range g.Layers {
			fmt.Printf("%3d %-12v %v → %v  k%dx%d s%d  out scale %g zp %d\n", i, l.Kind, l.In, l.Out, l.KH, l.KW, l.StrideH, l.OutQ.Scale, l.OutQ.ZP)
		}
		return nil
	}
	var opts []types.Option
	if cut >= 0 {
		opts = append(opts, kmodel.WithCut(cut))
	}
	if input != "" {
		hs, ws, ok := strings.Cut(input, "x")
		h, err1 := strconv.Atoi(hs)
		w, err2 := strconv.Atoi(ws)
		if !ok || err1 != nil || err2 != nil {
			return fmt.Errorf("-input %q: want HxW", input)
		}
		opts = append(opts, kmodel.WithInputSize(h, w))
	}
	m := kmodel.NewMarshaller(opts...)
	var buf bytes.Buffer
	if err := m.Marshal(&buf, data); err != nil {
		return err
	}
	if err := os.WriteFile(args[1], buf.Bytes(), 0o644); err != nil {
		return err
	}
	i := m.Info
	fmt.Printf("wrote %s: %d bytes, %d KPU layers\n", args[1], buf.Len(), i.KPULayers)
	fmt.Printf("input  %v int8 (scale %g, zp %d) → kmodel input, uint8 CHW = int8 + 128\n", i.Input, i.InputQ.Scale, i.InputQ.ZP)
	fmt.Printf("output %v int8 (scale %g, zp %d) → kmodel output 0, uint8 CHW = int8 + 128\n", i.Output, i.OutputQ.Scale, i.OutputQ.ZP)
	return nil
}
