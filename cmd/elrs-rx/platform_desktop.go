//go:build linux || windows

package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	devio "github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/crsf"
)

var serialPort = flag.String("serial", "", "Serial port (e.g. /dev/ttyUSB0 or COM3)")

func initHardware() {
	flag.Parse()
	if *serialPort == "" {
		fmt.Fprintln(os.Stderr, "usage: elrs-rx -serial <port>")
		os.Exit(1)
	}
}

func getStream() (io.Reader, io.Writer) {
	s, err := devio.NewSerialWithConfig(*serialPort, devio.SerialConfig{BaudRate: 420000})
	if err != nil {
		fmt.Fprintln(os.Stderr, "serial open:", err)
		os.Exit(1)
	}
	return s, s
}

func onChannels(ch *crsf.Channels) {
	fmt.Printf("CH: ")
	for i := range 6 {
		fmt.Printf("%4d ", ch.Microseconds(i))
	}
	fmt.Println()
}

func onLinkStats(ls *crsf.LinkStatistics) {
	fmt.Printf("LINK: RSSI=%ddBm Q=%d%% SNR=%ddB\n",
		ls.UplinkRSSI1, ls.UplinkQuality, ls.UplinkSNR)
}

func heartbeat() {
	// no-op on desktop
}

func readBatteryVoltage() float32 {
	return 12.0 // simulated
}
