package bridge

import (
	"github.com/foxis/EasyRobot/pkg/plugin"
	"github.com/foxis/EasyRobot/pkg/transport"
)

type Options struct {
	base      plugin.Options
	Network   string
	Address   string
	Transport transport.Transport
}
