package bridge

import (
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/transport"
)

type Options struct {
	base      plugin.Options
	Network   string
	Address   string
	Transport transport.Transport
}
