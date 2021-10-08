package bridge

import (
	"github.com/itohio/EasyRobot/pkg/core/plugin"
	"github.com/itohio/EasyRobot/pkg/core/transport"
)

type Options struct {
	base      plugin.Options
	Network   string
	Address   string
	Transport transport.Transport
}
