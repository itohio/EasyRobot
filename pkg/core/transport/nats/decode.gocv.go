package nats

import (
	"github.com/itohio/EasyRobot/pkg/core/store"
	"github.com/itohio/EasyRobot/pkg/core/transport"
)

func (s *nats) decode(data []byte) (store.Store, error) {
	var robot transport.Robot

	if err := robot.Unmarshal(data); err != nil {
		return nil, err
	}

	out := store.NewWithName(s.base.Name)
	out.Set(store.ROBOT_ID, robot.Robot)
	out.Set(store.ROBOT_STATUS, robot.Status)
	out.Set(store.EVENTS, robot.Events)
	out.Set(store.SIGNATURE, robot.Signature)

	s.decodeStreams(out, robot)

	return out, nil
}

func (s *nats) decodeStreams(out store.Store, robot transport.Robot) error {
	return nil
}
