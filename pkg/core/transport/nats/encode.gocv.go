package nats

import (
	"time"

	"github.com/itohio/EasyRobot/pkg/core/store"
	"github.com/itohio/EasyRobot/pkg/core/transport"
)

func (s *nats) encode(in store.Store) ([]byte, error) {
	robot := transport.Robot{
		Timestamp: time.Now().UnixMicro(),
	}

	if idVal, ok := in.Get(store.ROBOT_ID); ok {
		if id, ok := idVal.(transport.RobotMsg); ok {
			robot.Robot = id
		}
	}
	if statusVal, ok := in.Get(store.ROBOT_STATUS); ok {
		if status, ok := statusVal.(transport.StatusMsg); ok {
			robot.Status = status
		}
	}

	if err := s.encodeStreams(&robot, in); err != nil {
		return nil, err
	}

	data, err := robot.Marshal()
	if err != nil {
		return nil, err
	}

	return data, nil
}

func (s *nats) encodeStreams(robot *transport.Robot, in store.Store) error {
	return nil
}
