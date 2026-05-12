package rx

import "github.com/itohio/EasyRobot/x/devices/crsf"

// TelemetryFunc produces a frame to send, or nil if nothing to report this slot.
type TelemetryFunc func() *crsf.Frame

// Scheduler round-robins through telemetry providers, sending at most
// one frame per channel-frame reception to stay within the link budget.
type Scheduler struct {
	slots []TelemetryFunc
	idx   int
}

// Register adds a telemetry provider to the rotation.
func (s *Scheduler) Register(fn TelemetryFunc) {
	s.slots = append(s.slots, fn)
}

// Next returns the next telemetry frame to send, or nil if no slots have data.
// Cycles through all registered slots once per call to find a non-nil frame.
func (s *Scheduler) Next() *crsf.Frame {
	n := len(s.slots)
	if n == 0 {
		return nil
	}
	for range n {
		fn := s.slots[s.idx%n]
		s.idx = (s.idx + 1) % n
		if f := fn(); f != nil {
			return f
		}
	}
	return nil
}

// Len returns the number of registered telemetry providers.
func (s *Scheduler) Len() int {
	return len(s.slots)
}
