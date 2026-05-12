package rx

import (
	"testing"

	"github.com/itohio/EasyRobot/x/devices/crsf"
)

func TestSchedulerEmpty(t *testing.T) {
	var s Scheduler
	if f := s.Next(); f != nil {
		t.Errorf("expected nil from empty scheduler, got %v", f)
	}
}

func TestSchedulerRoundRobin(t *testing.T) {
	var s Scheduler
	s.Register(func() *crsf.Frame { return &crsf.Frame{Type: crsf.TypeBattery} })
	s.Register(func() *crsf.Frame { return nil })
	s.Register(func() *crsf.Frame { return &crsf.Frame{Type: crsf.TypeGPS} })

	f1 := s.Next()
	if f1 == nil || f1.Type != crsf.TypeBattery {
		t.Errorf("1st = %v, want Battery", f1)
	}
	f2 := s.Next()
	if f2 == nil || f2.Type != crsf.TypeGPS {
		t.Errorf("2nd = %v, want GPS (skips nil)", f2)
	}
	f3 := s.Next()
	if f3 == nil || f3.Type != crsf.TypeBattery {
		t.Errorf("3rd = %v, want Battery (wrap)", f3)
	}
}

func TestSchedulerAllNil(t *testing.T) {
	var s Scheduler
	s.Register(func() *crsf.Frame { return nil })
	s.Register(func() *crsf.Frame { return nil })
	if f := s.Next(); f != nil {
		t.Errorf("expected nil when all slots empty, got %v", f)
	}
}

func TestSchedulerLen(t *testing.T) {
	var s Scheduler
	if s.Len() != 0 {
		t.Errorf("empty len = %d", s.Len())
	}
	s.Register(func() *crsf.Frame { return nil })
	s.Register(func() *crsf.Frame { return nil })
	if s.Len() != 2 {
		t.Errorf("len = %d, want 2", s.Len())
	}
}
