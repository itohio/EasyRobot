package adaptive

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

type stubProcessor struct {
	id           string
	processCount int
	resetCount   int
	returnNil    int
	returnWrite  int
}

func (s *stubProcessor) Process(dst buffer.Writer[byte], src buffer.Reader[byte]) int {
	s.processCount++
	if dst == nil {
		return s.returnNil
	}
	return s.returnWrite
}

func (s *stubProcessor) Reset() {
	s.resetCount++
}

func TestAdaptiveProcessor_SelectsInitialTier(t *testing.T) {
	var created []*stubProcessor
	highFactory := func() chain.Processor[byte, byte] {
		p := &stubProcessor{id: "high", returnNil: 10, returnWrite: 1}
		created = append(created, p)
		return p
	}
	lowFactory := func() chain.Processor[byte, byte] {
		p := &stubProcessor{id: "low", returnNil: 5, returnWrite: 1}
		created = append(created, p)
		return p
	}

	tiers := []Tier[byte, byte]{
		{MinQuality: 20, Factory: highFactory},
		{MinQuality: 0, Factory: lowFactory},
	}

	ap, err := New(tiers, 5)
	require.NoError(t, err)

	dst := buffer.New(make([]byte, 0, 4))
	src := buffer.New(make([]byte, 0, 4))
	processed := ap.Process(dst, src)
	require.Equal(t, 1, processed)

	require.Len(t, created, 1)
	require.Equal(t, "low", created[0].id)
}

func TestAdaptiveProcessor_UpgradeAndDowngrade(t *testing.T) {
	var highInstances []*stubProcessor
	var lowInstances []*stubProcessor

	highFactory := func() chain.Processor[byte, byte] {
		p := &stubProcessor{id: "high", returnNil: 10, returnWrite: 2}
		highInstances = append(highInstances, p)
		return p
	}
	lowFactory := func() chain.Processor[byte, byte] {
		p := &stubProcessor{id: "low", returnNil: 5, returnWrite: 1}
		lowInstances = append(lowInstances, p)
		return p
	}

	tiers := []Tier[byte, byte]{
		{MinQuality: 30, Factory: highFactory},
		{MinQuality: -50, Factory: lowFactory},
	}

	ap, err := New(tiers, 40)
	require.NoError(t, err)
	require.Len(t, highInstances, 1)

	dst := buffer.New(make([]byte, 0, 2))
	src := buffer.New(make([]byte, 0, 2))

	require.Equal(t, 2, ap.Process(dst, src))
	require.Equal(t, 1, highInstances[0].processCount)

	ap.SetQuality(-10)
	require.Len(t, lowInstances, 1)
	require.Equal(t, 1, ap.Process(dst, src))
	require.Equal(t, 1, lowInstances[0].processCount)

	ap.SetQuality(35)
	require.Len(t, highInstances, 2)
	require.Equal(t, 2, ap.Process(dst, src))
	require.Equal(t, 1, highInstances[1].processCount)
}

func TestAdaptiveProcessor_ResetDelegates(t *testing.T) {
	var resets []*stubProcessor
	factory := func() chain.Processor[byte, byte] {
		p := &stubProcessor{returnWrite: 1}
		resets = append(resets, p)
		return p
	}

	ap, err := New([]Tier[byte, byte]{
		{MinQuality: 0, Factory: factory},
	}, 0)
	require.NoError(t, err)

	ap.Reset()
	require.Len(t, resets, 1)
	require.Equal(t, 1, resets[0].resetCount)

	ap.SetQuality(10)
	ap.Reset()
	require.Len(t, resets, 1)
	require.Equal(t, 2, resets[0].resetCount)
}

func TestAdaptiveProcessor_NilTierError(t *testing.T) {
	_, err := New([]Tier[byte, byte]{}, 0)
	require.Error(t, err)

	_, err = New([]Tier[byte, byte]{
		{MinQuality: 0, Factory: nil},
	}, 0)
	require.Error(t, err)
}
