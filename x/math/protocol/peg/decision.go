package peg

// Decision indicates what the caller should do with the current
// packet accumulation after evaluating a pattern.
type Decision int

const (
	DecisionContinue Decision = iota
	DecisionDrop
	DecisionEmit
)
