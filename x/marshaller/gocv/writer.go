package gocv

import "errors"

var errStopLoop = errors.New("gocv: stop loop")

// frameWriter is an alias for StreamSink for backward compatibility.
// It will be gradually phased out in favor of StreamSink.
type frameWriter = StreamSink
