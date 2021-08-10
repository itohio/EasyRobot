// +build !logless

package logger

import (
	"os"

	"github.com/rs/zerolog"
	logger "github.com/rs/zerolog/log"
)

var Log = logger.With().Caller().Logger().Output(zerolog.ConsoleWriter{Out: os.Stderr})

func init() {
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
}
