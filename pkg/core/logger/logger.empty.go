// +build logless

package logger

var (
	Log = EmptyLog{}
)

type EmptyLog struct{}

func (l EmptyLog) Debug() EmptyLog   { return l }
func (l EmptyLog) Error() EmptyLog   { return l }
func (l EmptyLog) Warning() EmptyLog { return l }
func (l EmptyLog) Warn() EmptyLog    { return l }
func (l EmptyLog) Info() EmptyLog    { return l }

func (l EmptyLog) Msg(string) EmptyLog { return l }
func (l EmptyLog) Err(error) EmptyLog  { return l }

func (l EmptyLog) Int(string, int) EmptyLog       { return l }
func (l EmptyLog) Str(string, string) EmptyLog    { return l }
func (l EmptyLog) Float(string, float64) EmptyLog { return l }

func (l EmptyLog) Ints(string, []int) EmptyLog       { return l }
func (l EmptyLog) Strs(string, []string) EmptyLog    { return l }
func (l EmptyLog) Floats(string, []float64) EmptyLog { return l }
