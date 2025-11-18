package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	devio "github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/math/protocol/peg"
)

var (
	serialPort      = flag.String("port", "", "Serial port device (e.g., /dev/ttyUSB0 or COM3)")
	baudRate        = flag.Int("baud", 115200, "Serial port baud rate")
	headerHex       = flag.String("header", "AA", "Packet pattern (e.g., AA*L or AA**LL***1)")
	bytesPerLine    = flag.Int("bytes-per-line", 64, "Number of bytes per line in hex output (default: 64)")
	strictCRC       = flag.Bool("crc", false, "Discard packets with invalid CRC (default: warn only)")
	showHelp        = flag.Bool("h", false, "Print detailed packet parsing format guide")
	captureMode     = flag.Bool("capture", false, "Capture all raw data and print as hex to stdout")
	showText        = flag.Bool("text", false, "Show printable text alongside hex output (like hex viewers)")
	precisionDigits = flag.Int("precision", 2, "Decimal digits when printing float/derived values")
	maxPacketLen    = flag.Int("maxlen", 2048, "Maximum packet length in bytes (safety cap)")
)

// PacketProcessor handles packet processing logic
type PacketProcessor struct {
	program   *peg.Program
	printer   *Printer
	crcVal    *CRCValidator
	strictCRC bool
	maxLen    int
}

// NewPacketProcessor creates a new packet processor
func NewPacketProcessor(program *peg.Program, strictCRC bool, maxLen int) *PacketProcessor {
	crcVal := NewCRCValidator()
	printer := NewPrinter(crcVal, *precisionDigits)

	return &PacketProcessor{
		program:   program,
		printer:   printer,
		crcVal:    crcVal,
		strictCRC: strictCRC,
		maxLen:    maxLen,
	}
}

// ProcessPacket processes a complete packet
func (pp *PacketProcessor) ProcessPacket(state peg.State, packetCount int, isAlternative bool, showText bool) {
	packet := state.Packet()
	crcValid, _ := pp.crcVal.Validate(state)
	shouldDiscard := pp.strictCRC && !crcValid

	if shouldDiscard {
		slog.Debug("Discarded packet due to invalid CRC", "length", len(packet))
		return
	}

	fmt.Println("\n========================================")
	if isAlternative {
		fmt.Printf("Packet #%d (ALTERNATIVE, length: %d bytes)", packetCount, len(packet))
	} else {
		fmt.Printf("Packet #%d (length: %d bytes)", packetCount, len(packet))
	}
	if crcValid && pp.crcVal.HasCRC(state) {
		fmt.Print(" [CRC VALID]")
	}
	fmt.Println()
	pp.printer.PrintPacket(state, *bytesPerLine, isAlternative, pp.strictCRC, showText)
	fmt.Println("========================================")
}

// runCaptureMode handles raw data capture and prints hex to stdout
func runCaptureMode(ctx context.Context, ser devio.Serial, printer *Printer) error {
	slog.Info("Capture mode enabled",
		"port", *serialPort,
		"baud", *baudRate,
	)

	buf := make([]byte, 1024)
	lineBuffer := make([]byte, 0, *bytesPerLine)

	for {
		select {
		case <-ctx.Done():
			// Print any remaining bytes in buffer
			if len(lineBuffer) > 0 {
				printer.PrintHexLine(lineBuffer, *bytesPerLine, *showText)
			}
			return nil
		default:
		}

		n, err := ser.Read(buf)
		if n > 0 {
			// Process bytes in chunks to respect bytes-per-line
			for i := 0; i < n; i++ {
				lineBuffer = append(lineBuffer, buf[i])
				if len(lineBuffer) >= *bytesPerLine {
					printer.PrintHexLine(lineBuffer, *bytesPerLine, *showText)
					lineBuffer = lineBuffer[:0]
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				slog.Debug("EOF received (continuing)")
				time.Sleep(10 * time.Millisecond)
				continue
			}
			slog.Warn("Read error", "err", err)
		}
	}
}

// ProcessSerialStream processes the serial stream for packet patterns
func processSerialStream(ctx context.Context, ser devio.Serial, processor *PacketProcessor) error {
	buf := make([]byte, 1024)
	packets := make([]peg.State, 0)
	packetCount := 0

	for {
		select {
		case <-ctx.Done():
			slog.Info("Stopping", "reason", "context cancelled", "packets", packetCount)
			return nil
		default:
		}

		n, err := ser.Read(buf)
		if n > 0 {
			packets, packetCount = processor.processBytes(buf[:n], packets, packetCount)
		}
		if err != nil {
			if err == io.EOF {
				slog.Debug("EOF received (continuing)", "packets", packetCount)
				time.Sleep(10 * time.Millisecond)
				continue
			}
			slog.Warn("Read error", "err", err, "packets", packetCount)
		}
	}
}

// processBytes processes incoming bytes using the PEG parser
// Follows the pattern from SPEC.md: maintain a list of packet states,
// append bytes to all, then evaluate and handle decisions
func (pp *PacketProcessor) processBytes(data []byte, packets []peg.State, packetCount int) ([]peg.State, int) {
	// Process each byte individually
	for _, b := range data {
		// Create new packet state if list is empty (per SPEC.md example)
		// This allows pattern matching to restart when all packets are dropped/emitted
		if len(packets) == 0 {
			state := peg.NewDefaultState()
			if pp.maxLen > 0 {
				state.SetMaxLength(pp.maxLen)
			}
			packets = append(packets, state)
		}

		// Add byte to all packet states
		for _, packet := range packets {
			packet.AppendPacket(b)
		}

		// Evaluate all packets
		for i := len(packets) - 1; i >= 0; i-- {
			packet := packets[i]
			decision, err := pp.program.Decide(packet)
			if err != nil {
				slog.Debug("Packet evaluation error", "err", err, "length", packet.CurrentLength())
				// Remove packet on error
				packets = append(packets[:i], packets[i+1:]...)
				continue
			}

			packetLen := packet.CurrentLength()
			slog.Debug("Packet decision",
				"decision", decision,
				"length", packetLen,
				"declared_length", packet.DeclaredLength(),
				"max_length", packet.MaxLength(),
			)

			switch decision {
			case peg.DecisionDrop:
				// Remove dropped packet
				slog.Debug("Dropping packet", "length", packetLen)
				packets = append(packets[:i], packets[i+1:]...)
			case peg.DecisionEmit:
				// Process complete packet
				packetCount++
				slog.Info("Emitting packet", "count", packetCount, "length", packetLen)
				pp.ProcessPacket(packet, packetCount, false, *showText)
				// Remove emitted packet
				packets = append(packets[:i], packets[i+1:]...)
			case peg.DecisionContinue:
				// Keep accumulating
				// Check for max length
				if pp.maxLen > 0 && packetLen > pp.maxLen {
					slog.Debug("Dropping packet (exceeded max length)", "length", packetLen, "max", pp.maxLen)
					packets = append(packets[:i], packets[i+1:]...)
				}
			}
		}
	}

	// Limit number of active packets to prevent memory issues
	maxPackets := 10
	if len(packets) > maxPackets {
		// Keep only the most recent packets
		packets = packets[len(packets)-maxPackets:]
	}

	return packets, packetCount
}

func printFormatGuide() {
	fmt.Println("Packet Pattern Format Guide")
	fmt.Println("===========================")
	fmt.Println()
	fmt.Println("BASIC SYNTAX:")
	fmt.Println("  - Hex bytes: AA, BB, 1C, etc. (exact match)")
	fmt.Println("  - Wildcard: * (ignore/skip one byte)")
	fmt.Println("  - Offset: #N (jump to absolute byte offset N)")
	fmt.Println()
	fmt.Println("LENGTH FIELDS:")
	fmt.Println("  - L     : 1-byte length field")
	fmt.Println("  - LL    : 2-byte length field (big-endian)")
	fmt.Println("  - ll    : 2-byte length field (little-endian)")
	fmt.Println()
	fmt.Println("DECODE FIELDS (use %% prefix):")
	fmt.Println("  Integers (signed):")
	fmt.Println("    %%i     : int8")
	fmt.Println("    %%ii    : int16 (little-endian)")
	fmt.Println("    %%II    : int16 (big-endian)")
	fmt.Println("    %%iii   : int24 (little-endian)")
	fmt.Println("    %%III   : int24 (big-endian)")
	fmt.Println("    %%iiii  : int32 (little-endian)")
	fmt.Println("    %%IIII  : int32 (big-endian)")
	fmt.Println()
	fmt.Println("  Integers (unsigned):")
	fmt.Println("    %%u     : uint8")
	fmt.Println("    %%uu    : uint16 (little-endian)")
	fmt.Println("    %%UU    : uint16 (big-endian)")
	fmt.Println("    %%uuu   : uint24 (little-endian)")
	fmt.Println("    %%UUU   : uint24 (big-endian)")
	fmt.Println("    %%uuuu  : uint32 (little-endian)")
	fmt.Println("    %%UUUU  : uint32 (big-endian)")
	fmt.Println()
	fmt.Println("  Floating point:")
	fmt.Println("    %%f     : float32 (little-endian)")
	fmt.Println("    %%F     : float64 (little-endian)")
	fmt.Println()
	fmt.Println("  CRC:")
	fmt.Println("    %%c     : CRC8")
	fmt.Println("    %%cc    : CRC16 cumulative (little-endian)")
	fmt.Println()
	fmt.Println("  Arrays:")
	fmt.Println("    %%5f    : array of 5 float32 values")
	fmt.Println("    %%10u   : array of 10 uint8 values")
	fmt.Println("    %%3ii   : array of 3 int16 values (little-endian)")
	fmt.Println()
	fmt.Println("EXAMPLES:")
	fmt.Println("  Simple header:")
	fmt.Println("    AA                    -> Match 0xAA byte")
	fmt.Println()
	fmt.Println("  Header with length:")
	fmt.Println("    AA*L                  -> 0xAA, skip 1 byte, 1-byte length")
	fmt.Println("    AA**LL                -> 0xAA, skip 2 bytes, 2-byte length (big-endian)")
	fmt.Println()
	fmt.Println("  With offset jump:")
	fmt.Println("    AA#7%%ii               -> 0xAA at start, jump to offset 7, read int16")
	fmt.Println("    AA#7%%ii%%5ii           -> 0xAA, jump to offset 7, read int16, then array of 5 int16")
	fmt.Println()
	fmt.Println("  Complex pattern:")
	fmt.Println("    AA*L%%u%%ii%%cc          -> 0xAA, skip 1, length, uint8, int16, CRC16")
	fmt.Println()
	fmt.Println("  With arrays:")
	fmt.Println("    AA*L%%10u%%cc           -> 0xAA, skip 1, length, array of 10 uint8, CRC16")
	fmt.Println()
	fmt.Println("FLAGS:")
	fmt.Println("  -crc                    : Discard packets with invalid CRC (default: warn only)")
	fmt.Println("  -bytes-per-line N        : Number of bytes per line in hex output (default: 64)")
	fmt.Println("  -capture                 : Capture all raw data and print as hex to stdout")
	fmt.Println("  -text                    : Show printable text alongside hex output (like hex viewers)")
	fmt.Println()
	fmt.Println("NOTES:")
	fmt.Println("  - Exact matches (hex bytes) must always match for packet detection")
	fmt.Println("  - Offset (#N) jumps to absolute byte position N from packet start")
	fmt.Println("  - Length fields are used to determine packet boundaries")
	fmt.Println("  - CRC fields are validated; use -crc to discard invalid packets")
	fmt.Println("  - Alternative packets with valid CRC are automatically promoted to main packets")
	fmt.Println("  - Capture mode writes raw binary data for later analysis")
	fmt.Println()
}

func main() {
	flag.Parse()

	if *showHelp {
		printFormatGuide()
		os.Exit(0)
	}

	if *serialPort == "" {
		slog.Error("Serial port required", "flag", "--serial")
		flag.Usage()
		os.Exit(1)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	config := devio.DefaultSerialConfig()
	config.BaudRate = *baudRate
	ser, err := devio.NewSerialWithConfig(*serialPort, config)
	if err != nil {
		slog.Error("Failed to open serial port", "port", *serialPort, "baud", *baudRate, "err", err)
		os.Exit(1)
	}
	defer func() {
		slog.Info("Closing serial port", "port", *serialPort)
		ser.Close()
	}()

	if *captureMode {
		crcVal := NewCRCValidator()
		printer := NewPrinter(crcVal, *precisionDigits)
		if err := runCaptureMode(ctx, ser, printer); err != nil {
			slog.Error("Capture mode failed", "err", err)
			os.Exit(1)
		}
		return
	}

	// Use pattern as-is; max length can be set via $N syntax or via state.SetMaxLength()
	pattern := *headerHex

	program, err := peg.Compile(pattern)
	if err != nil {
		slog.Error("Failed to compile packet pattern", "pattern", *headerHex, "err", err)
		os.Exit(1)
	}

	// Set max length on program if specified (via state.SetMaxLength in processBytes)

	slog.Info("Serial port opened successfully",
		"port", *serialPort,
		"baud", *baudRate,
		"pattern", *headerHex,
	)

	fmt.Println("Reading data from serial port (press Ctrl+C to stop)...")
	fmt.Printf("Pattern: %s\n", *headerHex)
	if *maxPacketLen > 0 {
		fmt.Printf("Max packet length: %d bytes\n", *maxPacketLen)
	}
	fmt.Println("Packet delimiter: ========================================")
	fmt.Println()

	processor := NewPacketProcessor(program, *strictCRC, *maxPacketLen)
	if err := processSerialStream(ctx, ser, processor); err != nil {
		slog.Error("Serial stream processing failed", "err", err)
		os.Exit(1)
	}
}
