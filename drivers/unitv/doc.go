// Package unitv is the pin map of the M5Stack UnitV (Kendryte K210) for the
// TinyGo target drivers/unitv/target/unitv.json.
//
// The target inherits TinyGo's maixbit target, because TinyGo 0.43 defines the
// K210 pins and FPIOA functions only under the maixbit build tag. Its board
// constants (machine.LED = D13, discrete RGB LED) do not exist on UnitV: use
// the constants in this package instead.
//
// Build from the repository root (the target's linker-script path is relative
// to the working directory):
//
//	tinygo build -target=drivers/unitv/target/unitv.json -o hello.bin ./drivers/unitv/examples/hello
package unitv
