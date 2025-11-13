
# EasyRobot Devices: TinyGo and Raspberry Pi Compatibility

This directory documents support for various hardware devices in the EasyRobot ecosystem, focusing on compatibility with TinyGo, Raspberry Pi (RPI), and conventional computers. Devices fall into several key categories:

## 1. Hardware-Specific Devices

These require direct hardware interfaces (I2C, SPI, UART, GPIO, PWM, CAN, etc.) and typically run on microcontrollers with TinyGo **or** on Raspberry Pi, which exposes these interfaces through its GPIO header.

- **I2C Sensors (Accelerometers, Gyros, Magnetometers):**  
  Devices like MPU6050, LSM9DS1, etc. Accessible via TinyGo's `machine.I2C` implementation or via `/dev/i2c-*` devices on Raspberry Pi.
- **SPI Peripherals:**  
  Displays (OLED, ePaper), flash memory, sensors, etc., using `machine.SPI` with TinyGo or `/dev/spidev*` devices on Raspberry Pi.
- **PWM Devices and Actuators:**  
  Servos, DC motors, etc., using the PWM capabilities of microcontrollers or Raspberry Pi (using libraries like `pigpio` or kernel PWM interfaces).
- **GPIO Devices and Expanders:**  
  Simple buttons, switches, LEDs, or buses like MCP23017, expandable via both TinyGo and RPI GPIO.
- **CAN Bus Devices:**  
  For applications involving CAN-enabled microcontrollers or Raspberry Pi (using SPI-based CAN controllers such as MCP2515).
- **Other buses (1-Wire, UART, etc.):**  
  For example, temperature sensors (DS18B20) or GPS modules via direct connections.

## 2. Hardware-Agnostic Devices

Some devices do not require direct pin or bus access, and can be used from any platform, including laptops, desktops, Raspberry Pi, or microcontrollers.

- **Serial-Connected Devices:**  
  LIDARs, GNSS modules, and barcode scanners that present as serial ports (over USB or UART). EasyRobot drivers work on RPI (e.g., `/dev/ttyAMA0`, `/dev/serial*`), any Linux/macOS/Windows host, or TinyGo targets.
- **Serial-to-I2C/SPI/CAN Adapters:**  
  Bridges/adapters such as FT232H, MCP2221, or USB2CAN, allowing a host computer or Raspberry Pi to talk to I2C/SPI/CAN peripherals as if on-board.
- **Networked Devices:**  
  Sensors/controllers accessible over TCP/IP/UDP, usable from any OS or embedded system with networking.
- **Simulation/Stubs:**  
  Some device drivers offer simulation backends, for development and testing on platforms without hardware attached.

## 3. Platform Notes

- On **Raspberry Pi**, direct hardware access is provided via the Linux kernel:  
  - I2C/SPI/CAN interfaces appear as `/dev/i2c-*`, `/dev/spidev*`, `/dev/can*`.
  - GPIO/PWM can be accessed via `/sys/class/gpio`, `/sys/class/pwm`, or external libraries (e.g. `pigpio`, `wiringPi`).
  - Many Go libraries and several TinyGo builds support these interfaces directly.

- On **TinyGo MCUs**, peripherals are accessed natively via the `machine` package.

- On **Laptops/Desktops**, serial and USB bridges allow device usage without physical pin access.

## Example Scenarios

- **MCU + TinyGo:**  
  STM32/nRF board using `machine.I2C` for a sensor.
- **Raspberry Pi (direct pins):**  
  Using Pi pins for I2C/SPI/CAN/1-Wire to connect sensors, or for GPIO/PWM devices.
- **Raspberry Pi/Laptop (serial/bridge adapters):**  
  Reading LIDAR data over USB-serial, or talking to I2C sensor via FT232H.
- **Laptop/Desktop:**  
  Communicating with networked devices or USB bridges.

## Device Integration in EasyRobot

- Drivers select backends (TinyGo `machine`, RPI I/O, OS drivers, etc.) at runtime or compile time for portability.
- Where available, abstractions are provided so code works unchanged across MCUs, RPI, and general computers.

---

## Build Tag Guidelines

Proper use of Go/TinyGo build tags is **strongly recommended** to ensure code is built only on supported platforms and to help the build process select the correct implementation. This aids code portability and reliability.

### Recommendations

- **Device drivers that support only TinyGo or specific MCUs/boards:**  
  Use tags like  
  ```go
  //go:build tinygo && (stm32 || nrf52840 || arduino)
  ```
  Place these near the top of your file.  
  _Example: Only for STM32/NRF under TinyGo:_
  ```go
  //go:build tinygo && (stm32 || nrf52840)
  ```

- **Raspberry Pi or Single Board Computer (SBC) support:**  
  Use tags to target ARM/Linux or even Pi-only:  
  ```go
  //go:build linux && (arm || arm64)
  ```
  For only Pi (with additional runtime check, if necessary):  
  ```go
  //go:build linux
  ```
  (But confirm at runtime with features or device tree if your code is Pi-specific.)

- **Code generic for any OS/host (via serial/network/USB bridges):**  
  _No build tags required_, or use:  
  ```go
  //go:build !tinygo
  ```
  If not meant for TinyGo.

- **Simulation/shims:**  
  Use custom tags such as `sim` or `stub` for simulation backends:  
  ```go
  //go:build sim
  ```

### Example Tag Table

| Device                | Tag Example                                   | Use Case                                |
|-----------------------|-----------------------------------------------|-----------------------------------------|
| STM32 TinyGo          | `//go:build tinygo && stm32`                  | Only builds on STM32 via TinyGo         |
| Arduino (generic)     | `//go:build arduino`                          | Any Arduino board on TinyGo             |
| Raspberry Pi (All)    | `//go:build linux && (arm || arm64)`          | All ARM-based SBC with Linux (Inc. Pi)  |
| Pi 4 Only             | `//go:build linux && arm64`                   | RPi 4 (64-bit)                          |
| Host PC (No TinyGo)   | `//go:build !tinygo`                          | Laptops/Desktops                        |
| Simulator             | `//go:build sim`                              | Simulated/stub device                   |

- For cross-platform drivers, split implementations by platform and tag files accordingly.

---

## Notes

- Not all hardware APIs are available on all platforms or all Raspberry Pi revisions. Review [TinyGo Hardware Support](https://tinygo.org/docs/reference/microcontrollers/) and Raspberry Pi [interface documentation](https://www.raspberrypi.org/documentation/computers/raspberry-pi.html#io-pins).
- Some device drivers support simulation modes for hardware-free development.
- Provide necessary permissions on RPI (`i2c`, `spi`, `gpio`, etc.) or use `sudo` as required.
- Always consult device datasheets for voltage and protocol compatibility.

Refer to per-device docs in this directory for specific instructions for TinyGo, Raspberry Pi, and general computers.


