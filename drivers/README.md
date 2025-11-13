
# EasyRobot Device "Nodes": Embedded Service Implementations (ROS Nomenclature)

> **Note:**  
> In accordance with [ROS terminology](https://docs.ros.org/en/rolling/Concepts/About-Nodes.html), modules in this directory are referred to as **nodes**.  
> Each "node" is a process (often on an embedded device) that provides an API for data exchange and control over hardware via the [dndm communication library](https://github.com/yourorg/dndm), using protocols such as serial, CAN, I2C, SPI, etc.

---

## What Are Device Nodes?

This directory contains **device nodes**—standalone executables or firmware modules that bridge hardware peripherals to higher-level dndm-based networks, analogous to ROS nodes interfacing with sensors, actuators, and other hardware.

Characteristics of these device nodes:

- **Protocol Adapters:**  
  Communicate with hardware via serial, I2C, SPI, CAN, USB, etc.
- **dndm Service Exposure:**  
  Each node acts as a dndm service, exposing hardware functionality for remote access through messages or RPC, using protobuf-defined APIs.
- **Abstraction & Composition:**  
  Hide board- and bus-specific logic behind clean, composable APIs.
- **Cross-Platform/Firmware Deployable:**  
  Mostly targeted at Microcontroller Units (MCUs) with TinyGo or similar, but can also serve simulation or local desktop test purposes.

---

## Why "Node" Instead of "Driver"?

In ROS, a **node** is any process that performs computation, typically communicating over topics, services, or actions.  
Here, "nodes" are the embedding of firmware or user-space processes that provide access to device features via dndm, rather than classic OS "drivers" located in kernel or HAL layers.

Other alternatives could be "device process", "hardware service", or "peripheral node", but **node** maps cleanly to the ROS model, emphasizing process-level service boundaries.

---

## Example Device Nodes

Examples of device nodes commonly managed here include:

- **Sensor Nodes:**  
  Deliver IMU, temperature, pressure, distance, or custom sensor data upstream via dndm.
- **Actuator Nodes:**  
  Provide external control of motors, servos, relays, LEDs, and other outputs using robust, high-level APIs.
- **Power & IO Nodes:**  
  Supply battery status, monitor power rails, manage GPIO expanders, or aggregate board-level diagnostics as dndm endpoints.
- **Bridging Nodes:**  
  Perform protocol bridging (serial-to-CAN, CAN-to-I2C, etc.) for hardware networks.
- **Custom Embedded Nodes:**  
  Support bespoke business logic, test harnesses, or other specialized embedded workflows.

---

## Integration Pattern

A typical node implementation in this directory:

1. **Hardware Access:**  
   Uses TinyGo’s `machine` package or similar to set up protocol-level access (I2C, SPI, UART, etc.).
2. **Node Service Logic:**  
   Handles requests/responses, streams, or periodic publishes according to dndm service definitions and proto contracts.
3. **dndm API Exposure:**  
   Registers node with dndm, using protobuf types for message/command interchange—compatible with remote ROS-like workflows.
4. **Multi-EndPoint Support:**  
   Nodes can multiplex multiple devices or channels under unified dndm endpoints.

---

## Design Principles

- **IDL-Driven:**  
  Proto files define all API contracts—nodes generate and use Go structs/types for robust communication.
- **Transport-Agnostic:**  
  Nodes can expose their dndm API via serial, CAN, TCP, or other supported backends.
- **Testable and Reusable:**  
  Designed for easy simulation/mocking; no hardcoded dependencies.

---

## Contributing

- Add or update a node-specific design document for every new node.
- Adhere to the high-level flow and patterns from the main documentation and dndm specs.
- Use ROS naming and communication concepts for clarity and consistency.

---




