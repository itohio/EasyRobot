# Robot controllers

Embedded optimized platform-specific implementations and generic interfaces for various
robot manipulators, kinematics algorithms and device dravers.

Usually the code in this package is compiled using Tinygo

# Protobuf

Actuators are configured using structures generated from Protobuf definitions.
This way it is easy to transmit configuration data between devices via I2C/SPI/Serial/CAN.

Protobuf compiler and the following should be installed:

```
go install google.golang.org/protobuf/cmd/protoc-gen-go
```

Also, gogoproto package should be downloaded:

```
go get github.com/gogo/protobuf/protoc-gen-gogofaster
```


# Protocol

## ID
