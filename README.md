# EasyRobot Golang implementation

This repo brings together EasyLocomotion, EasyVision, EasyNetwork concepts and ideas into one big cross platform project.

The biggest idea is that you can develop your algorithms and AI pipelines/graphs on whatever platform(even the cloud!) you wish and then be able to reproduce the same behavior on mobile robot platform (e.g. Raspberry Pi, Nano Pi or even ARM microcontrollers).

NOTE: You still need to keep track of the data types that are being passed around from step to step.

# Architecture

## Plugin system

In order to register a plugin, you must first import it.

## Backend implementations
E.g. OpenCV set of algorithms will operate on `gocv.Mat`, whilst TF implementation might use appropriate tensors and might need conversions.

### With*GoCV methods and steps
These steps/processors operate on `gocv.Mat` only and require GoCV to build.

### With*TF methods and steps

### With*TFLite methods and steps

### With*Image methods and steps
These steps operate on `image.Image` only.

### Display GIO
This sink will accept either `gocv.Mat`, `image.Image` or TF/TFLite tensors depending on build tags.

# Contribute

# Features

## Core
  [ ] Metrics
    [ ] registrable and optional metrics
    [ ] Prometheus
  [ ] Logs
    [x] Zero Log as backend
    [x] be able to optimize away the logs (for embedded: build tag `logless`)
  [ ] arbitrary data framework (store)
    [x] interface implementation
    [ ] Helper methods for get/set for common types
    [ ] unit tests
    [ ] benchmarks
  [ ] plugin framework
    [x] interface implementation
    [ ] unit tests
    [ ] benchmarks
  [ ] pipeline framework
    [x] generic implementation
      [x] image source
      [x] sink
      [x] processor
      [x] fan in
      [x] fan out
      [x] frame syncronizer
      [x] bridge
    [ ] unit tests
    [ ] benchmarks
    [x] Options marshalling
    [ ] save pipeline to json
    [ ] load pipeline from json
    [ ] pipeline elements referenced by NamedStep interface
  [ ] transports
    [ ] tcp
    [ ] udp
    [ ] EasyLocomotion C library

## EasyVision
  [ ] transforms
    [x] color transform
    [x] mat<->image
    [ ] undistort transform
    [ ] stereo rectify 
  [ ] extractors
    [x] features
      [x] ORB
      [x] SIFT
      [ ] calibration corners
    [x] OpenCV DNN
    [ ] tensorflow
    [ ] tensorflow lite
  [ ] algorithms
    [ ] monocular odometry
    [ ] stereo odometry
    [ ] map building
  [ ] sinks
    [ ] silent OpenCV image consumer
    [ ] OpenCV video/image writer
    [x] OpenCV image viewer
    [ ] Gio+OpenCV interface
  [ ] sources
    [x] OpenCV video/device/image reader
  [ ] tools
    [x] read and write video streams
    [ ] calibrate mono camera
    [ ] calibrate stereo camera
    [ ] learn hsv
    [ ] pipeline/graph editor/viewer
    [ ] standalone pipeline runner

## EasyLocomotion
  [ ] Locomotion
    [ ] Differential drive robot
    [ ] Omni wheel robot
    [ ] Mechanum wheel robot
    [ ] Hexapod
    [ ] Quadrupod
    [ ] Biped
  [ ] FK/IK Solvers
    [ ] Native
    [ ] gonum
    [ ] gorgonia
  [ ] Actuators
    [ ] PWM motor drivers
    [ ] PWM motor drivers with encoder
    [ ] Servo motor drivers
  [ ] Sensors
    [ ] Range sensors

## EasyNetwork
  [ ] Backend
    [ ] Native port from EasyNetwork
    [ ] gorgonia
    [ ] TFLite
  [ ] Q Learning algorithms
    [ ] DQN
    [ ] PPO


## Integrations
  [ ] Frontends
    [ ] Fyne
    [ ] GIO
    [ ] OpenCV
  [ ] Backend
    [ ] Native linear algebra (EasyLocomotion/EasyNetwork ports)
    [ ] gonum
    [ ] gorgonia
    [ ]
  [ ] Simulators
    [ ] GoBot
    [ ] OpenAI gym
    [ ] Webots
    [ ] Gazebo
    [ ] CopelliaSim
    [ ] ISAAC
    [ ] Platform
