# Kinematics for Wheeled Platforms

This module groups kinematic solvers for different drive bases.
Concrete implementations reside in subpackages:

- `differential`: two-wheel differential drive
- `mecanum`: omnidirectional mecanum platforms
- `steer4`: front-steer four-wheel bases
- `steer4dual`: four-wheel steering (front and rear)
- `steer6`: six-wheel rover with steerable end axles
- Additional steering-enabled variants (see `DESIGN.md`)