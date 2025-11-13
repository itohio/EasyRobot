# Eager Tensor Operations Benchmark Report

**Generated:** November 10, 2025
**Platform:** Linux (amd64)
**CPU:** Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
**Package:** `github.com/itohio/EasyRobot/pkg/core/math/tensor/eager_tensor`

## Overview

This report contains benchmark results for eager tensor operations, comparing in-place and destination-based implementations across different tensor sizes and dimensions. All operations are implemented using optimized primitives from the `primitive` package.

## Test Configuration

- **Platform**: Linux (amd64)
- **CPU**: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
- **Benchmark Duration**: Variable per test
- **Data Type**: float32
- **Operations Tested**: Element-wise binary, scalar, unary, activation, and utility functions

## Tensor Sizes

| Name | Elements | Shape | Description |
|------|----------|-------|-------------|
| 1K | 1,000 | [1000] | Small 1D tensor |
| 10K | 10,000 | [10000] | Medium 1D tensor |
| 100K | 100,000 | [100000] | Large 1D tensor |
| 1M | 1,000,000 | [1000000] | Very large 1D tensor |
| 2D_100x100 | 10,000 | [100, 100] | Medium 2D tensor |
| 2D_1000x100 | 100,000 | [1000, 100] | Large 2D tensor |
| 3D_50x50x50 | 125,000 | [50, 50, 50] | Medium 3D tensor |

## Element-Wise Binary Operations

### Add Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 2,915 | 5 | 544 |
| 1K | Destination | 2,172 | 5 | 544 |
| 10K | In-Place | 13,026 | 5 | 544 |
| 10K | Destination | 14,142 | 5 | 544 |
| 100K | In-Place | 164,595 | 5 | 544 |
| 100K | Destination | 292,458 | 5 | 544 |
| 1M | In-Place | 2,613,882 | 5 | 544 |
| 1M | Destination | 3,633,297 | 5 | 544 |
| 2D_100x100 | In-Place | 12,755 | 5 | 544 |
| 2D_100x100 | Destination | 14,252 | 5 | 544 |
| 2D_1000x100 | In-Place | 149,015 | 5 | 544 |
| 2D_1000x100 | Destination | 233,877 | 5 | 544 |
| 3D_50x50x50 | In-Place | 193,553 | 5 | 544 |
| 3D_50x50x50 | Destination | 255,659 | 5 | 544 |


### Subtract Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 1,943 | 5 | 544 |
| 1K | Destination | 2,724 | 5 | 544 |
| 10K | In-Place | 13,948 | 5 | 544 |
| 10K | Destination | 22,461 | 5 | 544 |
| 100K | In-Place | 159,698 | 5 | 544 |
| 100K | Destination | 444,410 | 5 | 544 |
| 1M | In-Place | 2,297,976 | 5 | 544 |
| 1M | Destination | 4,786,965 | 5 | 544 |
| 2D_100x100 | In-Place | 14,046 | 5 | 544 |
| 2D_100x100 | Destination | 15,898 | 5 | 544 |
| 2D_1000x100 | In-Place | 203,904 | 5 | 544 |
| 2D_1000x100 | Destination | 182,050 | 5 | 544 |
| 3D_50x50x50 | In-Place | 429,394 | 5 | 544 |
| 3D_50x50x50 | Destination | 297,122 | 5 | 544 |


### Multiply Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 6,718 | 5 | 544 |
| 1K | Destination | 3,494 | 5 | 544 |
| 10K | In-Place | 74,050 | 5 | 544 |
| 10K | Destination | 36,668 | 5 | 544 |
| 100K | In-Place | 635,644 | 5 | 544 |
| 100K | Destination | 338,829 | 5 | 544 |
| 1M | In-Place | 4,516,358 | 5 | 544 |
| 1M | Destination | 4,141,826 | 5 | 544 |
| 2D_100x100 | In-Place | 55,684 | 5 | 544 |
| 2D_100x100 | Destination | 31,623 | 5 | 544 |
| 2D_1000x100 | In-Place | 549,175 | 5 | 544 |
| 2D_1000x100 | Destination | 384,203 | 5 | 544 |
| 3D_50x50x50 | In-Place | 695,896 | 5 | 544 |
| 3D_50x50x50 | Destination | 555,644 | 5 | 544 |


### Divide Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 10,517 | 8 | 568 |
| 1K | Destination | 3,446 | 8 | 568 |
| 10K | In-Place | 74,185 | 8 | 568 |
| 10K | Destination | 30,516 | 8 | 568 |
| 100K | In-Place | 728,626 | 8 | 568 |
| 100K | Destination | 305,112 | 8 | 568 |
| 1M | In-Place | 7,090,320 | 8 | 568 |
| 1M | Destination | 6,304,957 | 8 | 568 |
| 2D_100x100 | In-Place | 75,375 | 8 | 592 |
| 2D_100x100 | Destination | 27,082 | 8 | 592 |
| 2D_1000x100 | In-Place | 888,077 | 8 | 592 |
| 2D_1000x100 | Destination | 345,086 | 8 | 592 |
| 3D_50x50x50 | In-Place | 995,633 | 8 | 616 |
| 3D_50x50x50 | Destination | 1,361,648 | 8 | 616 |


## Scalar Operations

### ScalarMul Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 1,364 | 3 | 336 |
| 1K | Destination | 3,955 | 3 | 336 |
| 10K | In-Place | 11,362 | 3 | 336 |
| 10K | Destination | 27,988 | 3 | 336 |
| 100K | In-Place | 117,283 | 3 | 336 |
| 100K | Destination | 300,995 | 3 | 336 |
| 1M | In-Place | 1,619,691 | 3 | 336 |
| 1M | Destination | 3,742,713 | 3 | 336 |
| 2D_100x100 | In-Place | 13,166 | 3 | 336 |
| 2D_100x100 | Destination | 30,984 | 3 | 336 |
| 2D_1000x100 | In-Place | 104,689 | 3 | 336 |
| 2D_1000x100 | Destination | 331,114 | 3 | 336 |
| 3D_50x50x50 | In-Place | 138,637 | 3 | 336 |
| 3D_50x50x50 | Destination | 443,971 | 3 | 336 |


### AddScalar Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 5,386 | 3 | 336 |
| 1K | Destination | 3,465 | 3 | 336 |
| 10K | In-Place | 63,297 | 3 | 336 |
| 10K | Destination | 31,873 | 3 | 336 |
| 100K | In-Place | 705,500 | 3 | 336 |
| 100K | Destination | 471,532 | 3 | 336 |
| 1M | In-Place | 5,664,578 | 3 | 336 |
| 1M | Destination | 5,253,750 | 3 | 336 |
| 2D_100x100 | In-Place | 29,307 | 3 | 336 |
| 2D_100x100 | Destination | 27,481 | 3 | 336 |
| 2D_1000x100 | In-Place | 346,906 | 3 | 336 |
| 2D_1000x100 | Destination | 306,389 | 3 | 336 |
| 3D_50x50x50 | In-Place | 453,037 | 3 | 336 |
| 3D_50x50x50 | Destination | 491,007 | 3 | 336 |


## Unary Operations

### Square Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 3,132 | 3 | 336 |
| 1K | Destination | 3,697 | 3 | 336 |
| 10K | In-Place | 29,795 | 3 | 336 |
| 10K | Destination | 28,472 | 3 | 336 |
| 100K | In-Place | 337,480 | 3 | 336 |
| 100K | Destination | 343,113 | 3 | 336 |
| 1M | In-Place | 3,369,661 | 3 | 336 |
| 1M | Destination | 3,278,447 | 3 | 336 |
| 2D_100x100 | In-Place | 27,197 | 3 | 336 |
| 2D_100x100 | Destination | 50,138 | 3 | 336 |
| 2D_1000x100 | In-Place | 291,222 | 3 | 336 |
| 2D_1000x100 | Destination | 484,055 | 3 | 336 |
| 3D_50x50x50 | In-Place | 416,962 | 3 | 336 |
| 3D_50x50x50 | Destination | 639,139 | 3 | 336 |


### Sqrt Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 7,477 | 3 | 336 |
| 1K | Destination | 7,400 | 3 | 336 |
| 10K | In-Place | 65,773 | 3 | 336 |
| 10K | Destination | 49,539 | 3 | 336 |
| 100K | In-Place | 803,440 | 3 | 336 |
| 100K | Destination | 624,699 | 3 | 336 |
| 1M | In-Place | 6,411,332 | 3 | 336 |
| 1M | Destination | 7,606,662 | 3 | 336 |
| 2D_100x100 | In-Place | 63,516 | 3 | 336 |
| 2D_100x100 | Destination | 65,246 | 3 | 336 |
| 2D_1000x100 | In-Place | 622,677 | 3 | 336 |
| 2D_1000x100 | Destination | 562,399 | 3 | 336 |
| 3D_50x50x50 | In-Place | 771,698 | 3 | 336 |
| 3D_50x50x50 | Destination | 733,096 | 3 | 336 |


### Exp Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 3,471 | 3 | 336 |
| 1K | Destination | 20,880 | 3 | 336 |
| 10K | In-Place | 31,557 | 3 | 336 |
| 10K | Destination | 199,901 | 3 | 336 |
| 100K | In-Place | 360,145 | 3 | 336 |
| 100K | Destination | 2,059,813 | 3 | 336 |
| 1M | In-Place | 4,739,498 | 3 | 336 |
| 1M | Destination | 23,301,359 | 3 | 336 |
| 2D_100x100 | In-Place | 33,583 | 3 | 336 |
| 2D_100x100 | Destination | 238,449 | 3 | 336 |
| 2D_1000x100 | In-Place | 364,328 | 3 | 336 |
| 2D_1000x100 | Destination | 1,924,229 | 3 | 336 |
| 3D_50x50x50 | In-Place | 421,159 | 3 | 336 |
| 3D_50x50x50 | Destination | 3,127,067 | 3 | 336 |


### Log Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 4,091 | 3 | 336 |
| 1K | Destination | 41,096 | 3 | 336 |
| 10K | In-Place | 40,593 | 3 | 336 |
| 10K | Destination | 283,489 | 3 | 336 |
| 100K | In-Place | 429,186 | 3 | 336 |
| 100K | Destination | 1,983,756 | 3 | 336 |
| 1M | In-Place | 4,509,836 | 3 | 336 |
| 1M | Destination | 21,419,918 | 3 | 336 |
| 2D_100x100 | In-Place | 58,822 | 3 | 336 |
| 2D_100x100 | Destination | 249,476 | 3 | 336 |
| 2D_1000x100 | In-Place | 418,262 | 3 | 336 |
| 2D_1000x100 | Destination | 2,444,600 | 3 | 336 |
| 3D_50x50x50 | In-Place | 583,737 | 3 | 336 |
| 3D_50x50x50 | Destination | 3,466,434 | 3 | 336 |


### Pow Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 18,975 | 3 | 336 |
| 1K | Destination | 129,284 | 3 | 336 |
| 10K | In-Place | 216,458 | 3 | 336 |
| 10K | Destination | 1,071,039 | 3 | 336 |
| 100K | In-Place | 1,824,882 | 3 | 336 |
| 100K | Destination | 12,351,957 | 3 | 336 |
| 1M | In-Place | 26,529,740 | 3 | 336 |
| 1M | Destination | 114,099,499 | 3 | 336 |
| 2D_100x100 | In-Place | 134,656 | 3 | 336 |
| 2D_100x100 | Destination | 1,069,671 | 3 | 336 |
| 2D_1000x100 | In-Place | 1,211,678 | 3 | 336 |
| 2D_1000x100 | Destination | 10,651,329 | 3 | 336 |
| 3D_50x50x50 | In-Place | 1,578,221 | 3 | 336 |
| 3D_50x50x50 | Destination | 14,060,158 | 3 | 336 |


### Abs Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 3,123 | 3 | 336 |
| 1K | Destination | 3,373 | 3 | 336 |
| 10K | In-Place | 29,793 | 3 | 336 |
| 10K | Destination | 30,843 | 3 | 336 |
| 100K | In-Place | 355,523 | 3 | 336 |
| 100K | Destination | 327,037 | 3 | 336 |
| 1M | In-Place | 3,573,572 | 3 | 336 |
| 1M | Destination | 3,919,014 | 3 | 336 |
| 2D_100x100 | In-Place | 28,635 | 3 | 336 |
| 2D_100x100 | Destination | 29,613 | 3 | 336 |
| 2D_1000x100 | In-Place | 299,266 | 3 | 336 |
| 2D_1000x100 | Destination | 289,989 | 3 | 336 |
| 3D_50x50x50 | In-Place | 400,758 | 3 | 336 |
| 3D_50x50x50 | Destination | 703,887 | 3 | 336 |


### Sign Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 4,155 | 3 | 336 |
| 1K | Destination | 3,659 | 3 | 336 |
| 10K | In-Place | 35,011 | 3 | 336 |
| 10K | Destination | 31,481 | 3 | 336 |
| 100K | In-Place | 340,072 | 3 | 336 |
| 100K | Destination | 386,392 | 3 | 336 |
| 1M | In-Place | 3,184,632 | 3 | 336 |
| 1M | Destination | 3,335,200 | 3 | 336 |
| 2D_100x100 | In-Place | 53,600 | 3 | 336 |
| 2D_100x100 | Destination | 30,330 | 3 | 336 |
| 2D_1000x100 | In-Place | 324,666 | 3 | 336 |
| 2D_1000x100 | Destination | 340,660 | 3 | 336 |
| 3D_50x50x50 | In-Place | 415,960 | 3 | 336 |
| 3D_50x50x50 | Destination | 466,302 | 3 | 336 |


### Cos Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 25,191 | 3 | 336 |
| 1K | Destination | 29,562 | 3 | 336 |
| 10K | In-Place | 225,872 | 3 | 336 |
| 10K | Destination | 232,922 | 3 | 336 |
| 100K | In-Place | 2,285,105 | 3 | 336 |
| 100K | Destination | 2,459,826 | 3 | 336 |
| 1M | In-Place | 21,901,645 | 3 | 336 |
| 1M | Destination | 23,649,918 | 3 | 336 |
| 2D_100x100 | In-Place | 265,408 | 3 | 336 |
| 2D_100x100 | Destination | 224,325 | 3 | 336 |
| 2D_1000x100 | In-Place | 1,937,103 | 3 | 336 |
| 2D_1000x100 | Destination | 2,701,385 | 3 | 336 |
| 3D_50x50x50 | In-Place | 2,787,869 | 3 | 336 |
| 3D_50x50x50 | Destination | 4,231,467 | 3 | 336 |


### Sin Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 40,018 | 3 | 336 |
| 1K | Destination | 36,372 | 3 | 336 |
| 10K | In-Place | 346,523 | 3 | 336 |
| 10K | Destination | 262,655 | 3 | 336 |
| 100K | In-Place | 5,054,736 | 3 | 336 |
| 100K | Destination | 5,438,536 | 3 | 336 |
| 1M | In-Place | 27,142,997 | 3 | 336 |
| 1M | Destination | 20,541,126 | 3 | 336 |
| 2D_100x100 | In-Place | 233,889 | 3 | 336 |
| 2D_100x100 | Destination | 192,315 | 3 | 336 |
| 2D_1000x100 | In-Place | 2,687,274 | 3 | 336 |
| 2D_1000x100 | Destination | 2,060,612 | 3 | 336 |
| 3D_50x50x50 | In-Place | 2,471,734 | 3 | 336 |
| 3D_50x50x50 | Destination | 2,801,583 | 3 | 336 |


### Negative Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 1,561 | 3 | 336 |
| 1K | Destination | 1,912 | 3 | 336 |
| 10K | In-Place | 14,985 | 3 | 336 |
| 10K | Destination | 15,414 | 3 | 336 |
| 100K | In-Place | 100,942 | 3 | 336 |
| 100K | Destination | 247,540 | 3 | 336 |
| 1M | In-Place | 1,543,893 | 3 | 336 |
| 1M | Destination | 2,216,243 | 3 | 336 |
| 2D_100x100 | In-Place | 9,379 | 3 | 336 |
| 2D_100x100 | Destination | 11,957 | 3 | 336 |
| 2D_1000x100 | In-Place | 126,684 | 3 | 336 |
| 2D_1000x100 | Destination | 153,720 | 3 | 336 |
| 3D_50x50x50 | In-Place | 181,525 | 3 | 336 |
| 3D_50x50x50 | Destination | 198,136 | 3 | 336 |


## Activation Functions

### ReLU Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 2,185 | 3 | 240 |
| 1K | Destination | 1,850 | 2 | 160 |
| 10K | In-Place | 19,892 | 3 | 240 |
| 10K | Destination | 15,802 | 2 | 160 |
| 100K | In-Place | 225,592 | 3 | 240 |
| 100K | Destination | 161,357 | 2 | 160 |
| 1M | In-Place | 1,790,338 | 3 | 240 |
| 1M | Destination | 1,868,780 | 2 | 160 |
| 2D_100x100 | In-Place | 18,072 | 3 | 240 |
| 2D_100x100 | Destination | 16,084 | 2 | 160 |
| 2D_1000x100 | In-Place | 179,753 | 3 | 240 |
| 2D_1000x100 | Destination | 160,899 | 2 | 160 |
| 3D_50x50x50 | In-Place | 203,419 | 3 | 240 |
| 3D_50x50x50 | Destination | 251,457 | 2 | 160 |


### Sigmoid Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 21,857 | 3 | 240 |
| 1K | Destination | 30,540 | 2 | 160 |
| 10K | In-Place | 205,584 | 3 | 240 |
| 10K | Destination | 262,222 | 2 | 160 |
| 100K | In-Place | 1,934,728 | 3 | 240 |
| 100K | Destination | 3,076,567 | 2 | 160 |
| 1M | In-Place | 21,596,840 | 3 | 240 |
| 1M | Destination | 30,500,760 | 2 | 160 |
| 2D_100x100 | In-Place | 266,584 | 3 | 240 |
| 2D_100x100 | Destination | 286,282 | 2 | 160 |
| 2D_1000x100 | In-Place | 2,199,530 | 3 | 240 |
| 2D_1000x100 | Destination | 2,769,153 | 2 | 160 |
| 3D_50x50x50 | In-Place | 6,680,596 | 3 | 240 |
| 3D_50x50x50 | Destination | 3,345,544 | 2 | 160 |


### Tanh Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 21,171 | 3 | 240 |
| 1K | Destination | 63,337 | 2 | 160 |
| 10K | In-Place | 212,425 | 3 | 240 |
| 10K | Destination | 635,818 | 2 | 160 |
| 100K | In-Place | 1,967,596 | 3 | 240 |
| 100K | Destination | 6,494,757 | 2 | 160 |
| 1M | In-Place | 22,899,471 | 3 | 240 |
| 1M | Destination | 61,287,516 | 2 | 160 |
| 2D_100x100 | In-Place | 201,406 | 3 | 240 |
| 2D_100x100 | Destination | 640,821 | 2 | 160 |
| 2D_1000x100 | In-Place | 2,168,756 | 3 | 240 |
| 2D_1000x100 | Destination | 6,294,216 | 2 | 160 |
| 3D_50x50x50 | In-Place | 2,927,416 | 3 | 240 |
| 3D_50x50x50 | Destination | 7,987,475 | 2 | 160 |


## Utility Operations

### Fill Operation

| Size | Mode | Duration (ns/op) | Allocations | Memory (B/op) |
|------|------|-----------------|-------------|---------------|
| 1K | In-Place | 834 | 2 | 160 |
| 1K | Destination | 685 | 1 | 80 |
| 10K | In-Place | 6,932 | 2 | 160 |
| 10K | Destination | 5,004 | 1 | 80 |
| 100K | In-Place | 56,889 | 2 | 160 |
| 100K | Destination | 50,579 | 1 | 80 |
| 1M | In-Place | 953,525 | 2 | 160 |
| 1M | Destination | 886,774 | 1 | 80 |
| 2D_100x100 | In-Place | 5,339 | 2 | 160 |
| 2D_100x100 | Destination | 6,806 | 1 | 80 |
| 2D_1000x100 | In-Place | 54,887 | 2 | 160 |
| 2D_1000x100 | Destination | 54,780 | 1 | 80 |
| 3D_50x50x50 | In-Place | 76,122 | 2 | 160 |
| 3D_50x50x50 | Destination | 71,307 | 1 | 80 |

