# CRC32C Performance Benchmark Suite

A comprehensive benchmarking suite for comparing different CRC32C implementations using 
SIMD optimizations including Intel SSE4.2, PCLMUL, and Intel AVX-512 instructions.

## Overview

This repository contains optimized CRC32C implementations and benchmarking tools to 
evaluate performance across different algorithms and buffer sizes. 
The implementations include:

- **Scalar**: Basic implementation using Intel CRC32 instructions
- **SSE4.2**: Optimized implementation using SSE4.2 and PCLMUL
- **SSE4.2 Corsix**: Alternative SSE4.2 implementation with different optimization approach
- **AVX-512**: High-performance implementation using AVX-512 and VPCLMULQDQ instructions

## Features

- Multiple CRC32C algorithm implementations
- Flexible command-line interface for selective benchmarking
- Google Benchmark integration for performance measurement
- Google Test integration for correctness validation
- Support for custom buffer sizes and algorithm selection
- Compatible with Intel VTune profiling tools

## Building

### Prerequisites

- GCC with support for SSE4.2, PCLMUL, and AVX-512 instructions
- CMake (optional, for build system integration)
- Google Benchmark library (for benchmarking)
- Google Test library (for testing)

#### Install Prerequisites:

> Instructions assume Debian variant of Linux. Tested on Ubuntu 24.04

1. Check for system support of CPU instructions:
    ```bash
    lscpu | grep -E 'sse4_2|pclmul|avx512'
    ```
    * You should see:
        * `sse4_2`
        * `pclmulqdq` (often shown as pclmul)
        * Some `avx512*` flags (e.g., `avx512f`, `avx512bw`, `avx512dq`, `avx512vl`) for AVX-512
    > If your CPU doesn’t show a given feature, you can still compile with those flags for cross-targeting, but binaries won't run on this machine.

2. Install necessary build packages:
    ```bash
    sudo apt update
    sudo apt install -y build-essential cmake
    ```

3. Install Google Test (gtest):
    ```console
    # Remove package version
    sudo apt remove libbenchmark-dev libbenchmark1

    # Build from source
    git clone https://github.com/google/benchmark.git
    cd benchmark
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    ```

### Compilation

For all scenarios, set the build flags:

```bash
export CFLAGS="-O2 -msse4.2 -mpclmul -mavx512f"
export CXXFLAGS="-O2 -msse4.2 -mpclmul -mavx512f"
```

#### Build both benchamrk and test executable

```bash
make 
```

#### For benchmark only:

```bash
make bench
```

#### For test suite only:

```bash
make test
```

#### Clean build artifacts:

```bash
make clean
```

### Usage

```bash
./bench --help
Usage: ./bench [options]
  --algorithm <name>          Run a single algorithm
  --algorithms a,b,c          Run a comma-separated list of algorithms
  --sizes n1,n2,n3            Buffer sizes (override defaults: 64,128,256,512,1024,2048)
  --all                       Run all algorithms (default if none specified)
  --list                      List supported algorithms
  --help                      Show this help
Algorithms: scalar, corsix, sse42, avx512
```

#### Sample baseline output

```bash
./bench
2025-10-23T10:21:02-07:00
Running ./bench
Run on (224 X 3800 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x112)
  L1 Instruction 32 KiB (x112)
  L2 Unified 2048 KiB (x112)
  L3 Unified 107520 KiB (x2)
Load Average: 0.13, 0.08, 0.06
***WARNING*** ASLR is enabled, the results may have unreproducible noise in them.
--------------------------------------------------------------------------------
Benchmark                      Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------
avx512/size=64/64           5.13 ns         5.13 ns    121172242 bytes_per_second=11.6243Gi/s
avx512/size=128/128         5.08 ns         5.08 ns    137868962 bytes_per_second=23.4629Gi/s
avx512/size=256/256         5.22 ns         5.22 ns    132470175 bytes_per_second=45.6422Gi/s
avx512/size=512/512         5.70 ns         5.70 ns    122836475 bytes_per_second=83.6697Gi/s
avx512/size=1024/1024       10.0 ns         10.0 ns     69647357 bytes_per_second=94.9705Gi/s
avx512/size=2048/2048       18.8 ns         18.8 ns     37388778 bytes_per_second=101.397Gi/s
corsix/size=64/64           2.93 ns         2.93 ns    237847987 bytes_per_second=20.3131Gi/s
corsix/size=128/128         3.75 ns         3.75 ns    186458636 bytes_per_second=31.7584Gi/s
corsix/size=256/256         7.93 ns         7.92 ns     88300456 bytes_per_second=30.0855Gi/s
corsix/size=512/512         17.0 ns         17.0 ns     41072623 bytes_per_second=27.9835Gi/s
corsix/size=1024/1024       34.0 ns         34.0 ns     20432472 bytes_per_second=28.0269Gi/s
corsix/size=2048/2048       68.6 ns         68.6 ns     10204961 bytes_per_second=27.8097Gi/s
scalar/size=64/64           3.70 ns         3.70 ns    189899894 bytes_per_second=16.117Gi/s
scalar/size=128/128         7.92 ns         7.92 ns     88364665 bytes_per_second=15.0513Gi/s
scalar/size=256/256         12.8 ns         12.8 ns     55177178 bytes_per_second=18.6821Gi/s
scalar/size=512/512         34.0 ns         34.0 ns     20573045 bytes_per_second=14.0175Gi/s
scalar/size=1024/1024       94.4 ns         94.4 ns      7413039 bytes_per_second=10.0989Gi/s
scalar/size=2048/2048        234 ns          234 ns      2986738 bytes_per_second=8.13636Gi/s
sse42/size=64/64            4.56 ns         4.56 ns    147230776 bytes_per_second=13.0686Gi/s
sse42/size=128/128          5.07 ns         5.07 ns    138085088 bytes_per_second=23.5121Gi/s
sse42/size=256/256          9.27 ns         9.27 ns     75546078 bytes_per_second=25.7286Gi/s
sse42/size=512/512          18.1 ns         18.1 ns     38642018 bytes_per_second=26.319Gi/s
sse42/size=1024/1024        36.0 ns         36.0 ns     19504194 bytes_per_second=26.4819Gi/s
sse42/size=2048/2048        70.9 ns         70.9 ns      9882542 bytes_per_second=26.9162Gi/s
```

#### Run specific algorithm

```bash
./bench --algorithm scalar
```

#### Run multiple algorithms

```bash
./bench --algorithms scalar,sse42,avx512
```

#### Custom buffer sizes

```bash
./bench --sizes 64,256,1024,4096
```

#### Run specific algorithm with custom buffer sizes

```bash
./bench --algorithm avx512 --sizes 1024,2048,4096
```

## Google Benchmark Integration

The benchamrk uses Google Benchmark for precise performance measurement. Additional
Google Benchmark options can be passed:

```bash
# Run with specific number of iterations
./bench --benchmark_repetitions=10

# Filter specific benchmarks
./bench --benchmark_filter="avx512.*"

# Output in different formats
./bench --benchmark_format=json --benchmark_out=results.json
```

## Testing

Run the test suite to verify algorithm correctness:

```bash
./test
```

The tests verify that all implementations produce identical results across different
 buffer sizes and data patterns.

## Algorithm Details

### Scalar Implementation
* Uses Intel CRC32 instructions (`_mm_crc32_u64`, `_mm_crc32_u32`, `_mm_crc32_u8`)
* Processes 8 bytes at a time when possible
* Falls back to smaller chunks for remaining bytes
* Target: `sse4.2`

### SSE4.2 Implementation

* Uses carry-less multiplication (PCLMUL) for parallel processing
* Processes 64-byte blocks in parallel
* Implements folding and Barrett reduction
* Target: `sse4.2`, `pclmul`

### SSE4.2 Corsix Implementation

* Alternative SSE4.2 approach with different constants and folding strategy
* Optimized for different workload characteristics
* Processes 64-byte and 16-byte blocks
* Target: `sse4.2`, `pclmul`

### AVX-512 Implementation

* Uses 512-bit wide SIMD operations
* Processes 256-byte blocks in parallel
* Leverages `_mm512_ternarylogic_epi64` for efficient XOR operations
* Uses VPCLMULQDQ instructions for carry-less multiplication
* Target: `avx512vl`, `vpclmulqdq`

### Performance Characteristics

Expected Performance Scaling
* **Scalar:** Baseline performance, ~1-2 GB/s
* **SSE4.2:** 2-4x improvement over scalar
* **AVX-512:** 4-8x improvement over scalar (on supported hardware)

#### Buffer Size Impact

* Small buffers (< 64 bytes): Scalar may be competitive due to setup overhead
* Medium buffers (64-1024 bytes): SSE4.2 implementations show best efficiency
* Large buffers (> 1024 bytes): AVX-512 demonstrates maximum throughput

## Hardware Requirements

### Minimum Requirements

* Intel processor with SSE4.2 support (Core i7/i5 2nd gen+, Xeon 5500+)
* PCLMUL instruction support

### Optimal Performance

* Intel 4th Generation Xeon Scalable processor or newer (for Intel AVX-512 VPCLMULQDQ)
* Intel 3rd Generation Xeon Scalable processor or newer (for improved Intel AVX-512 performance)

#### L3 Cache Considerations

The benchmark's performance can be significantly affected by L3 cache size, especially for larger buffer tests:
* **< 8MB L3:** May see performance degradation with buffers > 1KB
* **8-16MB L3:** Good performance up to 2-4KB buffers (covers default test range)
* **16-32MB L3:** Excellent performance up to 8-16KB buffers
* **32MB+ L3:** No cache pressure for typical benchmark scenarios

You can check your L3 cache size with:

```bash
lscpu | grep "L3 cache"
# or
cat /proc/cpuinfo | grep "cache size" | head -1
```

## Integration with Profiling Tools

### Intel VTune Integration

The benchmark is compatible with Intel VTune for detailed performance analysis:

```bash
# Collect performance data
vtune -collect hotspots -- ./bench --algorithm avx512 --sizes 2048

# Collect microarchitecture data
vtune -collect uarch-exploration -- ./bench --algorithm sse42
```

### Emon Integration

For automated performance data collection, the benchmark works with emon:

```bash
emon -collect-edp -f results.dat ./bench --algorithm scalar --sizes 1024
```

## Troubleshooting

### Common Issues

* "Illegal instruction" error: Your CPU doesn't support the required instruction set
    * Try running with `--algorithm scalar` first
    * Check CPU capabilities: 
        ```
        cat /proc/cpuinfo | grep flags
        ```
* Poor AVX-512 performance:
    * Some processors throttle frequency when using AVX-512
    * Check thermal throttling: `turbostat`
    * Consider using SSE4.2 implementations for sustained workloads
* Benchmark noise warnings:
    * ASLR (Address Space Layout Randomization) can cause measurement noise
    * For consistent results, disable temporarily: `setarch $(uname -m) -R ./bench`
    * Or run multiple iterations: `./bench --benchmark_repetitions=10`

## Performance Tuning

### System Configuration

```bash
# Set performance governor
sudo cpupower frequency-set -g performance

# Disable CPU frequency scaling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set process priority
nice -n -20 ./bench
```

### Memory Considerations

* Ensure sufficient memory bandwidth for large buffer tests
* Consider NUMA topology for multi-socket systems
* Monitor memory usage during large buffer benchmarks

## Contributing

### Adding New Algorithms

* Implement the algorithm function with appropriate target attributes
* Add benchmark wrapper function
* Update algorithm list in `default_algorithms()`
* Add to benchmark registration map
* Include tests for correctness verification

### Code Style

* Use Intel intrinsics for SIMD operations
* Include appropriate target attributes for function specialization
* Maintain consistent error handling and return codes
* Add comprehensive comments for complex algorithms

# References

* [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
* CRC32C specification: RFC 3720
* [Intel OneAPI (VTune, EMON)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-download.html?operatingsystem=linux&linux-install-type=offline
)
* [EMON user guide](https://www.intel.com/content/www/us/en/content-details/686077/emon-user-s-guide.html)
* [Google Benchmark](https://github.com/google/benchmark)
