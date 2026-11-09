# sycl-kernel-fusion
The software implements a zero-overhead compile-time kernel fusion framework for high-throughput 3D point cloud spatial preprocessing in SYCL.
By chaining pointwise spatial operators through template functor composition, the framework collapses multi-stage transformation and filtering pipelines into a single-pass hardware kernel.
This eliminates redundant global memory (VRAM) round-trips ($\Delta M = 2(K - 1)N \cdot \text{sizeof(point)}$) and mitigates kernel launch overheads on heterogeneous accelerators.

## Building
### Dependencies and environmental variables
To build and run this project, ensure the following dependencies are installed and properly configured:
* CMake (>= 3.28)
* Ninja (recommended build system)
* Google Benchmark library (optional, required for the benchmark suite)
* SYCL compiler
    * AdaptiveCpp is used in this project
    * AdaptiveCpp requires LLVM / Clang (version 21 recommended, version 22 is not fully supported yet by the AdaptiveCpp compiler version 21)

Depending on your LLVM and SYCL compiler installation paths, you may need to export the following environmental variables before building:
```bash
export CC=/usr/bin/clang-21
export CXX=/usr/bin/clang++-21
export LDFLAGS="-fuse-ld=lld"
export AR=/usr/bin/llvm-ar-21
export RANLIB=/usr/bin/llvm-ranlib-21
export CXXFLAGS="-stdlib=libstdc++"
```

### Building
1. Clone the repository: `git clone https://github.com/SavaLione/sycl-point-cloud`
2. Navigate to the project directory: `cd sycl-point-cloud`
3. Create and enter a build directory: `mkdir build && cd build`
4. Configure the project with CMake: `cmake -GNinja -DCMAKE_INSTALL_PREFIX=/root/adaptivecpp ..`
    * Where:
        * `/root/adaptivecpp` is the AdaptiveCpp compiler location on your system
        * `-GNinja` use the Ninja build system (you can omit it in order to use the default one)
5. Compile the project: `ninja` or `make`

## Usage
Run the compiled executable with default benchmarking parameters on synthetic data:
```sh
./sycl-kernel-fusion
```

Enumerate and display all available SYCL platforms and devices:
```sh
./sycl-kernel-fusion --device-list
```

Run a comparison between fused and unfused pipelines on 10000000 synthetic points with numerical verification:
```sh
./sycl-kernel-fusion -n 10000000 -m compare --translate 1.0,2.0,3.0 -r 500.0 --verify
```

Target a specific GPU device by name substring and execute over 20 iterations:
```sh
./sycl-kernel-fusion -d Arc -n 20000000 -m compare -i 20
```

Execute a fused single-pass pipeline on a real binary point cloud and export the result to Stanford PLY format:
```sh
./sycl-kernel-fusion -c input.bin -o output.ply -m fused --translate 0.0,0.0,-5.0 -s 0.0039215 -z -10.0,30.0
```

Execute an unfused multi-pass pipeline on an input dataset:
```sh
./sycl-kernel-fusion -c input.bin -o output.bin -m unfused --translate 5.0,-2.5,1.0 -r 1000.0
```

Run all Google Benchmarks:
```sh
./sycl-kernel-fusion-benchmarks
```

Benchmark only the comparative pipeline depth scaling (N*K grid):
```sh
./sycl-kernel-fusion-benchmarks --benchmark_filter="bm_sycl_.*_pipeline"
```

Benchmark a specific isolated filter:
```sh
./sycl-kernel-fusion-benchmarks --benchmark_filter=bm_sycl_sphere_crop
```

Benchmark all individual pointwise operators:
```sh
./sycl-kernel-fusion-benchmarks --benchmark_filter="bm_sycl_(translation|sphere_crop|intensity_scale|bounding_box_filter)"
```

## Licenses and Acknowledgements
This project is licensed under [The GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
See the [LICENSE](LICENSE) file for the full license text.

Copyright (C) 2026 Savelii Pototskii (savalione.com)

### Third-Party Libraries
This project incorporates code from several third-party libraries.
I am grateful to their developers and maintainers.
The full license texts for these libraries can be found in the `/licenses` directory.
* xgetopt - [zlib License](licenses/LICENSE-xgetopt.txt)
