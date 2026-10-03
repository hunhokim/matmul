## Build

Requires CMake 3.20+, a CUDA toolkit and host compiler with C++20 `std::source_location`
support (tested with CUDA 12.9 / GCC 13), and an NVIDIA GPU with a working driver.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Runtime experiments

The defaults are 100,000 elements, block sizes 32 through 1024 in steps of 32,
and 1,000 timed launches per configuration. No rebuild is needed to change them:

```bash
# Threads per block vs. latency at a fixed element count
./build/vector_add --n 100000 --threads 32,64,128,256,512,1024

# Element count vs. latency at a fixed block size
./build/vector_add --n 10,100,1000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000 --threads 256

# All combinations, with more launches per measurement and CSV output
mkdir -p results
./build/vector_add --n 10000,100000,1000000 --threads 64,128,256 --iterations 10000 --csv > results/experiment.csv

./build/vector_add --help
```

`--n` and `--threads` accept one positive integer or a comma-separated list.
`--iterations` accepts one positive integer. Block and grid sizes are checked
against the active GPU's limits. Each configuration gets one untimed warm-up
launch. Reported latency is the CUDA-event batch time divided by the number of
launches; it excludes allocation and memory copies and can include launch gaps.
Lists run in the supplied order, with element count as the outer loop.
The plotting script accepts the same runtime settings and can plot either axis.
When both settings are lists, each fixed value of the other setting gets its own curve.

## Timing plot

`./build/vector_add` prints a terminal chart; `--csv` prints machine-readable timings.
To plot the mean with sample-standard-deviation error bars over 30 repeated sweeps:

```bash
uv venv .venv
uv pip install --python .venv/bin/python matplotlib
# Threads/block vs. latency, holding n = 100000
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python plot_timings.py --axis threads --iterations 1000 --samples 30 --output-dir results/threads_sweep

# Element count vs. latency, holding threads/block = 256
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python plot_timings.py --axis n --iterations 1000 --samples 30 --output-dir results/elements_sweep
```

Each output directory contains raw `samples.csv`, `summary.csv`, and PNG/SVG
plots named `block_size_timings` or `element_count_timings`. Element-count plots
use a linear x-axis and default to 10, 100, and 1,000 elements, followed by
10,000 through 100,000 in steps of 10,000, with 256 threads per block. Override the inputs with `--n`.
The commands above use 30 batch samples per configuration and 1,000
launches per sample; the script defaults to 1,000 launches. Customize sweeps
with `--n`, `--threads`, `--iterations`, and `--samples` (`--repeats` is an alias).
Each sample times the entire batch with one CUDA event pair and divides by the
iteration count. Individual iterations are not timed.

Error bars show sample standard deviation between batch averages, not individual
kernel latencies or confidence intervals. Timings exclude memory copies but can
include gaps between launches. Sweeps use the supplied order, so clock or
temperature drift may affect comparisons.

Replot saved repeated samples without accessing the GPU (choose the matching axis):

```bash
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python plot_timings.py --axis n --input results/elements_sweep/samples.csv --output-dir results/elements_sweep
```

`--input` uses the configurations stored in the CSV; collection options such as
`--n`, `--threads`, and `--iterations` only apply when collecting new samples.

## IDE

The root `compile_commands.json` symlink provides compiler settings.
[`.clangd`](.clangd) filters NVCC-only flags for clangd; restart the language
server if stale errors remain.
