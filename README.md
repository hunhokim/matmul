## Build

Requires CMake 3.20+, a CUDA toolkit and host compiler with C++20 `std::source_location`
support (tested with CUDA 12.9 / GCC 13), and an NVIDIA GPU with a working driver.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

Pass the element count and threads per block as positive integers:

```bash
./build/vector_add 100000 256
```

The program warms up vector addition, reads a separate scratch buffer sized at
four times the GPU's L2 cache, then measures one vector-add launch with CUDA events
and checks the result. It prints the elapsed microseconds to stdout; errors go to
stderr. Allocation, memory transfers, warm-up, cache eviction, and result checking
are outside the timed region. Scratch-buffer eviction is a practical heuristic,
not a guaranteed hardware cache flush. Small kernels remain sensitive to event
timing resolution and scheduling overhead.
Invalid launch configurations are reported by CUDA.
Eviction reads the scratch buffer without modifying it to avoid leaving dirty
scratch cache lines that could add writeback traffic during the measured launch.

## Statistics and plots

The two plotting scripts share measurement and plotting helpers in
`scripts/timing_common.py`. Each runs the program once per configuration per sample and
calculates means and, when multiple samples are collected, sample standard deviations:

```bash
export UV_CACHE_DIR=.cache/uv
uv sync --locked
MPLCONFIGDIR=.cache/matplotlib uv run --locked python scripts/plot_block_sizes.py --output-dir results/threads_sweep
MPLCONFIGDIR=.cache/matplotlib uv run --locked python scripts/plot_element_counts.py --output-dir results/elements_sweep
```

Python is pinned in `.python-version`; `uv.lock` pins all Python dependencies.
Run `uv sync --locked` after cloning to recreate the environment. To intentionally
update dependencies, run `uv lock --upgrade` and commit the updated lockfile.

The default is 10 samples per configuration. The block-size sweep uses
64, 128, 256, 512, and 1024 threads per block at 100,000 elements. The element-count
sweep uses 128 threads per block and powers of two from 2^10 (1,024) through
2^25 (33,554,432) elements. Element-count plots use a base-2 logarithmic x-axis;
the latency plot also uses a logarithmic y-axis (log-log).

Each output directory contains `samples.csv`, `summary.csv`, and separate latency
and effective memory bandwidth PNG plots (`*_timings.png` and `*_bandwidth.png`).
Use `--samples` to set the number of independent program runs. The scripts print
the average and sample standard deviation for each configuration.
For compatibility with saved CSVs, new measurements retain `timing_iterations`
(always 1) and `average_us` (the single measured launch time). Older batch timing
CSVs can still be replotted. Each sample starts a fresh process and CUDA context.
Error bars show sample standard deviation across runs, not confidence intervals.
With `--samples 1`,
error bars are omitted and standard deviation fields in `summary.csv` are blank.

Effective memory bandwidth counts two float32 reads and one float32 write per
element: `bandwidth_GB/s = 12 * n / (elapsed_us * 1000)` (decimal GB/s).
The scripts calculate bandwidth for each sample, then report its mean and sample
standard deviation in `mean_bandwidth_gbps` and `std_bandwidth_gbps` in `summary.csv`.
Bandwidth error bars use these transformed samples. This measures useful bytes
per elapsed time; it does not measure physical DRAM traffic or guarantee peak
DRAM bandwidth, particularly for small inputs or cached historical measurements.

Bandwidth plots include a dashed theoretical VRAM bandwidth line at 112 GB/s for
this machine's GTX 1050 Ti ([manufacturer specifications](https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/GeForce%20Graphics/English/PNY-NVIDIA-GeForce-GTX-1050Ti-4GB.pdf)).
Use `--peak-bandwidth <GB/s>` for a different GPU or memory clock configuration;
the reference value is configured, not automatically detected.

New sample CSVs record `cache_mode` as `eviction`. Output writes can still remain
buffered in L2 during a short launch; the large-input plateau is more useful for
estimating sustained DRAM bandwidth.

Replot saved samples without accessing the GPU:

```bash
MPLCONFIGDIR=.cache/matplotlib uv run --locked python scripts/plot_element_counts.py --input results/elements_sweep/samples.csv --output-dir results/elements_sweep
```

## IDE

The root `compile_commands.json` symlink provides compiler settings.
[`.clangd`](.clangd) filters NVCC-only flags for clangd; restart the language
server if stale errors remain.
