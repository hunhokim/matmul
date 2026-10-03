"""Shared measurement, CSV, plotting, and CLI helpers for vector_add experiments."""

import argparse
import csv
import math
import statistics
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def positive_int(value):
    number = int(value)
    if not 0 < number <= 2147483647:
        raise argparse.ArgumentTypeError("expected an integer between 1 and INT_MAX")
    return number


def integer_list(value):
    try:
        numbers = [positive_int(part) for part in value.split(",")]
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    return list(dict.fromkeys(numbers))


def collect(binary, destination, n, threads, samples):
    with destination.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["sample", "n", "timing_iterations", "threads_per_block", "average_us", "cache_mode"])
        for sample in range(1, samples + 1):
            for size in n:
                for block in threads:
                    command = [str(binary), str(size), str(block)]
                    timing = float(subprocess.check_output(command, text=True))
                    writer.writerow([sample, size, 1, block, timing, "eviction"])
            print(f"Collected sample {sample}/{samples}", flush=True)


def plot(source, output_dir, axis, peak_bandwidth=112.0):
    groups = {}
    iteration_counts = set()
    with source.open(newline="") as data:
        for row in csv.DictReader(data):
            iteration_counts.add(int(row["timing_iterations"]))
            key = (int(row["n"]), int(row["threads_per_block"]))
            timing = float(row["average_us"])
            if not math.isfinite(timing) or timing <= 0:
                raise ValueError("Timing must be finite and positive to calculate bandwidth")
            groups.setdefault(key, []).append(timing)
    if len(iteration_counts) != 1 or not groups:
        raise ValueError("Expected one iteration count and at least one sample per (n, threads) pair")
    iterations = iteration_counts.pop()
    rows = []
    for (n, threads), values in sorted(groups.items()):
        # Two float32 reads and one float32 write; decimal GB/s.
        bandwidths = [12 * n / (time_us * 1000) for time_us in values]
        rows.append((n, threads, iterations, statistics.mean(values),
                     statistics.stdev(values) if len(values) > 1 else None, len(values),
                     statistics.mean(bandwidths),
                     statistics.stdev(bandwidths) if len(values) > 1 else None))
    with (output_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["n", "threads_per_block", "timing_iterations", "mean_us", "std_us", "samples",
                         "mean_bandwidth_gbps", "std_bandwidth_gbps"])
        writer.writerows(rows)
    for n, threads, _, mean, deviation, count, bandwidth, bandwidth_std in rows:
        std = f"{deviation:.6f} µs" if deviation is not None else "N/A"
        bw_std = f"{bandwidth_std:.3f} GB/s" if bandwidth_std is not None else "N/A"
        print(f"n={n:,}, threads/block={threads}: average={mean:.6f} µs, "
              f"sample std={std}; bandwidth={bandwidth:.3f} GB/s, "
              f"sample std={bw_std} ({count} samples)")

    plot_metric(rows, output_dir, axis, iterations, 3, "latency",
                "GPU time per launch (µs)", "timings")
    plot_metric(rows, output_dir, axis, iterations, 6, "effective memory bandwidth",
                "Effective memory bandwidth (GB/s)", "bandwidth", peak_bandwidth)


def plot_metric(rows, output_dir, axis, iterations, mean_index, metric, ylabel, suffix,
                peak_bandwidth=None):
    # Each curve holds the other independent variable fixed.
    x_index, series_index = (1, 0) if axis == "threads" else (0, 1)
    series = {}
    for row in rows:
        series.setdefault(row[series_index], []).append(
            (row[x_index], row[mean_index], row[mean_index + 1]))
    fig, timing_ax = plt.subplots(figsize=(11, 5.5))
    for fixed, points in sorted(series.items()):
        x, means, deviations = zip(*sorted(points))
        label = f"n = {fixed:,}" if axis == "threads" else f"{fixed} threads/block"
        line, = timing_ax.plot(x, means, "o-", markersize=4, label=label)
        error_points = [(a, b, d) for a, b, d in zip(x, means, deviations) if d is not None]
        if error_points:
            error_x, error_means, error_deviations = zip(*error_points)
            timing_ax.errorbar(error_x, error_means, yerr=error_deviations, fmt="none",
                              color=line.get_color(), capsize=3)
    if peak_bandwidth is not None:
        timing_ax.axhline(peak_bandwidth, color="tab:red", linestyle="--", linewidth=1.5,
                          label=f"Theoretical VRAM bandwidth: {peak_bandwidth:g} GB/s")
    timing_ax.set_ylabel(ylabel)
    title = "threads per block" if axis == "threads" else "element count"
    timing_ax.set_title(f"Vector addition: {metric} vs. {title}\n"
                        f"{iterations:,} {'launch' if iterations == 1 else 'launches'} per sample")
    has_deviations = any(row[mean_index + 1] is not None for row in rows)
    timing_ax.legend(title="Error bars: ±1 sample standard deviation" if has_deviations else None)
    timing_ax.set_xlabel("Threads per block" if axis == "threads" else "Number of elements")
    if axis == "n":
        timing_ax.set_xscale("log", base=2)
    else:
        ticks = sorted({row[1] for row in rows})
        if len(ticks) <= 16:
            timing_ax.set_xticks(ticks)
    if axis == "n" and suffix == "timings":
        timing_ax.set_yscale("log")
    else:
        timing_ax.set_ylim(bottom=0)
    timing_ax.grid(alpha=0.25)
    fig.tight_layout()
    stem = "block_size" if axis == "threads" else "element_count"
    fig.savefig(output_dir / f"{stem}_{suffix}.png", dpi=180)
    plt.close(fig)


def argument_parser(description, output_dir):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--binary", type=Path, default=Path("build/vector_add"))
    parser.add_argument("--samples", "--repeats", type=positive_int, default=10,
                        help="Independent program runs per configuration (default: 10)")
    parser.add_argument("--input", type=Path, help="Replot saved timings without running the GPU")
    parser.add_argument("--peak-bandwidth", type=float, default=112.0,
                        help="Theoretical VRAM bandwidth in GB/s (default: 112 for GTX 1050 Ti)")
    parser.add_argument("--output-dir", type=Path, default=Path(output_dir))
    return parser


def run(parser, args, n, threads, axis):
    if not math.isfinite(args.peak_bandwidth) or args.peak_bandwidth <= 0:
        parser.error("--peak-bandwidth must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.input or args.output_dir / "samples.csv"
    if args.input is None:
        collect(args.binary.resolve(), source, n, threads, args.samples)
    plot(source, args.output_dir, axis, args.peak_bandwidth)
    print(f"Saved plots to {args.output_dir}")
