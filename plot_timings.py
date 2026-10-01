"""Measure vector_add configurations and plot latency against block size or element count."""

import argparse
import csv
import io
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
    # Duplicate configurations would otherwise count twice in each run.
    return ",".join(str(number) for number in dict.fromkeys(numbers))


def collect(binary, destination, n, threads, iterations, samples):
    command = [str(binary), "--csv", "--n", n, "--threads", threads,
               "--iterations", str(iterations)]
    # Discard one full run before collecting samples.
    subprocess.run(command, check=True, capture_output=True, text=True)
    with destination.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["sample", "n", "timing_iterations", "threads_per_block", "average_us"])
        for sample in range(1, samples + 1):
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            for row in csv.DictReader(io.StringIO(result.stdout)):
                writer.writerow([sample, row["n"], row["timing_iterations"],
                                 row["threads_per_block"], row["average_us"]])
            print(f"Collected sample {sample}/{samples}", flush=True)


def plot(source, output_dir, axis):
    groups = {}
    iteration_counts = set()
    with source.open(newline="") as data:
        for row in csv.DictReader(data):
            iteration_counts.add(int(row["timing_iterations"]))
            key = (int(row["n"]), int(row["threads_per_block"]))
            groups.setdefault(key, []).append(float(row["average_us"]))
    if len(iteration_counts) != 1 or not groups or any(len(v) < 2 for v in groups.values()):
        raise ValueError("Expected one iteration count and at least two samples per (n, threads) pair")
    iterations = iteration_counts.pop()
    rows = [(n, threads, iterations, statistics.mean(values), statistics.stdev(values), len(values))
            for (n, threads), values in sorted(groups.items())]
    with (output_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["n", "threads_per_block", "timing_iterations", "mean_us", "std_us", "samples"])
        writer.writerows(rows)
    counts = [row[5] for row in rows]
    sample_count = str(counts[0]) if len(set(counts)) == 1 else f"{min(counts)}–{max(counts)}"

    # Each curve holds the other independent variable fixed.
    x_index, series_index = (1, 0) if axis == "threads" else (0, 1)
    series = {}
    for row in rows:
        series.setdefault(row[series_index], []).append(row)
    fig, timing_ax = plt.subplots(figsize=(11, 5.5))
    for fixed, points in sorted(series.items()):
        points.sort(key=lambda row: row[x_index])
        label = f"n = {fixed:,}" if axis == "threads" else f"{fixed} threads/block"
        timing_ax.errorbar([row[x_index] for row in points], [row[3] for row in points],
                          yerr=[row[4] for row in points], fmt="o-", markersize=4,
                          capsize=3, label=label)
    timing_ax.set_ylabel("GPU time per launch (µs)")
    title = "threads per block" if axis == "threads" else "element count"
    timing_ax.set_title(f"Vector addition: latency vs. {title}\n"
                        f"{iterations:,} launches per batch; {sample_count} samples per configuration")
    timing_ax.legend(title="Mean ± 1 sample standard deviation")
    timing_ax.set_xlabel("Threads per block" if axis == "threads" else "Number of elements")
    if axis == "n":
        timing_ax.set_xlim(left=0)
        timing_ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    else:
        ticks = sorted({row[1] for row in rows})
        if len(ticks) <= 16:
            timing_ax.set_xticks(ticks)
    timing_ax.set_ylim(bottom=0)
    timing_ax.grid(alpha=0.25)
    fig.text(0.5, 0.015, "Error bars: sample standard deviation of batch averages. "
             "Copies excluded; launch gaps included.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    stem = "block_size_timings" if axis == "threads" else "element_count_timings"
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/vector_add"))
    parser.add_argument("--samples", "--repeats", type=positive_int, default=30,
                        help="Batch samples per configuration (default: 30)")
    parser.add_argument("--axis", choices=("threads", "n"), default="threads")
    parser.add_argument("--n", type=integer_list, help="Comma-separated element counts")
    parser.add_argument("--threads", type=integer_list, help="Comma-separated block sizes")
    parser.add_argument("--iterations", type=positive_int, default=1000)
    parser.add_argument("--input", type=Path, help="Replot saved timings without running the GPU")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2 for error bars")
    n = args.n or ("100000" if args.axis == "threads" else ",".join(map(str, [10, 100, 1000, *range(10000, 100001, 10000)])))
    threads = args.threads or (",".join(map(str, range(32, 1025, 32))) if args.axis == "threads" else "256")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.input or args.output_dir / "samples.csv"
    if args.input is None:
        collect(args.binary.resolve(), source, n, threads, args.iterations, args.samples)
    plot(source, args.output_dir, args.axis)
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
