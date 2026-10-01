#include <cuda_runtime.h>

#include <charconv>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <source_location>
#include <string>
#include <vector>

void cuda_check(
    cudaError_t error,
    const std::source_location location = std::source_location::current()) {
    if (error != cudaSuccess) {
        std::cerr << location.file_name() << ':' << location.line()
                  << ": " << cudaGetErrorString(error) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Average microseconds per launch, including any gaps between launches.
float benchmark_vector_add(const float* d_a, const float* d_b, float* d_c,
                           int n, int threads_per_block, int timing_iterations) {
    const int blocks = 1 + (n - 1) / threads_per_block;
    // Warm up before timing so first-launch setup is excluded.
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    cuda_check(cudaGetLastError());  // Check whether the launch was valid.

    // Launches are asynchronous: wait here and check for execution errors.
    cuda_check(cudaDeviceSynchronize());

    // One event pair measures the whole batch; no per-launch instrumentation.
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start));
    cuda_check(cudaEventCreate(&stop));
    cuda_check(cudaEventRecord(start));
    for (int iteration = 0; iteration < timing_iterations; ++iteration) {
        vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    }
    cuda_check(cudaGetLastError());
    cuda_check(cudaEventRecord(stop));
    cuda_check(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop));
    cuda_check(cudaEventDestroy(start));
    cuda_check(cudaEventDestroy(stop));
    return elapsed_ms * 1000.0f / timing_iterations;
}

struct TimingResult {
    int n;
    int threads_per_block;
    float average_us;
};

void print_timings(const std::vector<TimingResult>& results) {
    float max_us = 0.0f;
    for (const auto& result : results) {
        if (result.average_us > max_us) {
            max_us = result.average_us;
        }
    }

    // All bars start at zero and share a scale; the longest is 40 characters.
    std::cout << "    Elements   Threads/block   Time (us)   Average GPU time (shorter is faster)\n"
              << std::fixed << std::setprecision(3);
    for (const auto& result : results) {
        const int width = max_us > 0.0f
            ? static_cast<int>(40.0f * result.average_us / max_us + 0.5f) : 0;
        std::cout << std::setw(12) << result.n << "   "
                  << std::setw(13) << result.threads_per_block << "   "
                  << std::setw(9) << result.average_us << "   |"
                  << std::string(width, '#') << '\n';
    }
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--n N[,N...]] [--threads T[,T...]] [--iterations I] [--csv]\n"
              << "  --n           Element counts (default: 100000)\n"
              << "  --threads     Threads per block (default: 32,64,...,1024)\n"
              << "  --iterations  Timed launches per configuration (default: 1000)\n"
              << "  --csv         Print CSV instead of a terminal chart\n"
              << "  --help        Show this help\n"
              << "Lists run all combinations. Values must be positive integers.\n";
}

std::vector<int> parse_values(const std::string& value, const std::string& option) {
    std::vector<int> values;
    size_t start = 0;
    do {
        const size_t end = value.find(',', start);
        const size_t length = (end == std::string::npos ? value.size() : end) - start;
        const char* first = value.data() + start;
        int number = 0;
        const auto parsed = std::from_chars(first, first + length, number);
        if (parsed.ec != std::errc{} || parsed.ptr != first + length || number <= 0) {
            throw std::invalid_argument(option + " requires positive integers up to INT_MAX");
        }
        values.push_back(number);
        if (end == std::string::npos) break;
        start = end + 1;
    } while (true);
    return values;
}

int main(int argc, char** argv) {
    bool csv = false;
    std::vector<int> sizes{100000};
    std::vector<int> thread_counts;
    for (int threads = 32; threads <= 1024; threads += 32) {
        thread_counts.push_back(threads);
    }
    int timing_iterations = 1000;
    try {
        for (int arg = 1; arg < argc; ++arg) {
            const std::string option = argv[arg];
            if (option == "--help") {
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            }
            if (option == "--csv") {
                csv = true;
                continue;
            }
            if (option != "--n" && option != "--threads" && option != "--iterations") {
                throw std::invalid_argument("Unknown option: " + option);
            }
            if (++arg == argc) throw std::invalid_argument("Missing value for " + option);
            auto values = parse_values(argv[arg], option);
            if (option == "--n") sizes = values;
            else if (option == "--threads") thread_counts = values;
            else {
                if (values.size() != 1) {
                    throw std::invalid_argument("--iterations requires one integer");
                }
                timing_iterations = values.front();
            }
        }

        int device = 0;
        cuda_check(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device));
        cudaFuncAttributes attributes{};
        cuda_check(cudaFuncGetAttributes(&attributes, vector_add));
        for (int threads : thread_counts) {
            if (threads > properties.maxThreadsPerBlock ||
                threads > properties.maxThreadsDim[0] || threads > attributes.maxThreadsPerBlock) {
                throw std::invalid_argument("--threads exceeds the device/kernel limit: " +
                                            std::to_string(threads));
            }
            for (int n : sizes) {
                if (1 + (n - 1) / threads > properties.maxGridSize[0]) {
                    throw std::invalid_argument("Requested grid exceeds the device limit");
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::vector<TimingResult> results;
    for (int n : sizes) {
        if (static_cast<size_t>(n) > std::numeric_limits<size_t>::max() / sizeof(float)) {
            std::cerr << "Element count exceeds addressable allocation size\n";
            return EXIT_FAILURE;
        }
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);

        // Prepare input on the host (CPU). h_ and d_ are naming conventions.
        std::vector<float> h_a(n), h_b(n);
        for (int i = 0; i < n; ++i) {
            h_a[i] = static_cast<float>(i);
            h_b[i] = 2.0f;
        }

        // Allocate separate arrays on the device (GPU).
        float* d_a = nullptr;
        float* d_b = nullptr;
        float* d_c = nullptr;
        cuda_check(cudaMalloc(&d_a, bytes));
        cuda_check(cudaMalloc(&d_b, bytes));
        cuda_check(cudaMalloc(&d_c, bytes));

        // Copy inputs from CPU memory to GPU memory.
        cuda_check(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

        for (int threads_per_block : thread_counts) {
            const float average_us = benchmark_vector_add(
                d_a, d_b, d_c, n, threads_per_block, timing_iterations);
            results.push_back({n, threads_per_block, average_us});
        }

        // Release GPU memory. The host vectors clean themselves up.
        cuda_check(cudaFree(d_a));
        cuda_check(cudaFree(d_b));
        cuda_check(cudaFree(d_c));
    }

    if (csv) {
        std::cout << "n,timing_iterations,threads_per_block,average_us\n"
                  << std::setprecision(9);
        for (const auto& result : results) {
            std::cout << result.n << ',' << timing_iterations << ','
                      << result.threads_per_block << ',' << result.average_us << '\n';
        }
    } else {
        std::cout << timing_iterations
                  << " launches per configuration; excludes memory copies.\n";
        print_timings(results);
    }

    return EXIT_SUCCESS;
}
