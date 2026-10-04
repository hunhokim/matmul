#include <cuda_runtime.h>

#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <source_location>
#include <string_view>
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

// cudaMalloc provides the alignment required by float4. Each full group
// uses two 16-byte loads and one 16-byte store; the last thread handles a tail.
__global__ void vector_add_float4(const float* a, const float* b, float* c, int n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t groups = static_cast<size_t>(n) / 4;
    if (i < groups) {
        const float4 av = reinterpret_cast<const float4*>(a)[i];
        const float4 bv = reinterpret_cast<const float4*>(b)[i];
        reinterpret_cast<float4*>(c)[i] = make_float4(
            av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w);
    } else if (i == groups) {
        for (int component = 0; component < 3; ++component) {
            if (component < n % 4) {
                const size_t j = groups * 4 + component;
                c[j] = a[j] + b[j];
            }
        }
    }
}

// Touch a separate working set larger than L2 to evict the vector data.
__global__ void evict_l2(const volatile unsigned int* scratch, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) (void)scratch[i];
}

int main(int argc, char** argv) {
    int n = 0, threads = 0;
    auto parse = [](const char* argument, int& value) {
        const std::string_view text(argument);
        const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
        return result.ec == std::errc{} && result.ptr == text.data() + text.size() && value > 0;
    };
    if (argc != 3 || !parse(argv[1], n) || !parse(argv[2], threads)) {
        std::cerr << "Usage: " << argv[0] << " <elements> <threads_per_block>\n"
                  << "Arguments must be positive integers up to INT_MAX.\n";
        return EXIT_FAILURE;
    }

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> a(n), b(n, 2.0f), c(n);
    for (int i = 0; i < n; ++i) a[i] = static_cast<float>(i);

    float *d_a, *d_b, *d_c;
    cuda_check(cudaMalloc(&d_a, bytes));
    cuda_check(cudaMalloc(&d_b, bytes));
    cuda_check(cudaMalloc(&d_c, bytes));
    cuda_check(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    int device = 0;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device));
    cuda_check(cudaGetDeviceProperties(&properties, device));
    if (properties.l2CacheSize <= 0) {
        std::cerr << "Cannot determine L2 size for cache eviction.\n";
        return EXIT_FAILURE;
    }
    const size_t scratch_bytes = 4 * static_cast<size_t>(properties.l2CacheSize);
    const size_t scratch_count = scratch_bytes / sizeof(unsigned int);
    unsigned int* scratch;
    cuda_check(cudaMalloc(&scratch, scratch_bytes));
    cuda_check(cudaMemset(scratch, 0, scratch_bytes));

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start));
    cuda_check(cudaEventCreate(&stop));
    const size_t work_items = (static_cast<size_t>(n) + 3) / 4;
    const size_t blocks = 1 + (work_items - 1) / threads;
    // Warm up kernel loading and execution, then evict outside the timed region.
    vector_add_float4<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());
    evict_l2<<<(scratch_count + 255) / 256, 256>>>(scratch, scratch_count);
    cuda_check(cudaGetLastError());
    cuda_check(cudaEventRecord(start));
    vector_add_float4<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cuda_check(cudaGetLastError());
    cuda_check(cudaEventRecord(stop));
    cuda_check(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop));

    cuda_check(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (c[i] != a[i] + b[i]) {
            std::cerr << "Incorrect result at element " << i << '\n';
            correct = false;
            break;
        }
    }
    cuda_check(cudaEventDestroy(start));
    cuda_check(cudaEventDestroy(stop));
    cuda_check(cudaFree(scratch));
    cuda_check(cudaFree(d_a));
    cuda_check(cudaFree(d_b));
    cuda_check(cudaFree(d_c));
    if (!correct) return EXIT_FAILURE;

    // One measured launch in microseconds; diagnostics go to stderr.
    std::cout << std::fixed << std::setprecision(3)
              << elapsed_ms * 1000.0f << '\n';
    return EXIT_SUCCESS;
}
