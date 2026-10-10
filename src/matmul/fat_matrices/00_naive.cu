#include <cuda_runtime.h>

#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <source_location>
#include <string_view>

void cuda_check(cudaError_t error,
                const std::source_location location = std::source_location::current()) {
    if (error != cudaSuccess) {
        std::cerr << location.file_name() << ':' << location.line() << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

unsigned int block_dimension(std::string_view text) {
    unsigned int value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
        std::cerr << "Block dimensions must be positive integers.\n";
        std::exit(EXIT_FAILURE);
    }
    return value;
}

__global__ void matmul(const float *a, const float *b, float *c, const int M, const int N,
                       const int K) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= N || y >= M)
        return;

    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += a[y * K + i] * b[i * N + x];
    }
    c[y * N + x] = sum;
}

int main(int argc, char **argv) {
    const bool help = argc == 2 && std::string_view(argv[1]) == "--help";
    if (help || (argc != 1 && argc != 3)) {
        std::cout << "Usage: " << argv[0] << " [block_x block_y]\n"
                  << "Default block: 32 32\n";
        return help ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    const unsigned int block_x = argc == 3 ? block_dimension(argv[1]) : 32;
    const unsigned int block_y = argc == 3 ? block_dimension(argv[2]) : 32;
    constexpr auto M = 4096, N = 4096, K = 4096;
    constexpr size_t a_bytes = M * K * sizeof(float);
    constexpr size_t b_bytes = K * N * sizeof(float);
    constexpr size_t c_bytes = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cuda_check(cudaMalloc(&d_a, a_bytes));
    cuda_check(cudaMalloc(&d_b, b_bytes));
    cuda_check(cudaMalloc(&d_c, c_bytes));
    cuda_check(cudaMemset(d_a, 0, a_bytes));
    cuda_check(cudaMemset(d_b, 0, b_bytes));

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start));
    cuda_check(cudaEventCreate(&stop));

    const dim3 threads_per_block(block_x, block_y);
    const dim3 num_blocks(1 + (N - 1) / block_x, 1 + (M - 1) / block_y);

    // Warm up the kernel before measuring a single launch.
    matmul<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, M, N, K);
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());

    cuda_check(cudaEventRecord(start));
    matmul<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, M, N, K);
    cuda_check(cudaGetLastError());
    cuda_check(cudaEventRecord(stop));
    cuda_check(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop));
    // Count each multiply-add as two floating-point operations.
    const double tflops = (2.0 * M * N * K) / (elapsed_ms * 1.0e9);
    std::cout << "M=" << M << " N=" << N << " K=" << K << " block_x=" << block_x
              << " block_y=" << block_y << '\n';
    std::cout << std::fixed << std::setprecision(3) << elapsed_ms << " (ms) elapsed\n";
    std::cout << std::setprecision(6) << tflops << " TFLOPS\n";

    cuda_check(cudaEventDestroy(start));
    cuda_check(cudaEventDestroy(stop));
    cuda_check(cudaFree(d_a));
    cuda_check(cudaFree(d_b));
    cuda_check(cudaFree(d_c));

    return EXIT_SUCCESS;
}
