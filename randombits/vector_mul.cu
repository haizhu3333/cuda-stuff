#include <cstdio>
#include <cuda.h>
#include <numeric>
#include <string>
#include <vector>
#include "utils.h"

__global__ void element_mul(int *r, int *a, int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        r[i] = a[i] * b[i];
    }
}

constexpr int N = 1 << 20;

int main(int argc, char **argv) {
    int nBlocks, nThreads;
    if (argc == 3) {
        nBlocks = std::stoi(argv[1]);
        nThreads = std::stoi(argv[2]);
    } else if (argc == 1) {
        CHECK(cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, element_mul, 0, 0));
    } else {
        fprintf(stderr, "Usage: %s <gridDim> <blockDim>\n", argv[0]);
        return 1;
    }
    printf("Using %d blocks, %d threads/block\n", nBlocks, nThreads);

    std::vector<int> a(N);
    std::vector<int> b(N);
    std::vector<int> r(N);
    for (int i = 0; i < N; i++) {
        a[i] = (i % 5 == 0) ? 1 : 0;
        b[i] = (i % 8 == 0) ? 1 : 0;
    }
    auto da = new_cuda_array<int>(N);
    auto db = new_cuda_array<int>(N);
    auto dr = new_cuda_array<int>(N);
    CHECK(cudaMemcpy(da.get(), a.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db.get(), b.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    for (int i = 0; i < 10; i++) {
        element_mul<<< nBlocks, nThreads >>>(dr.get(), da.get(), db.get(), N);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaMemcpy(r.data(), dr.get(), N * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Sum: %d\n", std::accumulate(r.begin(), r.end(), 0));
    return 0;
}
