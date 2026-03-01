#include <stdio.h>
#include <cuda.h>
#include "utils.h"

__global__ void go() {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    int s = __shfl_xor_sync(0xFFFFFFFFu, lane, 1 << warp);
    printf("thread %d: s = %d\n", threadIdx.x, s);
}

int main(int argc, char **argv) {
  go<<< 1, 64 >>>();
  CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  return 0;
}
