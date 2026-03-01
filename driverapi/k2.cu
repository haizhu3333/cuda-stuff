#include <stdio.h>

extern "C" __global__ void test3() {
    printf("test3: (%d,%d,%d).(%d,%d,%d)\n",
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, threadIdx.z);
}
