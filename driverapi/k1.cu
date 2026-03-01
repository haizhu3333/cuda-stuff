#include <stdio.h>

extern "C" __global__ void test1() {
    printf("test1: (%d,%d,%d).(%d,%d,%d)\n",
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, threadIdx.z);
}

extern "C" __global__ void test2() {
    printf("test2: (%d,%d,%d).(%d,%d,%d)\n",
        blockIdx.x, blockIdx.y, blockIdx.z,
        threadIdx.x, threadIdx.y, threadIdx.z);
}
