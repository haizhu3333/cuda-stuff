#include <stdio.h>
#include <cuda.h>
#include "utils.h"

void printDevices(void) {
  int nDevices;
  CHECK(cudaGetDeviceCount(&nDevices));
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp dp;
    CHECK(cudaGetDeviceProperties(&dp, i));
    printf("Device %d:\n", i);
    printf("  Name: %s\n", dp.name);
    printf("  Capability: %d.%d\n", dp.major, dp.minor);
    printf("  Global Memory: %ldM\n", dp.totalGlobalMem/1024/1024);
    printf("  Memory clock rate: %d\n", dp.memoryClockRate);
    printf("  Memory bus: %d\n", dp.memoryBusWidth);
    printf("  # of SM = %d\n", dp.multiProcessorCount);
    printf("  Blocks/SM: %d\n", dp.maxBlocksPerMultiProcessor);
    printf("  Threads/SM = %d\n", dp.maxThreadsPerMultiProcessor);
    printf("  SMem/SM = %ldK\n", dp.sharedMemPerMultiprocessor / 1024);
    printf("  Regs/SM = %d\n", dp.regsPerMultiprocessor);
    printf("  Max Threads/Block: %d\n", dp.maxThreadsPerBlock);
    printf("  SMem/Block: %ldK\n", dp.sharedMemPerBlock/1024);
    printf("  Regs/Block = %d\n", dp.regsPerBlock);
    printf("  warpSize: %d\n", dp.warpSize);
    printf("  Max blockDim = %d, %d, %d\n",
        dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
    printf("  Max gridDim = %d, %d, %d\n",
        dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);
    printf("  managedMemory = %d\n", dp.managedMemory);
    printf("  concurrentManagedAccess = %d\n", dp.concurrentManagedAccess);
    printf("  pageableMemoryAccess = %d\n", dp.pageableMemoryAccess);
    printf("  pageableMemoryAccessUsesHostPageTables = %d\n",
        dp.pageableMemoryAccessUsesHostPageTables);
    printf("  cooperativeLaunch = %d\n", dp.cooperativeLaunch);
    printf("\n");
  }
}

int main(int argc, char **argv) {
  printDevices();
  return 0;
}
