#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

void check(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char *errorName, *message;
        cuGetErrorName(result, &errorName);
        cuGetErrorString(result, &message);
        fprintf(stderr, "%s: %s\n", errorName, message);
        exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <kernel>\n", argv[0]);
        exit(1);
    }
    char *name = argv[1];

    CUdevice device;
    CUcontext ctx;
    CUlibrary library;
    CUkernel kernel;

    check(cuInit(0));
    check(cuDeviceGet(&device, 0));
    check(cuDevicePrimaryCtxRetain(&ctx, device));
    check(cuCtxPushCurrent(ctx));
    check(cuLibraryLoadFromFile(&library, "bin/klib.a", NULL, NULL, 0, NULL, NULL, 0));
    check(cuLibraryGetKernel(&kernel, library, name));
    CUlaunchConfig config = {
        .gridDimX = 2,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = 3,
        .blockDimY = 2,
        .blockDimZ = 1,
        .sharedMemBytes = 0,
        .hStream = NULL,
        .attrs = NULL,
        .numAttrs = 0,
    };
    check(cuLaunchKernelEx(&config, (CUfunction)kernel, NULL, NULL));
    check(cuCtxSynchronize());
    check(cuLibraryUnload(library));
    check(cuDevicePrimaryCtxRelease(device));
    return 0;
}
