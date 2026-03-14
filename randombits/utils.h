#pragma once
#include <stdio.h>
#include <cuda.h>
#include <memory>

#define CHECK(expr) cuda_check((expr), __FILE__, __LINE__)

inline void cuda_check(cudaError_t result, const char *file, int line) {
  if (result != cudaSuccess) {
    const char *message = cudaGetErrorString(result);
    fprintf(stderr, "Failure at %s:%d: %s\n", file, line, message);
    exit(-1);
  }
}

template <typename T>
struct CudaDeleter {
  void operator()(T *ptr) {
    CHECK(cudaFree(ptr));
  }
};

template <typename T>
struct CudaHostDeleter {
    void operator()(T *ptr) {
        CHECK(cudaFreeHost(ptr));
    }
};

template <typename T>
std::unique_ptr<T[], CudaDeleter<T>> new_cuda_array(size_t n) {
    T *ptr;
    CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    return std::unique_ptr<T[], CudaDeleter<T>>(ptr);
}

template <typename T>
std::unique_ptr<T[], CudaHostDeleter<T>> new_host_array(size_t n) {
    T *ptr;
    CHECK(cudaMallocHost(&ptr, n * sizeof(T)));
    return std::unique_ptr<T[], CudaHostDeleter<T>>(ptr);
}

__host__ __device__ inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}
