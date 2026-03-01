#pragma once
#include <stdio.h>
#include <cuda.h>
#include <GL/gl.h>

#define CUDA_CHECK(expr) cuda_check((expr), __FILE__, __LINE__)

inline void cuda_check(cudaError_t result, const char *file, int line) {
  if (result != cudaSuccess) {
    const char *message = cudaGetErrorString(result);
    fprintf(stderr, "Failure at %s:%d: %s\n", file, line, message);
    exit(-1);
  }
}

template <typename T>
struct Board {
    T *ptr;
    size_t pitch;

    __host__ __device__ const T &operator()(int x, int y) const {
        T *row = reinterpret_cast<T *>(reinterpret_cast<char *>(ptr) + y * pitch);
        return row[x];
    }

    __host__ __device__ T &operator()(int x, int y) {
        T *row = reinterpret_cast<T *>(reinterpret_cast<char *>(ptr) + y * pitch);
        return row[x];
    }

};

struct Size {
    int width;
    int height;

    template <typename T>
    __host__ __device__ size_t rowBytes() const {
        return sizeof(T) * width;
    }

    template <typename T>
    __host__ __device__ size_t totalBytes() const {
        return rowBytes<T>() * height;
    }

    __host__ __device__ int totalElems() const {
        return width * height;
    }

    __host__ __device__ bool inBounds(int x, int y) const {
        return x >= 0 && y >= 0 && x < width && y < height;
    }
};

__host__ __device__ inline int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}
