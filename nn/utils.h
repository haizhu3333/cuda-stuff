#pragma once
#include <cuda.h>
#include <stdexcept>
#include <sstream>

#define CUDA_CHECK(expr) cudaCheck((expr), __FILE__, __LINE__)
#define SIZE_CHECK(lhs, rhs) sizeCheck((lhs), (rhs), __FILE__, __LINE__)

class CudaException : public std::exception {
    cudaError_t error;
    const char *file;
    int line;

public:
    CudaException(cudaError_t error, const char *file, int line);
    virtual const char *what() const noexcept override;
};

class SizeException : public std::exception {
    size_t lhs;
    size_t rhs;
    const char *file;
    int line;

public:
    SizeException(size_t lhs, size_t rhs, const char *file, int line);
    virtual const char *what() const noexcept override;
};

inline void cudaCheck(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        throw CudaException(result, file, line);
    }
}

inline void sizeCheck(size_t lhs, size_t rhs, const char *file, int line) {
    if (lhs != rhs) {
        throw SizeException(lhs, rhs, file, line);
    }
}

__host__ __device__ inline int ceilDiv(int x, int y) {
    return (x + y - 1) / y;
}
