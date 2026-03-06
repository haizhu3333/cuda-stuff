#pragma once
#include <cuda.h>
#include <stdexcept>
#include <sstream>

#define CUDA_CHECK(expr) cudaCheck((expr), __FILE__, __LINE__)

class CudaException : public std::exception {
    cudaError_t error;
    const char *file;
    int line;

public:
    CudaException(cudaError_t error, const char *file, int line);
    virtual const char *what() const noexcept override;
};

inline void cudaCheck(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        throw CudaException(result, file, line);
    }
}
