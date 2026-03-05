#pragma once
#include <cuda.h>
#include <stdexcept>
#include <sstream>

#define CHECK(expr) cuda_check((expr), __FILE__, __LINE__)

inline void cuda_check(cudaError_t result, const char *file, int line) {
  if (result != cudaSuccess) {
    std::stringstream ss;
    ss << "Failure at " << file << ", line " << line << ": " << cudaGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}

