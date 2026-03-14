#include <cuda.h>
#include <sstream>
#include "utils.h"

CudaException::CudaException(cudaError_t error, const char *file, int line) :
    error(error), file(file), line(line)
{}

const char *CudaException::what() const noexcept {
    std::stringstream ss;
    ss << "Failure at " << file << ", line " << line << ": " << cudaGetErrorString(error);
    return ss.str().c_str();
}

SizeException::SizeException(size_t lhs, size_t rhs, const char *file, int line) :
    lhs(lhs), rhs(rhs), file(file), line(line)
{}

const char *SizeException::what() const noexcept {
    std::stringstream ss;
    ss << "Size mismatch at " << file << ", line " << line << ": " << lhs << " != " << rhs;
    return ss.str().c_str();
}
