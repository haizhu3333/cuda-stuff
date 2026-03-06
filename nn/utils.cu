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
