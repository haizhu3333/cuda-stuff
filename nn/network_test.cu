#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <stddef.h>
#include "network.h"
#include "tensor.h"
#include "utils.h"

void fill(float *ptr, size_t size, int period) {
    for (size_t i = 0; i < size; i++) {
        ptr[i] = static_cast<float>((i + 1) % period);
    }
}

int main() {
    size_t A = 167, B = 214, C = 83;

    Matrix h_input(A, B);
    Matrix h_weights(B, C);
    Vector h_biases(C);
    Matrix h_output(A, C);
    CUDA_CHECK(cudaMallocHost(&h_input.ptr, h_input.allocBytes()));
    CUDA_CHECK(cudaMallocHost(&h_weights.ptr, h_weights.allocBytes()));
    CUDA_CHECK(cudaMallocHost(&h_biases.ptr, h_biases.allocBytes()));
    CUDA_CHECK(cudaMallocHost(&h_output.ptr, h_output.allocBytes()));
    fill(h_input.ptr, h_input.size(), 7);
    fill(h_weights.ptr, h_weights.size(), 11);
    fill(h_biases.ptr, h_biases.size(), 13);

    Linear linear(B, C);
    CUDA_CHECK(cudaMemcpy2D(
        linear.weights.ptr, linear.weights.pitch, h_weights.ptr, h_weights.pitch,
        h_weights.allocWidth(), h_weights.allocHeight(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        linear.biases.ptr, h_biases.ptr, h_biases.sizeBytes(), cudaMemcpyHostToDevice));

    Matrix d_input(A, B);
    Matrix d_output(A, C);
    CUDA_CHECK(cudaMallocPitch(
        &d_input.ptr, &d_input.pitch, d_input.allocWidth(), d_input.allocHeight()));
    CUDA_CHECK(cudaMallocPitch(
        &d_output.ptr, &d_output.pitch, d_output.allocWidth(), d_output.allocHeight()));

    std::cout << "launching" << std::endl;
    CUDA_CHECK(cudaMemcpy2D(
        d_input.ptr, d_input.pitch, h_input.ptr, h_input.pitch,
        h_input.allocWidth(), h_input.allocHeight(), cudaMemcpyHostToDevice));
    linear.forward(d_output, d_input);
    CUDA_CHECK(cudaMemcpy2D(
        h_output.ptr, h_output.pitch, d_output.ptr, d_output.pitch,
        h_output.allocWidth(), h_output.allocHeight(), cudaMemcpyDeviceToHost));
    std::cout << "returned" << std::endl;

    std::cout << std::fixed << std::setprecision(1);
    for (size_t i = 0; i < h_output.shape[0]; i++) {
        for (size_t j = 0; j < h_output.shape[1]; j++) {
            std::cout << h_output(i, j) << "\t";
        }
        std::cout << std::endl;
    }

    CUDA_CHECK(cudaFreeHost(h_input.ptr));
    CUDA_CHECK(cudaFreeHost(h_weights.ptr));
    CUDA_CHECK(cudaFreeHost(h_biases.ptr));
    CUDA_CHECK(cudaFreeHost(h_output.ptr));
    CUDA_CHECK(cudaFree(d_input.ptr));
    CUDA_CHECK(cudaFree(d_output.ptr));
}
