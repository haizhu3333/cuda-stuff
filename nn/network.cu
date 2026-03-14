#include <assert.h>
#include <cuda.h>
#include <stdexcept>
#include "network.h"
#include "tensor.h"
#include "utils.h"

constexpr int TILE_SIZE = 16;
constexpr dim3 GRID_SIZE {10, 8};

/**
Computes out = matmul(A, B) + broadcast(C).

Dimensions:
    out: Y, K
    A  : Y, K
    B  : K, X
    C  :    X
*/
__global__ void matMulAndAddKernel(Matrix out, Matrix inA, Matrix inB, Vector inC) {
    __shared__ float tA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tB[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tC[TILE_SIZE];

    int y = threadIdx.y;
    int x = threadIdx.x;
    int ySize = out.shape[0];
    int kSize = inB.shape[0];
    int xSize = out.shape[1];
    int yTileCount = ceilDiv(ySize, TILE_SIZE);
    int kTileCount = ceilDiv(kSize, TILE_SIZE);
    int xTileCount = ceilDiv(xSize, TILE_SIZE);

    for (int xTile = blockIdx.x; xTile < xTileCount; xTile += gridDim.x) {
        // Iterate over X tiles first since they share the same tile from C
        int xGlobal = xTile * TILE_SIZE + x;
        if (y == 0) tC[x] = NAN;
        if (y == 0 && xGlobal < xSize) {
            tC[x] = inC(xGlobal);
        }
        __syncthreads();

        for (int yTile = blockIdx.y; yTile < yTileCount; yTile += gridDim.y) {
            int yGlobal = yTile * TILE_SIZE + y;
            float sum = tC[x];
            for (int kTile = 0; kTile < kTileCount; kTile++) {
                tA[y][x] = NAN;
                tB[y][x] = NAN;
                // Load kth tile from input A/B
                int ky = kTile * TILE_SIZE + y;
                int kx = kTile * TILE_SIZE + x;
                if (yGlobal < ySize && kx < kSize) {
                    tA[y][x] = inA(yGlobal, kx);
                }
                if (ky < kSize && xGlobal < xSize) {
                    tB[y][x] = inB(ky, xGlobal);
                }
                __syncthreads();

                // Multiply and add
                if (yGlobal < ySize && xGlobal < xSize) {
                    for (int k = 0; k < TILE_SIZE && kTile * TILE_SIZE + k < kSize; k++) {
                        sum += tA[y][k] * tB[k][x];
                    }
                }
                __syncthreads();
            }
            if (yGlobal < ySize && xGlobal < xSize) {
                out(yGlobal, xGlobal) = sum;
            }
            __syncthreads();
        }
    }
}

Linear::Linear(int in, int out) :
    weights(in, out), biases(out)
{
    CUDA_CHECK(cudaMallocPitch(
        &weights.ptr, &weights.pitch, weights.allocWidth(), weights.allocHeight()));
    CUDA_CHECK(cudaMalloc(&biases.ptr, biases.allocBytes()));
}

Linear::~Linear() {
    CUDA_CHECK(cudaFree(weights.ptr));
    CUDA_CHECK(cudaFree(biases.ptr));
}

void Linear::forward(const Matrix &output, const Matrix &input) {
    SIZE_CHECK(input.shape[0], output.shape[0]);
    SIZE_CHECK(input.shape[1], weights.shape[0]);
    SIZE_CHECK(output.shape[1], weights.shape[1]);

    matMulAndAddKernel<<<GRID_SIZE, dim3(TILE_SIZE, TILE_SIZE)>>>(output, input, weights, biases);
    CUDA_CHECK(cudaGetLastError());
}
