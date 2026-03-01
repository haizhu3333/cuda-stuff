#include <cuda.h>
#include <stdio.h>
#include "utils.h"

// length of in = N
// length of out = N - K + 1
__global__ void conv1d(float *out, float *in, int Nin, int K) {
    // length of sd = blockDim.x + K - 1.
    extern __shared__ float sd[];
    int Nout = Nin - K + 1;
    int Nsd = blockDim.x + K - 1;

    for (int base = blockIdx.x * blockDim.x;
            base < Nout;
            base += gridDim.x * blockDim.x) {
        for (int sid = threadIdx.x; sid < Nsd && base + sid < Nin; sid += blockDim.x) {
            // printf("<%d,%d> | sd[%d] = in[%d]\n",
            //     blockIdx.x, threadIdx.x, sid, base + sid);
            sd[sid] = in[base + sid];
        }
        __syncthreads();

        int tid = base + threadIdx.x;
        if (tid < Nout) {
            // printf("<%d,%d> | out[%d] = sum sd[%d-%d]\n",
            //     blockIdx.x, threadIdx.x, tid, threadIdx.x, threadIdx.x + K - 1);
            float sum = 0.f;
            for (int i = 0; i < K; i++) {
                sum += sd[threadIdx.x + i];
            }
            out[tid] = sum;
        }
        __syncthreads();
    }
}

int main(int argc, char *argv[]) {
    int Nin = 50000;
    int K = 7;
    int blockSize = 512;
    int gridSize = 20;
    int Nout = Nin - K + 1;
    int Nsd = blockSize + K - 1;

    auto h_in = new_host_array<float>(Nin);
    auto h_out = new_host_array<float>(Nout);
    auto d_in = new_cuda_array<float>(Nin);
    auto d_out = new_cuda_array<float>(Nout);

    for (int i = 0; i < Nin; i++) {
        h_in[i] = static_cast<float>(i);
    }
    CHECK(cudaMemcpy(d_in.get(), h_in.get(), Nin * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < 10; i++) {
        conv1d<<< gridSize, blockSize, Nsd * sizeof(float) >>>(
            d_out.get(), d_in.get(), Nin, K);
    }
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(h_out.get(), d_out.get(), Nout * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < Nout; i++) {
        float value = h_out[i];
        float expected = static_cast<float>(i * K + (K * (K - 1) / 2));
        if (value != expected) {
            printf("Mismatch: out[%d] = %f, expected %f\n", i, value, expected);
        }
    }
    printf("Completed\n");
    return 0;
}
