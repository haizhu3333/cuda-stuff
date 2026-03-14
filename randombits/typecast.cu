#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include "utils.h"

__global__ void u8_to_f32_kernel1(float *out, uint8_t *in, int size) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {
        out[i] = static_cast<float>(in[i]);
    }
}

// This will have a problem if size isn't a multiple of 4.
__global__ void u8_to_f32_kernel2(float *out, uint8_t *in, int size) {
    uchar4 *in4 = reinterpret_cast<uchar4*>(in);
    float4 *out4 = reinterpret_cast<float4*>(out);
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size/4; i += gridDim.x * blockDim.x) {
        uchar4 val = in4[i];
        float4 fvec;
        fvec.x = static_cast<float>(val.x);
        fvec.y = static_cast<float>(val.y);
        fvec.z = static_cast<float>(val.z);
        fvec.w = static_cast<float>(val.w);
        out4[i] = fvec;
    }
}

// __global__ void u8_to_f32_kernel4(float *out, uint8_t *in, int size) {
//     int lane = threadIdx.x & 31;
//     int base = lane & ~3;
//     // Rotate the two lowest bits above next 3 bits
//     int reqLane = (lane & 3) << 3 | (lane >> 2);
//     for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += gridDim.x * blockDim.x) {

//         uchar4 val = in4[i];
//         float a0 = static_cast<float>(val.x);
//         float a1 = static_cast<float>(val.y);
//         float a2 = static_cast<float>(val.z);
//         float a3 = static_cast<float>(val.w);

//         float b0 = __shfl_sync(0xFFFFFFFF, a0, reqLane);
//         float b1 = __shfl_sync(0xFFFFFFFF, a1, reqLane);
//         float b2 = __shfl_sync(0xFFFFFFFF, a2, reqLane);
//         float b3 = __shfl_sync(0xFFFFFFFF, a3, reqLane);

//         // lane: 10 (0b01010)
//         // base: 8 (0b01000)
//         // reqLane: 18 (0b10010)
//         float c0 = __shfl_sync(0xFFFFFFFF, b0, base); // gets t8's b0
//     }
// }
constexpr int N = 10 * 1000 * 1000;
constexpr int ITERS = 1000;

void fill_u8(uint8_t *ptr) {
    for (int i = 0; i < N; i++) {
        ptr[i] = static_cast<uint8_t>(i % 100);
    }
}

void check_f32(float *ptr) {
    for (int i = 0; i < N; i++) {
        float v = ptr[i];
        float expected = static_cast<float>(i % 100);
        if (v != expected) {
            printf("Mismatch: element %d has value %f, should be %f\n", i, v, expected);
            // exit(1);
        }
    }
    printf("check succeeded\n");
}

int main() {
    auto h_in = new_host_array<uint8_t>(N);
    auto d_in = new_cuda_array<uint8_t>(N);
    auto h_out = new_host_array<float>(N);
    auto d_out = new_cuda_array<float>(N);

    fill_u8(h_in.get());
    CHECK(cudaMemcpy(d_in.get(), h_in.get(), N * sizeof(uint8_t), cudaMemcpyHostToDevice));

    for (int i = 0; i < ITERS; i++) {
        // 805.88us  800.13us  816.96us
        u8_to_f32_kernel1<<< 320, 1024 >>>(d_out.get(), d_in.get(), N);
    }
    CHECK(cudaMemcpy(h_out.get(), d_out.get(), N * sizeof(float), cudaMemcpyDeviceToHost));
    check_f32(h_out.get());

    for (int i = 0; i < ITERS; i++) {
        // 763.15us  746.57us  777.10us
        u8_to_f32_kernel2<<< 80, 1024 >>>(d_out.get(), d_in.get(), N);
    }
    CHECK(cudaMemcpy(h_out.get(), d_out.get(), N * sizeof(float), cudaMemcpyDeviceToHost));
    check_f32(h_out.get());
}
