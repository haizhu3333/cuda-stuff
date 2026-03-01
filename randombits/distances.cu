#include <stdio.h>
#include <cuda.h>
#include "utils.h"

struct Point {
    float x;
    float y;
};

constexpr int TILE_SIZE = 4;

// grid: (N/TILE)x(N/TILE), block: TILExTILE
__global__ void populate_distances(int n, const Point *points, float *distances) {
    __shared__ Point s_points1[TILE_SIZE];
    __shared__ Point s_points2[TILE_SIZE];

    int px = blockIdx.x * TILE_SIZE + threadIdx.x;
    int py = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (threadIdx.y == 0 && px < n) {
        s_points1[threadIdx.x] = points[px];
    }
    if (threadIdx.x == 0 && py < n) {
        s_points2[threadIdx.y] = points[py];
    }
    __syncthreads();

    if (px < n && py < n) {
        float x1 = s_points1[threadIdx.x].x;
        float y1 = s_points1[threadIdx.x].y;
        float x2 = s_points2[threadIdx.y].x;
        float y2 = s_points2[threadIdx.y].y;
        distances[px * n + py] = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    }
}

int main(int argc, char **argv) {
    constexpr int N = 10;
    Point points[N];
    for (int i = 0; i < N; i++) {
        points[i].x = (float)i;
        points[i].y = (float)(i % 2);
    }
    auto d_points = new_cuda_array<Point>(N);
    auto d_distances = new_cuda_array<float>(N * N);
    CHECK(cudaMemcpy(d_points.get(), points, sizeof points, cudaMemcpyHostToDevice));

    int n_blocks = ceil_div(N, TILE_SIZE);
    dim3 grid_dim(n_blocks, n_blocks, 1);
    dim3 block_dim(TILE_SIZE, TILE_SIZE, 1);
    populate_distances<<< grid_dim, block_dim >>>(N, d_points.get(), d_distances.get());
    CHECK(cudaGetLastError());

    float distances[N][N];
    CHECK(cudaMemcpy(distances, d_distances.get(), sizeof distances, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.0f\t", distances[i][j] * distances[i][j]);
        }
        printf("\n");
    }
}
