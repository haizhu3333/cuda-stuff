#include <algorithm>
#include <cuda.h>
#include <random>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "board_manager.h"
#include "utils.h"

constexpr int PAD = 2;
constexpr int TILE_SIZE = 32;
constexpr int GRID_SIZE = 10;
constexpr int EXTRA_TILE_COL_PADDING = 0;
constexpr int WRITE_TEX_GRID_SIZE = 10;
constexpr int WRITE_TEX_BLOCK_SIZE = 512;

#define COLORS_VALUES_ \
    {0,     0,   0, 255},\
    {192, 192, 192, 255},\
    {255,   0,   0, 255},\
    {0,   255,   0, 255},\
    {0    , 0, 255, 255}

const Pixel H_COLORS[] = {COLORS_VALUES_};
__constant__ Pixel D_COLORS[] = {COLORS_VALUES_};

template <typename Function>
__device__ Cell traverseNeighbours(const Board<Cell> padded, int x, int y, Function function) {
    function(padded(x    , y    ));
    function(padded(x + 1, y    ));
    function(padded(x + 2, y    ));
    function(padded(x    , y + 1));
    function(padded(x + 2, y + 1));
    function(padded(x    , y + 2));
    function(padded(x + 1, y + 2));
    function(padded(x + 2, y + 2));
    return   padded(x + 1, y + 1);
}

__device__ Cell nextState(const Board<Cell> padded) {
    uint8_t pairMask = 0;
    uint8_t mask = 0;
    uint8_t count = 0;
    Cell self = traverseNeighbours(padded, threadIdx.x, threadIdx.y, [&](Cell value) {
        pairMask |= (mask & value);
        mask |= value;
        count += (value != 0);
    });

    if (self == 0 && count == 3) {
        // 3 neighbours produces new cell
        // If the 3 neighbours have 2 or 3 in common, pairMask is the common element.
        // Otherwise, pairMask = 0 and there are 3 distinct values, so return 0xF ^ mask.
        return pairMask | ((__popc(mask) == 3) * (0xF ^ mask));
    } else {
        // If neighbour count is not 2 or 3, cell should be dead. (count & 2) >> 1 = 0.
        // Otherwise, count is 2 (dead -> dead, alive -> alive),
        // or count is 3 AND self is alive.
        return self * ((count & 2) >> 1);
    }
}

/*
board: the input state. Contains total size information.
output: the output state to be populated. Size should be the same as input board. Pitch might not.
paddedTile: contains pointer to shared memory. The tile size is assumed to be
            (blockDim.x + 2) x (blockDim.y + 2).
*/

__device__ void loadPadded(
    Board<Cell> paddedOut, const Board<Cell> board, Size size, int tileOffsetX, int tileOffsetY
) {
    int id = threadIdx.x + blockDim.x * threadIdx.y;
    int stride = blockDim.x * blockDim.y;
    int padCols = blockDim.x + PAD;
    int padRows = blockDim.y + PAD;

    for (; id < padCols * padRows; id += stride) {
        int padX = id / padCols;
        int padY = id % padCols;
        int x = tileOffsetX + padX - 1;
        int y = tileOffsetY + padY - 1;
        Cell value = 0;
        if (size.inBounds(x, y)) {
            value = board(x, y);
        }
        paddedOut(padX, padY) = value;
    }
}

__device__ void nextStateTile(
    Board<Cell> output, Board<Cell> paddedTile, const Board<Cell> board, Size size,
    int tileOffsetX, int tileOffsetY
) {
    loadPadded(paddedTile, board, size, tileOffsetX, tileOffsetY);
    __syncthreads();

    int x = tileOffsetX + threadIdx.x;
    int y = tileOffsetY + threadIdx.y;
    if (size.inBounds(x, y)) {
        output(x, y) = nextState(paddedTile);
    }
    __syncthreads();
}

__global__ void nextStateKernel(
    Board<Cell> output, size_t tilePitch, Board<Cell> board, Size size
) {
    extern __shared__ Cell tilePtr[];
    Board<Cell> paddedTile = {tilePtr, tilePitch};
    int tileCols = ceil_div(size.width, blockDim.x);
    int tileRows = ceil_div(size.height, blockDim.y);
    for (int b = blockIdx.x; b < tileCols * tileRows; b += gridDim.x) {
        int tileOffsetX = b / tileCols * blockDim.x;
        int tileOffsetY = b % tileCols * blockDim.y;
        nextStateTile(output, paddedTile, board, size, tileOffsetX, tileOffsetY);
    }
}

__global__ void writeTextureKernel(Board<Pixel> tex, Board<Cell> board, Size size) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
        i < size.totalElems();
        i += gridDim.x * blockDim.x
    ) {
        int x = i % size.width;
        int y = i / size.width;
        int colorIndex = __ffs(board(x, y));
        tex(x, y) = D_COLORS[colorIndex];
    }
}

BoardManager::BoardManager(Size size) : size(size) {
    size_t wCell = size.rowBytes<Cell>();
    size_t wPixel = size.rowBytes<Pixel>();
    size_t h = size.height;
    CUDA_CHECK(cudaMallocPitch(&d_curr.ptr, &d_curr.pitch, wCell, h));
    CUDA_CHECK(cudaMallocPitch(&d_next.ptr, &d_next.pitch, wCell, h));
    CUDA_CHECK(cudaMallocPitch(&d_tex.ptr, &d_tex.pitch, wPixel, h));
}

BoardManager::~BoardManager() {
    CUDA_CHECK(cudaFree(d_curr.ptr));
    CUDA_CHECK(cudaFree(d_next.ptr));
    CUDA_CHECK(cudaFree(d_tex.ptr));
}

void BoardManager::initialize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 24);
    size_t w = size.rowBytes<Cell>();
    size_t h = size.height;
    Board<Cell> board = {nullptr, w};
    CUDA_CHECK(cudaMallocHost(&board.ptr, w * h));
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            int n = dist(gen);
            if (n >= 4) {
                board(x, y) = 0;
            } else {
                board(x, y) = 1 << n;
            }
        }
    }
    CUDA_CHECK(cudaMemcpy2D(
        d_curr.ptr, d_curr.pitch, board.ptr, w, w, h, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFreeHost(board.ptr));
}

void BoardManager::step() {
    dim3 blockSize {TILE_SIZE, TILE_SIZE};
    size_t d_tilePitch = sizeof(Cell) * (TILE_SIZE + PAD + EXTRA_TILE_COL_PADDING);
    size_t smemSize = (TILE_SIZE + PAD) * d_tilePitch;
    nextStateKernel<<< GRID_SIZE, blockSize, smemSize >>>(d_next, d_tilePitch, d_curr, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::swap(d_curr, d_next);
}

void BoardManager::writeTexture(Board<Pixel> h_tex) {
    writeTextureKernel<<< WRITE_TEX_GRID_SIZE, WRITE_TEX_BLOCK_SIZE >>>(d_tex, d_curr, size);
    CUDA_CHECK(cudaGetLastError());
    int w = size.rowBytes<Pixel>();
    int h = size.height;
    CUDA_CHECK(cudaMemcpy2D(
        h_tex.ptr, h_tex.pitch, d_tex.ptr, d_tex.pitch, w, h, cudaMemcpyDeviceToHost));
}

static char debugColor(Pixel pixel) {
    for (int c = 0; c < 5; c++) {
        if (memcmp(&pixel, &H_COLORS[c], sizeof(Pixel)) == 0) {
            return ".WRGB"[c];
        }
    }
    return '?';
}

void BoardManager::debugPrint(Board<Pixel> h_tex) {
    Board<Pixel> tex = {nullptr, size.rowBytes<Pixel>()};
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            printf("%c ", debugColor(h_tex(x, y)));
        }
        printf("\n");
    }
}

#if defined(BM_DBG_MAIN)
// Use with: make -CCFLAGS=-DBM_DBG_MAIN
#include <memory>

int main(int argc, char **argv) {
    Size size = {28, 28};
    BoardManager bm {size};
    auto h_texPtr = std::make_unique<Pixel[]>(size.totalElems());
    Board<Pixel> h_tex = {h_texPtr.get(), size.rowBytes<Pixel>()};

    bm.initialize();
    for (int i = 0; i < 100; i++) {
        bm.writeTexture(h_tex);
        bm.debugPrint(h_tex);
        bm.step();
        printf("----------------------------\n");
    }
    return 0;
}

#endif
