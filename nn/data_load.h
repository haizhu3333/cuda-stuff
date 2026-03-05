#pragma once
#include <random>
#include <stdint.h>
#include "tensor.h"

template <uint8_t Nd>
class IDXFile {
    void *bytes;
    size_t size;

public:
    IDXFile(const char *path);
    ~IDXFile();

    Tensor<const uint8_t, Nd> getTensor() const;
};

class BatchGenerator {
    Tensor<const uint8_t, 3> sourceImages;
    Tensor<const uint8_t, 1> sourceLabels;

    std::minstd_rand rng;
    std::vector<size_t> indices;
    size_t offset;

    Tensor<uint8_t, 3> batchImages;
    Tensor<uint8_t, 1> batchLabels;

public:
    BatchGenerator(const IDXFile<3> &imagesFile, const IDXFile<1> &labelsFile, size_t batchSize);
    ~BatchGenerator();
    size_t batchSize() { return batchImages.shape[0]; }
    void shuffle();
    bool next();
    const Tensor<uint8_t, 3> &getImages() { return batchImages; }
    const Tensor<uint8_t, 1> &getLabels() { return batchLabels; }
};

