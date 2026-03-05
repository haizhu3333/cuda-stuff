#include <arpa/inet.h>
#include <assert.h>
#include <bit>
#include <fcntl.h>
#include <numeric>
#include <random>
#include <sstream>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <system_error>
#include <tuple>
#include <unistd.h>
#include "data_load.h"
#include "tensor.h"
#include "utils.h"

[[noreturn]]
static void posixFailure(const char *what) {
    throw std::system_error(errno, std::generic_category(), what);
}

class FdWrapper {
    int fd;

public:
    FdWrapper(const char *path) {
        printf("Opening %s\n", path);
        fd = open(path, O_RDONLY);
        if (fd == -1) {
            posixFailure("open() failed");
        }
    }

    ~FdWrapper() {
        if (close(fd) == -1) {
            posixFailure("close() failed");
        }
    }

    int getFd() {
        return fd;
    }

    size_t getSize() {
        struct stat fileInfo = {0};
        if (fstat(fd, &fileInfo) == -1) {
            posixFailure("stat() failed");
        }
        return fileInfo.st_size;
    }
};

template <uint8_t DIM>
IDXFile<DIM>::IDXFile(const char *path) {
    FdWrapper fdWrapper {path};
    size = fdWrapper.getSize();
    bytes = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fdWrapper.getFd(), 0);
    if (bytes == MAP_FAILED) {
        posixFailure("mmap() failed");
    }
}

template <uint8_t DIM>
IDXFile<DIM>::~IDXFile() {
    if (munmap(bytes, size) == -1) {
        posixFailure("munmap() failed");
    }
}

static uint32_t nextU32(const uint8_t *&ptr, const uint8_t *end) {
    const uint8_t *nextPtr = ptr + sizeof(uint32_t);
    if (nextPtr > end) {
        throw std::runtime_error("Unexpected end of file");
    }
    uint32_t value;
    memcpy(&value, ptr, sizeof(uint32_t));
    ptr = nextPtr;
    return ntohl(value);
}

template <uint8_t Nd>
Tensor<const uint8_t, Nd> IDXFile<Nd>::getTensor() const {
    const uint8_t *ptr = static_cast<const uint8_t *>(bytes);
    const uint8_t *end = ptr + size;
    std::array<size_t, Nd> shape;

    uint32_t magic = nextU32(ptr, end);
    if (magic != 0x0800 + Nd) {
        std::stringstream ss;
        ss << "Wrong magic number: " << std::hex << magic << std::dec;
        throw std::runtime_error(ss.str());
    }

    for (size_t i = 0; i < Nd; i++) {
        shape[i] = nextU32(ptr, end);
    }

    auto tensor = std::make_from_tuple<Tensor<const uint8_t, Nd>>(shape);
    if (end - ptr != static_cast<ptrdiff_t>(tensor.sizeBytes())) {
        std::stringstream ss;
        ss << "Expected " << tensor.sizeBytes() << " data bytes, got " << end - ptr;
        throw std::runtime_error(ss.str());
    }
    tensor.ptr = ptr;
    return tensor;
}

template class IDXFile<1>;
template class IDXFile<3>;

BatchGenerator::BatchGenerator(
    const IDXFile<3> &imagesFile, const IDXFile<1> &labelsFile, size_t batchSize
) :
    sourceImages(imagesFile.getTensor()),
    sourceLabels(labelsFile.getTensor()),
    indices(sourceImages.shape[0]),
    offset(0),
    batchImages(batchSize, sourceImages.shape[1], sourceImages.shape[2]),
    batchLabels(batchSize)
{
    if (sourceImages.shape[0] != sourceLabels.shape[0]) {
        throw std::invalid_argument("images and labels have different number of items");
    }
    std::random_device rd;
    rng.seed(rd());
    std::iota(indices.begin(), indices.end(), 0);

    CHECK(cudaMallocHost(&batchImages.ptr, batchImages.allocBytes()));
    CHECK(cudaMallocHost(&batchLabels.ptr, batchLabels.allocBytes()));
}

BatchGenerator::~BatchGenerator() {
    CHECK(cudaFreeHost(batchImages.ptr));
    CHECK(cudaFreeHost(batchLabels.ptr));
}

void BatchGenerator::shuffle() {
    std::shuffle(indices.begin(), indices.end(), rng);
    offset = 0;
}

bool BatchGenerator::next() {
    if (offset + batchSize() > indices.size()) {
        return false;
    }
    for (size_t i = 0; i < batchSize(); i++) {
        size_t iSrc = indices[offset + i];
        Tensor<const uint8_t, 2> src = sourceImages.outerSlice(iSrc);
        Tensor<uint8_t, 2> dst = batchImages.outerSlice(i);
        assert(std::ranges::equal(src.shape, dst.shape));
        memcpy(dst.ptr, src.ptr, src.sizeBytes());

        batchLabels(i) = sourceLabels(iSrc);
    }
    offset += batchSize();
    return true;
}
