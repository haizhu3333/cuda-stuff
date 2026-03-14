#pragma once
#include <algorithm>
#include <array>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <utility>

template <size_t Nd, typename... Args>
concept IndexArgs = (sizeof...(Args) == Nd) && (std::convertible_to<Args, size_t> && ...);

struct Empty {};

template <typename T, size_t Nd>
struct Tensor {
    static constexpr bool Pitched = (Nd >= 2);
    using Byte = std::conditional_t<std::is_const_v<T>, const std::byte, std::byte>;
    using Pitch = std::conditional_t<Pitched, size_t, Empty>;

    T *ptr;
    size_t shape[Nd];
    [[no_unique_address]] Pitch pitch;

    Tensor() {}

    template <typename... Args>
    requires IndexArgs<Nd, Args...>
    Tensor(Args... args) : Tensor(nullptr, args...) {}

    template <typename... Args>
    requires IndexArgs<Nd, Args...>
    Tensor(T *ptr, Args... args) : ptr(ptr), shape{static_cast<size_t>(args)...} {
        if constexpr (Pitched) {
            pitch = allocWidth();
        }
    }

    template <typename... Args>
    requires IndexArgs<Nd, Args...> && Pitched
    Tensor(T *ptr, size_t pitch, Args... args) :
        ptr(ptr), shape{astatic_cast<size_t>(args)...}, pitch(pitch)
    {
        if (pitch < allocWidth()) {
            throw std::invalid_argument("pitch < row width");
        }
    }

    template <typename... Args>
    requires IndexArgs<Nd, Args...>
    __host__ __device__ T &operator()(Args... args) const {
        return *getAddr(args...);
    }

    size_t size() const {
        return std::accumulate(shape, shape + Nd, 1, std::multiplies());
    }

    size_t sizeBytes() const {
        if constexpr (Pitched) {
            return pitch * allocHeight();
        } else {
            return allocBytes();
        }
    }

    size_t allocBytes() const {
        return size() * sizeof(T);
    }

    size_t allocWidth() const requires Pitched {
        return shape[Nd - 1] * sizeof(T);
    }

    size_t allocHeight() const requires Pitched {
        return std::accumulate(shape, shape + Nd - 1, 1, std::multiplies());
    }

    Tensor<T, Nd - 1> outerSlice(size_t index) const requires Pitched {
        size_t midDims = std::accumulate(shape + 1, shape + Nd - 1, 1, std::multiplies());
        Byte *bptr = reinterpret_cast<Byte*>(ptr);
        T *slicePtr = reinterpret_cast<T*>(bptr + index * midDims * pitch);
        Tensor<T, Nd - 1> slice;
        slice.ptr = slicePtr;
        std::copy(shape + 1, shape + Nd, slice.shape);
        if constexpr (slice.Pitched) {
            slice.pitch = pitch;
        }
        return slice;
    }

private:
    // Special case for Nd = 0 or 1
    __host__ __device__ T *getAddr() const {
        static_assert(Nd == 0);
        return ptr;
    }

    __host__ __device__ T *getAddr(size_t i) const {
        static_assert(Nd == 1);
        return ptr + i;
    }

    // Nd >= 2 is always Pitched, use recursive template.
    template <typename... Args>
    __host__ __device__ T *getAddr(size_t i, Args... args) const {
        static_assert(Pitched);
        static_assert(sizeof...(Args) == Nd - 1);
        return pitchedAddrImpl(i, args...);
    }

    __host__ __device__ T *pitchedAddrImpl(size_t accIndex, size_t i) const {
        Byte *bptr = reinterpret_cast<Byte*>(ptr);
        T *row = reinterpret_cast<T*>(bptr + accIndex * pitch);
        return row + i;
    }

    template <typename... Args>
    __host__ __device__ T *pitchedAddrImpl(size_t accIndex, size_t i, Args... args) const {
        constexpr size_t D = Nd - sizeof...(Args) - 1;
        static_assert(D > 0 && D < Nd - 1);
        return pitchedAddrImpl(accIndex * shape[D] + i, args...);
    }
};
