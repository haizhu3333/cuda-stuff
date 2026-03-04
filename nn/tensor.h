#pragma once
#include <algorithm>
#include <utility>

using std::byte;

struct Empty {
};

struct Pitch {
    size_t pitch;
};

template <size_t Nd, typename... Args>
concept IndexArgs = (sizeof...(Args) == Nd) && (std::convertible_to<Args, size_t> && ...);

template <typename T, size_t Nd, bool Pitched = (Nd > 1)>
requires (!Pitched || Nd > 1)
struct Tensor : public std::conditional_t<Pitched, Pitch, Empty> {
    T *ptr;
    size_t shape[Nd];

    Tensor() : ptr(nullptr) {
        std::fill(shape, shape + Nd, 0);
        if constexpr (Pitched) {
            this->pitch = 0;
        }
    }

    template <typename... Args>
    requires IndexArgs<Nd, Args...>
    Tensor(Args... args) : Tensor<Args...>(nullptr, args...) {}

    template <typename... Args>
    requires IndexArgs<Nd, Args...>
    Tensor(T *ptr, Args... args) : ptr(ptr) {
        fillShape(std::make_index_sequence<Nd>{}, args...);
        if constexpr (Pitched) {
            this->pitch = calcPitch(args...);
        }
    }

    template <typename... Args>
    requires IndexArgs<Nd, Args...>
    T &operator()(Args... args) {
        byte *bptr = reinterpret_cast<byte*>(ptr);
        bptr += bytesOffset(0, args...);
        return *reinterpret_cast<T*>(bptr);
    }

    Tensor<T, Nd - 1, false> outerSlice(size_t index) requires Pitched {
        Tensor<T, Nd - 1, false> slice;
        slice.ptr = outerSlicePtr(index, std::make_index_sequence<Nd - 1>{});
        std::copy(shape + 1, shape + Nd, slice.shape);
        return slice;
    }

private:
    template <size_t... Is, typename... Args>
    void fillShape(std::index_sequence<Is...>, Args... args) {
        ((shape[Is] = args), ...);
    }

    template <size_t... Is, typename... Args>
    static size_t calcPitch(size_t unused_size0, Args... args) {
        return sizeof(T) * (args * ...);
    }

    size_t bytesOffset(size_t acc) {
        return acc * sizeof(T);
    }

    template <typename... Args>
    size_t bytesOffset(size_t acc, size_t i, Args... args) {
        constexpr size_t D = Nd - sizeof...(Args) - 1;
        if constexpr (Pitched && D == 0) {
            return i * this->pitch + bytesOffset(acc, args...);
        } else {
            return bytesOffset(acc * shape[D] + i, args...);
        }
    }

    template <size_t... Is>
    T *outerSlicePtr(size_t index, std::index_sequence<Is...>) {
        return &operator()(index, ((void)Is, 0)...);
    }
};
