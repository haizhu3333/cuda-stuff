#include <numeric>
#include <iostream>
#include "tensor.h"


int main() {
    int arr[10000];
    std::iota(arr, arr + 10000, 0);

    Tensor<int, 0> t0 {arr};
    Tensor<int, 1> t1 {arr, 5};
    Tensor<int, 2> t2 {arr, 5, 10};
    Tensor<int, 3> t3 {arr, 5, 10, 10};
    Tensor<int, 4> t4 {arr, 5, 10, 10, 10};

    std::cout << "sizeof t0 = " << sizeof t0 << std::endl;
    std::cout << "sizeof t1 = " << sizeof t1 << std::endl;
    std::cout << "sizeof t2 = " << sizeof t2 << std::endl;
    std::cout << "sizeof t3 = " << sizeof t3 << std::endl;
    std::cout << "sizeof t4 = " << sizeof t4 << std::endl;

    std::cout << "t0 bytes = " << t0.allocBytes() << std::endl;
    std::cout << "t1 bytes = " << t1.allocBytes() << std::endl;
    std::cout << "t2 bytes = " << t2.allocBytes() << std::endl;
    std::cout << "t3 bytes = " << t3.allocBytes() << std::endl;
    std::cout << "t4 bytes = " << t4.allocBytes() << std::endl;
    std::cout << "t4 width = " << t4.allocWidth() << std::endl;
    std::cout << "t4 height = " << t4.allocHeight() << std::endl;

    std::cout << "t0() = " << t0() << std::endl;
    std::cout << "t1(4) = " << t1(4) << std::endl;
    std::cout << "t2(4, 1) = " << t2(4, 1) << std::endl;
    std::cout << "t3(4, 1, 2) = " << t3(4, 1, 2) << std::endl;
    std::cout << "t4(4, 1, 2, 7) = " << t4(4, 1, 2, 7) << std::endl;
    std::cout << "t4.pitch = " << t4.pitch << std::endl;
    t4.pitch *= 2;
    std::cout << "t4(4, 1, 2, 7) = " << t4(4, 1, 2, 7) << std::endl;


    auto s = t4.outerSlice(4);
    std::cout << "s.pitch = " << s.pitch << std::endl;
    std::cout << "s(1, 2, 7) = " << s(1, 2, 7) << std::endl;

    auto ss = s.outerSlice(1);
    std::cout << "ss.pitch = " << ss.pitch << std::endl;
    std::cout << "ss(2, 7) = " << ss(2, 7) << std::endl;

    auto sss = ss.outerSlice(2);
    std::cout << "sss(7) = " << sss(7) << std::endl;
}
