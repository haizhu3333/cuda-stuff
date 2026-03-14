#pragma once
#include "tensor.h"

typedef Tensor<float, 1> Vector;
typedef Tensor<float, 2> Matrix;

struct Linear {
    Matrix weights;  // size: in x out
    Vector biases;   // size: out

    Linear(int in, int out);
    ~Linear();

    // input : batch x in
    // output : batch x out
    void forward(const Matrix &output, const Matrix &input);
};
