#include <memory>
#include <iostream>
#include <stddef.h>
#include "data_load.h"
#include "tensor.h"
#include "utils.h"

int main() {
    IDXFile<3> imagesFile {"data/mnist/train-images.idx3-ubyte"};
    IDXFile<1> labelsFile {"data/mnist/train-labels.idx1-ubyte"};
    BatchGenerator bg {imagesFile, labelsFile, 5};

    bg.shuffle();
    for (int b = 0; b < 2; b++) {
        std::cout << "next() = " << bg.next() << std::endl;
        auto images = bg.getImages();
        auto labels = bg.getLabels();

        for (size_t i = 0; i < bg.batchSize(); i++) {
            int label = labels(i);
            std::cout << i << ": label = " << label << std::endl;
            for (size_t r = 0; r < 28; r++) {
                for (size_t c = 0; c < 28; c++) {
                    int pixel = images(i, r, c);
                    std::cout << (pixel < 0x80 ? ' ' : '#');
                }
                std::cout << std::endl;
            }
        }
    }
}
