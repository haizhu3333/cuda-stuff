#pragma once
#include "utils.h"

typedef uint8_t Cell;
typedef uchar4 Pixel;

class BoardManager {

    Size size;
    Board<Cell> d_curr;
    Board<Cell> d_next;
    Board<Pixel> d_tex;

public:
    BoardManager(Size size);
    ~BoardManager();

    void initialize();
    void step();
    void writeTexture(Board<Pixel> h_tex);
    void debugPrint(Board<Pixel> h_tex);
};
