// Este archivo será el "contrato" entre la GPU y Qt.

#ifndef CUDAREGRESSOR_H
#define CUDAREGRESSOR_H
#include "PointManager.h"

class CudaRegressor {
public:
    float w2 = 0.0f, w1 = 0.0f, b = 0.0f, lr = 0.05f;
    void train(const PointManager& pm, int iterations);
};
#endif
