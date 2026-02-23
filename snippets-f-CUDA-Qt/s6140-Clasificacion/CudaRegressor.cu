#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// #include "PointManager.h"
#include "CudaRegressor.h" // <--- USA EL HEADER

// --- KERNEL ---
__global__ void parabolaKernel(const float* x, const float* x2, const float* y, 
                               float* dw2, float* dw1, float* db, 
                               float w2, float w1, float b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float err = (w2 * x2[i] + w1 * x[i] + b) - y[i];
        atomicAdd(dw2, err * x2[i]);
        atomicAdd(dw1, err * x[i]);
        atomicAdd(db, err);
    }
}

// Implementación del método (Nota que ya no lleva "class CudaRegressor {")
void CudaRegressor::train(const PointManager& pm, int iterations) {
    int n = pm.getCount(); if (n == 0) return;
    float *dx, *dx2, *dy, *dw2, *dw1, *db;
    cudaMalloc(&dx, n*4); cudaMalloc(&dx2, n*4); cudaMalloc(&dy, n*4);
    cudaMalloc(&dw2, 4);  cudaMalloc(&dw1, 4);  cudaMalloc(&db, 4);
    
    cudaMemcpy(dx, pm.getX().data(), n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dx2, pm.getX2().data(), n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, pm.getY().data(), n*4, cudaMemcpyHostToDevice);

    for (int i = 0; i < iterations; ++i) {
        cudaMemset(dw2, 0, 4); cudaMemset(dw1, 0, 4); cudaMemset(db, 0, 4);
        parabolaKernel<<<(n+255)/256, 256>>>(dx, dx2, dy, dw2, dw1, db, w2, w1, b, n);
        float g2, g1, gb;
        cudaMemcpy(&g2, dw2, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&g1, dw1, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&gb, db, 4, cudaMemcpyDeviceToHost);
        w2 -= lr * (g2 / n); w1 -= lr * (g1 / n); b -= lr * (gb / n);
    }
    cudaFree(dx); cudaFree(dx2); cudaFree(dy); cudaFree(dw2); cudaFree(dw1); cudaFree(db);
}

