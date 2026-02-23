#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void trainStep(float* x, float* y, float* w, float* b, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float prediction = (*w) * x[i] + (*b);
        float error = prediction - y[i];
        atomicAdd(w, -lr * error * x[i] / n);
        atomicAdd(b, -lr * error / n);
    }
}

extern "C" void runTrainingStep(float* d_x, float* d_y, float& w, float& b, float lr, int n) {
    float *d_w, *d_b;
    cudaMalloc(&d_w, sizeof(float));
    cudaMalloc(&d_b, sizeof(float));
    cudaMemcpy(d_w, &w, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(float), cudaMemcpyHostToDevice);

    trainStep<<< (n+255)/256, 256 >>>(d_x, d_y, d_w, d_b, lr, n);

    cudaMemcpy(&w, d_w, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_w); cudaFree(d_b);
}
