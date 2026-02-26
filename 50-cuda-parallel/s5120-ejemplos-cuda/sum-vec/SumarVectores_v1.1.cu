/**
 * Programa: Suma de Vectores en CUDA (Vector Addition)
 * Descripción: Implementación del flujo básico de 5 etapas para procesamiento paralelo.
 * Autor: Thought Partner & Estudiante de CUDA
 */

#include <iostream>
#include <cuda_runtime.h>

// [ETAPA 4.1] DEFINICIÓN DEL KERNEL: Ejecución en el Device
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    // [ETAPA 4.2] CÁLCULO DEL ÍNDICE GLOBAL: Mapeo de hilos a datos
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Verificación de límites para evitar accesos fuera de rango
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000; 
    size_t size = N * sizeof(float);

    // [ETAPA 1] HOST ALLOCATION: Reserva en memoria RAM de la CPU
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Inicialización de datos en el Host
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // [ETAPA 2] DEVICE ALLOCATION: Reserva en VRAM de la GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // [ETAPA 3] HOST-TO-DEVICE MEMORY TRANSFER: Carga de datos a la GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // [ETAPA 4.3] KERNEL LAUNCH: Configuración y lanzamiento de hilos
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // [ETAPA 5] DEVICE-TO-HOST MEMORY TRANSFER: Recuperación de resultados
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verificación del cálculo
    std::cout << "Resultado final: " << h_C[0] << " (Debe ser 3.0)" << std::endl;

    // LIMPIEZA: Liberación de recursos en Device y Host
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

/*
    Salida del programa:
    
    Resultado final: 3 (Debe ser 3.0)
*/

// Compilar
// nvcc SumarVectores_v1.1.cu -o ./build/SumarVectores_v1.1

// ./build/SumarVectores_v1.1
