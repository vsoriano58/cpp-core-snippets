/**
 * Programa: Suma_Matrices_3x3.cu
 * Descripción: Implementación completa 2D con verificación de resultados.
 */

#include <iostream>
#include <cuda_runtime.h>

// [ETAPA 4.1] DEFINICIÓN DEL KERNEL
__global__ void matrixAdd(const float *A, const float *B, float *C, int width, int height) {
    // [ETAPA 4.2] CÁLCULO DE ÍNDICES 2D
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Mapeo de 2D a índice lineal 1D
    int i = row * width + col;

    // Boundary check (importante aunque sea 3x3)
    if (col < width && row < height) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Definición de dimensiones
    const int width = 3;
    const int height = 3;
    const int N = width * height;
    size_t size = N * sizeof(float);

    // [ETAPA 1] HOST ALLOCATION e Inicialización (1s y 2s)
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // [ETAPA 2] DEVICE ALLOCATION (Reserva de VRAM)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // [ETAPA 3] HOST-TO-DEVICE TRANSFER
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // [ETAPA 4.3] KERNEL LAUNCH (Configuración 2D)
    // Usamos bloques de 16x16 por estándar, aunque sobren hilos para 3x3
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width, height);

    // [ETAPA 5] DEVICE-TO-HOST TRANSFER
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // VERIFICACIÓN: Imprimir la matriz resultado
    std::cout << "Matriz Resultante (Debe ser todo 3.0):" << std::endl;
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            std::cout << h_C[r * width + c] << " ";
        }
        std::cout << std::endl;
    }

    // LIMPIEZA
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
/**
 * Fin del programa.
 * Nota: Aunque la matriz sea pequeña, el overhead de mover datos (Etapa 3 y 5) 
 * es mayor que el cálculo, pero sirve para validar la lógica 2D.
 */

// Compilar
// nvcc SumarMatrices.cu -o SumarMatrices

// Ejecutar
// ./SumarMatrices


