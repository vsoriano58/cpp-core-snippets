/**
 * @file ProductoMatrices_v1.0.cu
 * @brief Multiplica dos matrices (A * B = C) y mide el tiempo de ejecución del kernel.
 * @author alcón68
 * 
 * CONCEPT: Calculamos el producto elemento a elemento de dos matrices A y B.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

// [ETAPA 4.1] KERNEL: Producto elemento a elemento
__global__ void productoMatrices(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] * B[i]; // Operación solicitada
    }
}

int main() {
    // PARÁMETRO: Tamaño de la matriz
    const int N = 1024;
    size_t size = N * sizeof(float);

    // [ETAPA 1] HOST ALLOCATION e inicialización aleatoria
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    srand(time(NULL)); // Semilla para aleatorios
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / (float)RAND_MAX;
        h_B[i] = (float)rand() / (float)RAND_MAX;
    }

    // [ETAPA 2] DEVICE ALLOCATION
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // [ETAPA 3] HOST-TO-DEVICE TRANSFER
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // --- PREPARACIÓN PARA MEDIR TIEMPO ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // [ETAPA 4.3] KERNEL LAUNCH
    // Tal como pediste: 1 bloque con N hilos
    cudaEventRecord(start); // Marca de tiempo INICIO

    productoMatrices<<<1, N>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);  // Marca de tiempo FINAL
    // -------------------------------------

    // [ETAPA 5] DEVICE-TO-HOST TRANSFER
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // CÁLCULO DEL TIEMPO TRANSCURRIDO
    cudaEventSynchronize(stop); // Espera a que la GPU termine realmente
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // SALIDA DE RESULTADOS
    std::cout << "Tamaño de la matriz: " << N << " elementos." << std::endl;
    std::cout << "Tiempo de ejecucion del Kernel: " << milliseconds << " ms" << std::endl;
    std::cout << "Verificacion (indice 0): " << h_A[0] << " * " << h_B[0] << " = " << h_C[0] << std::endl;

    // LIMPIEZA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}

/*
    Salida del programa
    ===================
    Tamaño de la matriz: 1024 elementos.
    Tiempo de ejecucion del Kernel: 0.178176 ms
    Verificacion (indice 0): 0.888929 * 0.0420423 = 0.0373726
*/

/*
    Compilar
    ========
    nvcc ProductoMatrices_v1.0.cu -o ./build/ProductoMatrices_v1.0
    ./build/ProductoMatrices_v1.0
*/



