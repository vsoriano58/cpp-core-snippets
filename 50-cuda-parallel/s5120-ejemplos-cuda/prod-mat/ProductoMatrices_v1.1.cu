/**
 * @file ProductoMatrices_v1.1.cu
 * @brief Multiplica dos matrices (A * B = C) y mide el tiempo total de ejecución. 
 *              Transferencias + Kernel
 *              Soporta N > 1024
 * 
 * @author alcón68
 * CONCEPT: Calculamos el producto elemento a elemento de dos matrices A y B.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

// [ETAPA 4.1] KERNEL: Ejecución en el Device
__global__ void productoMatrices(const float *A, const float *B, float *C, int N) {
    // [ETAPA 4.2] CÁLCULO DEL ÍNDICE GLOBAL
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] * B[i];
    }
}

int main() {
    // PARÁMETRO: Puedes subir este número (ej. 1000000) y seguirá funcionando
    const int N = 1048576; // 1024 * 1024 (1 Megaelemento)
    size_t size = N * sizeof(float);

    // [ETAPA 1] HOST ALLOCATION e Inicialización
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / (float)RAND_MAX;
        h_B[i] = (float)rand() / (float)RAND_MAX;
    }

    // [ETAPA 2] DEVICE ALLOCATION (Reserva en VRAM)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // --- INICIO DE MEDICIÓN TOTAL (ETAPAS 3, 4 y 5) ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // [ETAPA 3] HOST-TO-DEVICE TRANSFER (Carga)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // [ETAPA 4] KERNEL LAUNCH (Configuración Dinámica)
    int threadsPerBlock = 256; 
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    productoMatrices<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // [ETAPA 5] DEVICE-TO-HOST TRANSFER (Descarga)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // --- FIN DE MEDICIÓN ---
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Sincronización crucial

    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, start, stop);

    // SALIDA DE RESULTADOS
    std::cout << ">>> RESULTADOS CUDA <<<" << std::endl;
    std::cout << "Elementos procesados: " << N << std::endl;
    std::cout << "Configuracion: " << blocksPerGrid << " bloques de " << threadsPerBlock << " hilos." << std::endl;
    std::cout << "Tiempo total (Carga + Kernel + Descarga): " << ms_total << " ms" << std::endl;
    
    // Verificación rápida del primer y último elemento
    std::cout << "\nVerificacion de calculo:" << std::endl;
    std::cout << "Indice [0]: " << h_A[0] << " * " << h_B[0] << " = " << h_C[0] << std::endl;
    std::cout << "Indice [" << N-1 << "]: " << h_A[N-1] << " * " << h_B[N-1] << " = " << h_C[N-1] << std::endl;

    // LIMPIEZA
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}

/** 
 *  Salida del programa (Clarificador comparado con el ejemplo anterior)
 *  ===================
 *  Tiempo total (Carga + Kernel + Descarga): 6.13421 ms

    Verificacion de calculo:
    Indice [0]: 0.411995 * 0.447815 = 0.184498
    Indice [1048575]: 0.631638 * 0.253176 = 0.159916
 */

 /*
    Compilar
    ========
    nvcc ProductoMatrices_v1.1.cu -o ./build/ProductoMatrices_v1.1
    ./build/ProductoMatrices_v1.1
 */


