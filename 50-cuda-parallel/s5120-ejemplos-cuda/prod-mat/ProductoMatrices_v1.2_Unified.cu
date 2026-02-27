/**
 * @file ProductoMatrices_v1.2_Unified.cu
 * @brief Multiplica matrices usando MEMORIA UNIFICADA (cudaMallocManaged).
 * @author alcón68
 * 
 * CONCEPT: Eliminamos cudaMemcpy. CPU y GPU ven el mismo puntero.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <iomanip> // Necesario para setprecision y fixed

__global__ void productoMatrices(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] * B[i];
}

int main() {
    const int N = 1048576;
    size_t size = N * sizeof(float);

    // [ETAPA 1 y 2 UNIFICADAS] Reserva de memoria gestionada
    float *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // [ETAPA 3] INICIALIZACIÓN (Directamente en el puntero "mágico")
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f; // Simplificado para verificar rápido
        B[i] = 2.0f;
    }

    // [ETAPA 4] LANZAMIENTO
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    productoMatrices<<<blocks, threads>>>(A, B, C, N);

    // [CRUCIAL] Sincronización
    // Como no hay cudaMemcpy (que bloquea), debemos esperar a la GPU antes de leer
    cudaDeviceSynchronize();

    // [ETAPA 5] VERIFICACIÓN (Leemos C directamente)
    std::cout << std::fixed << std::setprecision(1);
    std::cout << ">>> MEMORIA UNIFICADA <<<" << std::endl;
    std::cout << "Resultado indice 0: " << C[0] << " (Debe ser 2.0)" << std::endl;

    // LIMPIEZA
    cudaFree(A); cudaFree(B); cudaFree(C);

    return 0;
}

/*  NOTA TÉCNICA
    ============
    - cudaMallocManaged: Es la palabra clave. Reserva memoria que es válida tanto para el main() 
      (CPU) como para el kernel (GPU). Adiós a los prefijos h_ y d_.
    - Adiós cudaMemcpy: Ya no los ves, pero siguen ocurriendo. El driver de NVIDIA detecta cuando 
      la GPU intenta acceder a A y, por debajo, mueve las páginas de memoria necesarias. Es más 
      cómodo, pero no siempre más rápido.
    - cudaDeviceSynchronize() es OBLIGATORIO: En los códigos anteriores, cudaMemcpy hacía de 
      "barrera" (la CPU esperaba a que terminara la copia). Aquí, la CPU lanzaría el kernel y 
      seguiría directa al std::cout sin esperar a que la GPU termine. Si quitas esa línea, podrías 
      imprimir un valor vacío o basura.
*/

/*
    Salida del programa
    ===================
    >>> MEMORIA UNIFICADA <<<
    Resultado indice 0: 2.0 (Debe ser 2.0)
*/

/*
    Compilar
    ========
    nvcc ProductoMatrices_v1.2_Unified.cu -o ./build/ProductoMatrices_v1.2_Unified
    ./build/ProductoMatrices_v1.2_Unified
*/

