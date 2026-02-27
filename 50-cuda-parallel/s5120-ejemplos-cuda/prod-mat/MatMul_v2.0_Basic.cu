/**
 * @file MatMul_v2.0_Basic.cu
 * @brief Producto de matrices REAL (C = A x B)
 * @author alcón68
 */

#include <iostream>
#include <cuda_runtime.h>

// [KERNEL] Cada hilo calcula UN elemento de la matriz C
__global__ void matMulKernel(float *A, float *B, float *C, int N) {
    // 1. Identificar fila y columna que le toca a este hilo
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        // 2. El "corazón" del producto escalar: recorrer fila de A y columna de B
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        // 3. Guardar el resultado final en la posición correspondiente
        C[row * N + col] = sum;
    }
}

int main() {
    // Usamos una matriz de 32x32 para que sea fácil de ver (1024 elementos)
    const int N = 32; 
    size_t size = N * N * sizeof(float);

    float *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // Inicialización: A = 1s, B = 2s. Resultado esperado en C: N * (1*2) = 64
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Configuración 2D: bloques de 16x16 hilos
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    matMulKernel<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();

    std::cout << "Matriz " << N << "x" << N << " multiplicada." << std::endl;
    std::cout << "Resultado C[0][0]: " << C[0] << " (Esperado: " << (float)N * 1 * 2 << ")" << std::endl;

    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}

/*
    NOTA TÉCNICA
    ============
    - El índice k: Es el bucle que "viaja" por la fila de A y la columna de B. Fíjate cómo row se 
      mantiene fijo mientras k avanza en A, y col se mantiene fijo mientras k baja en B.
    - A[row * N + k]: Acceso por filas (horizontal).
    - B[k * N + col]: Acceso por columnas (vertical).
    - Eficiencia: Este kernel es el "básico". En el futuro verás que acceder a la memoria de la 
      GPU saltando de fila en fila (en la matriz B) es lento. Para arreglarlo se usa la Memoria 
      Compartida (Shared Memory), pero eso lo dejamos para cuando este código esté bien digerido.

*/

/*
    Salida del programa
    ===================
    Matriz 32x32 multiplicada.
    Resultado C[0][0]: 64 (Esperado: 64)
*/

/*
    Compilar
    ========
    nvcc MatMul_v2.0_Basic.cu -o ./build/MatMul_v2.0_Basic
    ./build/MatMul_v2.0_Basic
*/

/*
    NOTA FINAL
    ==========
    La potencia oculta del paralelismo
    En una CPU tradicional, para una matriz de 32x32, el procesador hace:
    1. Calcula Coo (recorriendo 32 veces el bucle k).
    2. Calcula Co1 (otras 32 veces...).
    Y así hasta terminar los 1024 elementos.
    En este código de GPU, gracias a dim3 threads(16, 16), has lanzado 1024 hilos simultáneos.
    Cada hilo tiene su propio registro para la variable sum.
    Todos los hilos entran en el bucle for(k...) a la vez.
    En el tiempo que la CPU calcularía un par de elementos, la GPU ha terminado la matriz entera.
*/