#include <device_launch_parameters.h>
#include <iostream>
#include <cuda_runtime.h>

/**
 * KERNEL: Esta función se ejecuta en la GPU.
 * Cada hilo (thread) que lancemos ejecutará este código en paralelo.
 */
__global__ void miPrimerKernel() {
    // threadIdx.x es una variable mágica que nos dice el ID del hilo actual
    int id = threadIdx.x;
    printf("Hola desde la GPU! Soy el hilo número: %d\n", id);
}

int main() {
    std::cout << "=== SNIPPET #0010 (CUDA): LANZAMIENTO DE HILOS ===\n\n";

    // Lanzamos el kernel: <<< Bloques, Hilos por bloque >>>
    // Vamos a lanzar 8 hilos en paralelo.
    miPrimerKernel<<<1, 8>>>();

    // ¡CRÍTICO! La CPU debe esperar a que la GPU termine de imprimir.
    cudaDeviceSynchronize();

    std::cout << "\n=== Fin del programa (CPU) ===\n";
    return 0;
}

// Compilar
// nvcc hola_cuda.cu -o hola_cuda

// Ejecutar
// ./hola_cuda