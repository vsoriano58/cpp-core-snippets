#include <device_launch_parameters.h>
#include <iostream>
#include <cuda_runtime.h>

// KERNEL: Cada hilo sumará una posición del array
__global__ void sumarVectores(int* a, int* b, int* c) {
    int i = threadIdx.x; 
    c[i] = a[i] + b[i]; // La GPU hace el trabajo
}

int main() {
    const int N = 5;
    int h_a[N] = {1, 2, 3, 4, 5}; // h_ = Host (CPU)
    int h_b[N] = {10, 20, 30, 40, 50};
    int h_c[N]; // Aquí guardaremos el resultado final

    // 1. Punteros para la memoria de la GPU (d_ = Device)
    int *d_a, *d_b, *d_c;

    // 2. Reservar memoria en la GPU
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // 3. Copiar datos de la CPU a la GPU
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // 4. Lanzar el kernel con N hilos (un hilo por número)
    sumarVectores<<<1, N>>>(d_a, d_b, d_c);

    // 5. Copiar el resultado de la GPU de vuelta a la CPU
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Mostrar resultados
    std::cout << "Resultado de la suma en GPU:" << std::endl;
    for(int i=0; i<N; i++) std::cout << h_c[i] << " ";
    std::cout << std::endl;

    // 6. Limpiar la casa (VRAM)
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}


// Compilar
// nvcc hola_cuda2.cu -o hola_cuda2

// Ejecutar
// ./hola_cuda2