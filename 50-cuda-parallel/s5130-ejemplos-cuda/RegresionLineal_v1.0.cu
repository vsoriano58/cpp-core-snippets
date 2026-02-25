/**
 * Programa: Deeplearning_v1.0.cu
 * Objetivo: Regresión lineal simple con Descenso del Gradiente Atómico.
 */

#include <iostream>
#include <cuda_runtime.h>

// [ETAPA 4.1] KERNEL: Calcula el gradiente de un solo paso
__global__ void calcularGradiente(const float *x, const float *y, float w, float b, float *d_grad_w, float *d_grad_b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float prediccion = w * x[i] + b;
        float error = prediccion - y[i];

        // Suma atómica segura para acumular el gradiente total
        atomicAdd(d_grad_w, error * x[i]);
        atomicAdd(d_grad_b, error);
    }
}

int main() {
    // 1. DATOS REALES (Muestra de entrenamiento)
    const int N = 20;
    float h_x[] = {0.080, 9.000, 0.001, 0.100, 8.000, 5.000, 0.100, 6.000, 0.050, 0.500, 0.002, 2.000, 0.005, 10.00, 0.010, 7.000, 6.000, 5.000, 1.000, 1.000};
    float h_y[] = {0.135, 0.739, 0.067, 0.126, 0.646, 0.435, 0.069, 0.497, 0.068, 0.116, 0.070, 0.289, 0.076, 0.744, 0.083, 0.560, 0.480, 0.399, 0.153, 0.149};

    // 2. RESERVA EN GPU
    float *d_x, *d_y, *d_grad_w, *d_grad_b;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMalloc((void**)&d_grad_w, sizeof(float));
    cudaMalloc((void**)&d_grad_b, sizeof(float));

    // 3. CARGA INICIAL
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Parámetros del modelo e hiperparámetros
    float w = 0.0f; 
    float b = 0.0f;
    float learning_rate = 0.001f; // Empezamos con algo pequeño y seguro
    int epocas = 1000;

    std::cout << "Entrenando modelo..." << std::endl;

    // EL BUCLE DE ENTRENAMIENTO (En la CPU)
    for (int epoca = 0; epoca < epocas; epoca++) {
        // Limpiamos los gradientes antes de cada pasada
        cudaMemset(d_grad_w, 0, sizeof(float));
        cudaMemset(d_grad_b, 0, sizeof(float));

        // Lanzamos el Kernel (20 hilos en 1 bloque)
        calcularGradiente<<<1, 256>>>(d_x, d_y, w, b, d_grad_w, d_grad_b, N);

        // Bajamos los gradientes calculados por la GPU para actualizar w y b
        float h_grad_w, h_grad_b;
        cudaMemcpy(&h_grad_w, d_grad_w, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_grad_b, d_grad_b, sizeof(float), cudaMemcpyDeviceToHost);

        // Actualización de los coeficientes (Descenso del Gradiente)
        // Usamos (2.0/N) para normalizar el gradiente según la derivada del MSE
        w = w - learning_rate * (2.0f / N) * h_grad_w;
        b = b - learning_rate * (2.0f / N) * h_grad_b;

        if (epoca % 100 == 0) {
            std::cout << "Epoca " << epoca << " | w: " << w << " b: " << b << std::endl;
        }
    }

    // 5. RESULTADO FINAL
    std::cout << "\nEntrenamiento completado." << std::endl;
    std::cout << "Modelo final: Tiempo = " << w << " * Tamaño + " << b << std::endl;

    // Limpieza
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_grad_w); cudaFree(d_grad_b);
    return 0;
}

// Compilar
// nvcc RegresionLineal_v1.0.cu -o RegresionLineal_v1.0