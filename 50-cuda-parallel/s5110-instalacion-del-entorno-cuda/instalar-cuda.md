# 1. El Instalación y bases del entorno CUDA (S5110)

# Índice

- 1.- Instalar CUDA
  - 1.1.- Preparación del Sistema (Linux/Ubuntu)
  - 1.2.- El "¡Hola Mundo!" de la GPU
  - 1.3.- Compilación y Ejecución
  - 1.4.- Resultados
  - 1.5.- Salida del programa
  - 1.6.- El orden en la GPU vs CPU (Kotlin/Java/C++ Threads)
  - 1.6.1.- SumarVectores_1.0.cu, de un vistazo
  - 1.6.2.- 📝 Análisis técnico de sumarVectores.cu
  - 1.6.3.- SumarVectores_1.1.cu
- 2.- Metodología Universal
  - 2.1.- ⚙️ El Flujo de Trabajo en CUDA (Los 6 Pasos)

---

## 1.1.- Preparación del Sistema (Linux/Ubuntu)
Antes de nada, confirma que tienes una `tarjeta NVIDIA` reconocida por el sistema con el comando `lspci`: 

```bash
lspci | grep -i nvidia
```

La salida del coomando debe ser algo así:

```bash
02:00.0 VGA compatible controller: NVIDIA Corporation AD106 [GeForce RTX 4060 Ti] (rev a1)
02:00.1 Audio device: NVIDIA Corporation Device 22bd (rev a1)
```

### Paso A: Instalar el Driver de NVIDIA

Comprueba primero si lo tienes instalado con el comando `nvidia-smi`. Si no lo tienes instalado, lo más secillo en `Ubuntu` es usar el repositorio oficial:

```bash
sudo apt update
sudo apt install nvidia-driver-535  # O la versión más reciente compatible
sudo reboot
```

En mi caso tengo instalado `nvidia-driver-580-privativo`.

### Paso B: Instalar el CUDA Toolkit

Este paquete contiene las librerías y el compilador necesario: 
```bash
sudo apt install nvidia-cuda-toolkit
```

Para verificar la instalación, escribe `nvcc -V` en un terminal. Debería mostrarte la versión del compilador de NVIDIA. La salida del comando debe verse como algo así:

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
```

---

## 1.2 El "¡Hola Mundo!" de la GPU

En CUDA, los archivos no son `.cpp`, sino `.cu`. Esto le indica al compilador que hay código mixto: para la `CPU (host)` y para la `GPU (device)`. 
Crea un archivo llamado `hola-cuda.cu` con este código: 

**holaCuda.cu**

```cpp
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
    std::cout << "=== SNIPPET s5110 (CUDA): LANZAMIENTO DE HILOS ===\n\n";

    // Lanzamos el kernel: <<< Bloques, Hilos por bloque >>>
    // Vamos a lanzar 8 hilos en paralelo.
    miPrimerKernel<<<1, 8>>>();

    // ¡CRÍTICO! La CPU debe esperar a que la GPU termine de imprimir.
    cudaDeviceSynchronize();

    std::cout << "\n=== Fin del programa (CPU) ===\n";
    return 0;
}

// COMPILAR
// ========
// nvcc holaCuda.cu -o ./build/holaCudaS
// ./holaCuda
```

---

## 1.3 Compilación y Ejecución

Para compilar este código, no usamos g++, sino nvcc (NVIDIA CUDA Compiler Driver):

```bash
nvcc holaCuda.cu -o ./build/holaCuda
./holaCuda
```

**Resumen**
* __global__: La palabra clave que marca una función para que corra en la tarjeta de video.
* **<<<1, 10>>>**: La sintaxis de "lanzamiento". Aquí le decimos: "Usa 1 bloque de 10 hilos paralelos".
* **cudaDeviceSynchronize()**: Fundamental. La CPU y la GPU trabajan de forma asíncrona; sin esta línea, la CPU podría cerrar el programa antes de que la GPU termine de imprimir su mensaje. 

¿Te ha dado algún error la instalación o ya has podido ver los 8 mensajes paralelos de la GPU en tu terminal?

---

## 1.4 Resultados

**Salida de `nvcc -V`**

En mi caso:

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302₀
```

---

## 1.5 Salida del programa

```bash
=== SNIPPET s5110 (CUDA): LANZAMIENTO DE HILOS ===

Hola desde la GPU! Soy el hilo número: 0
Hola desde la GPU! Soy el hilo número: 1
Hola desde la GPU! Soy el hilo número: 2
Hola desde la GPU! Soy el hilo número: 3
Hola desde la GPU! Soy el hilo número: 4
Hola desde la GPU! Soy el hilo número: 5
Hola desde la GPU! Soy el hilo número: 6
Hola desde la GPU! Soy el hilo número: 7
```

Los hilos salen ordenados del 0 al 7. Uno tiende a pensar que al lanzar varios hilos de forma simultánea terminarían de forma caótica, como ocurre por ejemplo en kotlin,  donde tenemos una instrucción `join()` para esperar a que todos acaben.

---

## 1.6 El orden en la GPU vs CPU (Kotlin/Java/C++ Threads)

**¿Por qué aparecen los mensajes ordenados en la consola?**

1. En **CPU** (como en Kotlin), el sistema operativo decide cuándo darle un "respiro" a un hilo y dárselo a otro. Es un caos total. 

2. En **CUDA**, la arquitectura es `SIMT (Single Instruction, Multiple Threads)`. Los hilos se lanzan en `grupos de 32 llamados Warps`. Como hemos lanzado solo 8 hilos, todos pertenecen al mismo Warp. Los hilos de un mismo Warp se ejecutan en SIMT (Single Instruction, Multiple Threads), es decir, ejecutan la misma instrucción exactamente al mismo tiempo.

3. **El Buffer de Salida**: Aunque la ejecución sea paralela, el printf dentro de un kernel de CUDA escribe en un buffer compartido. En programas tan pequeños y con tan pocos hilos, el hardware suele despachar el buffer al sistema operativo de forma secuencial por ID de hilo.

4. **Determinismo vs. Caos**: Si usted lanzara 1000 hilos (<<<1, 1000>>>) o repartiera el trabajo en varios bloques (<<<10, 10>>>), empezaría a ver que el orden se rompe. Los bloques pueden ejecutarse en diferentes SM (Streaming Multiprocessors) y ahí sí vería que el bloque 5 puede terminar antes que el bloque 0.

Una vez comprobado que la `CPU` y `GPU` se entienden bien y el programa `holaCuda.cu` funciona, veamos otro ejemplo.

---

## 1.7 Sumar vectores

El programa realiza una suma en paralelo de dos vectores de cinco componentes. Estos vectores se identifican en el código como:

```cpp
int h_a[N] = {1, 2, 3, 4, 5}; 	// h_ = Host (CPU); d_ = Dispositivo (GPU)
int h_b[N] = {10, 20, 30, 40, 50};
```

**SumarVectores_v1.0.cu**

```cpp
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

/*
    SALIDA del programa
    ===================

    Resultado de la suma en GPU:
    11 22 33 44 55 
*/

/*
    COMPILAR
    ========
    nvcc SumarVectores_v1.0.cu -o ./build/SumarVectores_1.0
    ./sumarVectores_1.0
*/
```

---

### sumarVectores.cu, de un vistazo

En este programa, el script `SumarVectores.cu` muestra la suma de vectores en paralelo mediante cuatro acciones clave: 

* El uso de la convención de nomenclatura `h_/d_` para diferenciar memoria (host/dispositivo)
* La reserva de `VRAM` con `cudaMalloc`
* La transferencia de datos vía `cudaMemcpy` (HostToDevice/DeviceToHost)
* Y el uso de `threadIdx.x` para asignar trabajo a cada hilo, concluyendo con `cudaFree` para liberar memoria. 

El código transfiere datos al dispositivo (GPU), ejecuta el kernel en paralelo sin bucles for explícitos en la GPU y recupera los resultados, ilustrando la `gestión básica de memoria y paralelismo` en CUDA.

### 📝 Análisis técnico de sumarVectores.cu

Lo más brillante de este código es que elimina el bucle for tradicional. En lugar de que un solo hilo sume N  veces, N hilos suman 1 vez.

Destaquemos lo siguiente:

1. `int i = threadIdx.x;`: Esta es la línea más importante del Kernel. Es el "DNI" del hilo. Si lanzamos 5 hilos, cada uno entra aquí con un valor de i distinto (0, 1, 2, 3, 4). Así, cada hilo sabe exactamente qué posición del array le toca procesar.

2. `cudaMalloc((void**)&d_a, ...)`: Aquí estamos reservando espacio en la VRAM de la RTX 4060 Ti. A diferencia de malloc en C++, aquí pasamos la dirección del puntero porque la función debe modificar su valor para que apunte a la dirección de memoria de la tarjeta. La GPU necesita escribir la dirección de memoria en el puntero del Host.


3. `cudaMemcpy(..., cudaMemcpyHostToDevice)`: Es el "puente". Estmos cruzando los datos por el bus PCIe desde la memoria RAM (CPU) a la memoria de la tarjeta de video. En aplicaciones reales, a veces el tiempo de cudaMemcpy supera al de cálculo si el dataset es pequeño. Por eso, habitualmente  buscamos minimizar los viajes de datos. Es el cuello de botella habitual en GPU, por eso es vital hacerlo bien. 

4. `sumarVectores<<<1, N>>>`: La configuración de lanzamiento. Al poner N (que es 5), le dices a la GPU: "Crea 5 hilos y dales a cada uno el código del kernel".

5. `cudaFree(d_a)`: Imprescindible. Al igual que con los punteros en C++, si no liberas la memoria de la GPU, se queda ocupada hasta que reinicies, lo que podría causar un "Memory Leak" en la tarjeta gráfica.

6. `Escalabilidad`: Al usar **threadIdx.x**, hemos sentado las bases del Paralelismo Masivo. Si mañana decidimos sumar 1.000.000 de elementos en lugar de 5, solo tendríamos que ajustar la configuración de bloques y mallas (Grid & Block), manteniendo el kernel prácticamente intacto.


---

# 2. Metodología Universal

Calificamos de Metodología Universal al conjunto de pasos secuenciales que daremos casi siempre en programas CUDA:

## 2.1 ⚙️ El Flujo de Trabajo en CUDA (Los 6 Pasos)

Para que la CPU y la GPU colaboren, siempre seguiremos este orden lógico:

1. **Reserva de Memoria (Allocation)**: Preparamos el terreno en la tarjeta de vídeo reservando espacio en su VRAM mediante `cudaMalloc`. Es el equivalente al new de C++, pero para el dispositivo.

2. **Transferencia de Datos (Host to Device)**: Copiamos la información necesaria (vectores, matrices o imágenes) desde nuestra memoria RAM (CPU) hacia la VRAM (GPU) usando `cudaMemcpy` con el flag `cudaMemcpyHostToDevice`.

3. **Configuración y Lanzamiento (Kernel Launch)**: Invocamos la función **__global__** usando la sintaxis de los "triples bracket"s **<<<bloques, hilos>>>**. Aquí decidimos cuántos obreros (hilos) trabajarán en paralelo.

4. **Sincronización (Opcional pero Crítico)**: Si el kernel imprime mensajes o si la CPU necesita los datos inmediatamente, usamos `cudaDeviceSynchronize()`. Esto obliga al "jefe" (CPU) a esperar a que todos los "obreros" (GPU) terminen su tarea.

5. **Recogida de Resultados (Device to Host)**: Traemos el trabajo terminado de vuelta a la CPU mediante `cudaMemcp`y con el flag `cudaMemcpyDeviceToHost`.

6. Limpieza (Free): Liberamos la memoria utilizada en la GPU con `cudaFree`. Mantener la VRAM limpia es vital para evitar bloqueos en la tarjeta gráfica.