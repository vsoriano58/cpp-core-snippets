# 🚀 Suma de Vectores en CUDA: De la Teoría a la Escala Real

Este repositorio contiene la progresión técnica del aprendizaje de computación paralela utilizando el `NVIDIA CUDA Toolkit`. Se presentan dos versiones que ilustran desde el concepto básico hasta la gestión de mallas masivas.

## 📁 Contenido del Directorio

### 1. `SumarVectores_v1.0.cu` (Concepto Didáctico)

Es la implementación mínima para entender la comunicación entre CPU (Host) y GPU (Device).

* **Enfoque**: Educativo y de depuración.
* **Configuración**: 1 solo bloque con 5 hilos.
* **Clave Técnica**: Introducción a la nomenclatura `h_` vs `d_` y el uso de `threadIdx.x`.
* **Ideal para**: Confirmar que el entorno CUDA está correctamente instalado.

### 2. `SumarVectores_v1.1.cu` (Escala Industrial)

Evolución profesional que gestiona un millón de datos (`float`) y utiliza una arquitectura de red de hilos.

* **Enfoque**: Rendimiento y escalabilidad.
* **Configuración Dinámica**: Cálculo automático de bloques y hilos mediante `(N + threadsPerBlock - 1) / threadsPerBlock`.
* **Cálculo de Índice Global**: Uso de la fórmula `blockIdx.x * blockDim.x + threadIdx.x` para mapear hilos en múltiples bloques.
* **Seguridad**: Incluye verificación de límites (`if (i < N)`) para evitar desbordamientos de memoria en la GPU.

---

## ⚙️ El Flujo Universal de Trabajo (Las 5 Etapas)

Ambos programas respetan el ciclo de vida estándar de una aplicación en la Arquitectura CUDA:

1. **Host Allocation**: Reserva de memoria RAM para los datos iniciales.
2. **Device Allocation**: Reserva de VRAM en la GPU mediante `cudaMalloc`.
3. **Memcpy (H2D)**: Transferencia de datos de la CPU a la GPU a través del bus PCIe mediante `cudaMemcpy (HostToDevice)`.
4. **Kernel Launch**: Ejecución paralela masiva en los núcleos de la tarjeta.
5. **Memcpy (D2H)**: Recuperación de los resultados procesados hacia la CPU.

---

## 🛠️ Compilación y Ejecución

Para compilar cualquiera de los dos archivos, utilice el compilador nvcc incluido en su instalación de drivers.

```bash
# Compilar Versión 1.0
nvcc SumarVectores_v1.0.cu -o ./build/SumarVectores_v1.0

# Compilar Versión 1.1
nvcc SumarVectores_v1.1.cu -o ./build/SumarVectores_v1.1
```

---

## 📊 Comparativa Técnica


| Caracteristica | v1.0 (Básica) | v1.1 (Avanzada) |
| :--- | :--- | :--- |
| **Tipo de dato** | `int` | `float` |
| **Tamaño (N)** | 5 | 1.000.000 |
| **Jerarquía** | Bloque único | Malla de bloques (Grid) |
| **Índice** | Local ( `threadIdx.x` ) | Global (Bloque + Hilo) |
| **Memoria Host** | Estática (Stack) | Dinámica ( `malloc` ) |