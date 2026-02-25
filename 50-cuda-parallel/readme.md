# 🚀Bloque 50: CUDA Parallel Computing
En este itinerario, el lector aprenderá que la GPU no es solo para jugar, sino un monstruo de cálculo que puede procesar miles de datos a la vez. El menú de aprendizaje es el siguiente:


---

1. **s5110-instalacion-del-entorno-cuda**
**Antes de correr**, hay que empezar a andar. Aquí documentaremos:
* **Drivers de NVIDIA**: El puente necesario para que el SO vea la GPU.
* **CUDA Toolkit:** El compilador nvcc y las librerías base.
* **Integración**: Cómo configurar Visual Studio o VS Code para que entiendan los archivos .cu. 

---

2. **s5130-ejemplos-cuda (El laboratorio de potencia)**
Aquí es donde el código se vuelve "mágico". En lugar de un bucle for que suma uno a uno, lanzaremos miles de hilos que suman todo a la vez.
* **hola-cuda.cu**: El primer contacto con el kernel (__global__).
* **SumarVectores_v1.x.cu**: La base del paralelismo. Dividimos un vector en trozos y cada hilo suma su posición.
* **SumarMatrices.cu y ProductoMatrices_v1.x.cu**: Operaciones 2D donde la GPU brilla de verdad.
* **RegresionLineal_v1.0.cu**: Un ejemplo real de estadística acelerada por hardware. 


