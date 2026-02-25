# 📝Adelanto para el README de s5130
El concepto clave que explicaremos es el Flujo de Trabajo CUDA: 
1. **Host (CPU)**: Prepara los datos en la RAM.
2. **Transferencia**: Copiamos los datos a la VRAM de la GPU (cudaMemcpyHostToDevice).
3. **Kernel**: Lanzamos la función en la GPU con la sintaxis <<<bloques, hilos>>>.
4. **Recogida**: Traemos el resultado de vuelta a la CPU (cudaMemcpyDeviceToHost).

