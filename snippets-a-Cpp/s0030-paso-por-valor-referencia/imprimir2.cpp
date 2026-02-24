#include <iostream>

/**
 * @title: Paso por Referencia (La Llave)
 * @description: Evidencia como al trabajar sobre la dirección original permite 
 * mutar datos y ahorrar recursos de memoria.
 */

// [REF-004] El símbolo '&' indica que 'x' es un alias de la variable original.
void cambiarRealmente(int& x) {
    x = 99; // [REF-005] Estamos escribiendo directamente en la celda de memoria original.
}

int main() {

// [REF-006] El puntero 'num' vive en la Pila y guarda la dirección del dato que reside en el Heap.

    int *num = new int(10);

    std::cout << "Valor inicial: " << *num << std::endl;    // 10

    // [REF-007] Usamos '*' para extraer al paciente y pasárselo a la función.
    cambiarRealmente(*num);

    std::cout << "Valor mutado:  " << *num << " (Cambio persistente)" << std::endl;     // 99

    delete num; 
    return 0;
}

/**
 * GPS DEL PROGRAMA [REF]
 * ======================
 * [REF-004] ALIAS DE MEMORIA: No se crea una nueva variable. 'x' es simplemente 
 *           otro nombre para la misma dirección de memoria que le pasemos.
 * [REF-005] MUTACIÓN DIRECCIONADA: Cualquier cambio aquí es un "golpe de remo" 
 *           en el barco original. No hay marcha atrás.
 * [REF-006] El puntero 'num' se aloja en la Pila y contiene la dirección del dato real, 
 *           que vive en el Heap.
 * [REF-007] DESREFERENCIACIÓN: '*num' es la forma de decir "no me des la dirección, 
 *           dame lo que hay dentro, lo que vive en el Heap".
 */

 // Compilar
 // ========
 // g++ imprimir2.cpp -o ./build/imprimir2
