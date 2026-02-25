#include <iostream>

/**
 * @title: El Peligro de la Copia Superficial (Shallow Copy). 
 * @decription: Demostración de colapso de memoria por Copia Superficial (Shallow Copy).
 * Ocurre cuando dos objetos en la Pila comparten el mismo puntero al Heap sin un contador
 * de referencias.
 */

class GestorPeligroso {
public:
    int* datos;
    
    GestorPeligroso() {
        datos = new int[10]; // Asignación de recursos en el Heap
        // El casting (void) es para que 'cout' imprima la dirección.
        // Si ponemos solo 'datos' imprimiria el entero al que apunta 'datos'
        std::cout << "[NACE] Objeto creado. Memoria Heap en: " << (void*)datos << std::endl;
    }

    ~GestorPeligroso() {
        std::cout << "[MUERE] Intentando liberar Heap en: " << (void*)datos << std::endl;
        // El problema: si ya fue liberado por otro objeto, aquí el programa "falla".
        delete[] datos; 
    }
};

void causarDesastre() {
    // 1. Nace objetoA y reserva memoria.
    GestorPeligroso objetoA; 
    
    // 2. EL ERROR: Copia por defecto (Shallow Copy).
    // C++ copia el valor del puntero 'datos' bit a bit. 
    // Ahora objetoB.datos apunta a la MISMA dirección que objetoA.datos.
    GestorPeligroso objetoB = objetoA; 
    
    std::cout << "Puntero A: " << (void*)objetoA.datos << std::endl;
    std::cout << "Puntero B: " << (void*)objetoB.datos << std::endl;

} // 3. FINAL DEL ÁMBITO: [REF-01]
  // - Primero muere objetoB (último creado): Llama a delete[] y el Heap queda libre.
  // - Luego muere objetoA: ¡Intenta llamar a delete[] sobre memoria YA LIBERADA! 
  //   Esto se llama "Double Free Error" y suele colapsar el programa.

int main() {
    std::cout << "--- Inicio del Experimento Peligroso ---" << std::endl;
    
    // Ejecutamos en un entorno controlado (función aparte)
    causarDesastre();
    
    std::cout << "--- Fin del experimento (Si ves esto, el S.O. fue indulgente) ---" << std::endl;
    return 0;
}

/**
 * ANALISIS TÉCNICO:
 * 1. PROPIEDAD AMBIGUA: C++ no sabe por defecto quién es el "dueño" del recurso en el Heap.
 * 2. EXCEPCIÓN DE LIBERACIÓN: El [Double Free Error](https://cwe.mitre.org) 
 *    es una vulnerabilidad crítica de seguridad y estabilidad.
 * 3. DIAGNÓSTICO: Al no declarar un Constructor de Copia, el compilador genera uno 
 *    que simplemente copia el valor del puntero, no el contenido.
 * 4. La solución la veremos en 'PeligroCopiaSuperficial.cpp'
 */

 // Compilar
 // ========
 // g++ PeligroCopiaSuperficial.cpp -o ./build/PeligroCopiaSuperficial
