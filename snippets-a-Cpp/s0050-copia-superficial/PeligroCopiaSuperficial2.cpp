#include <iostream>

/**
 * SNIPPET #0040 (C): El Peligro de la Copia Superficial (Shallow Copy).
 * 
 * CONCEPTO: Demostrar cómo, el fallo catastrófico que ocurre cuando dos objetos 
 * en la Pila comparten el mismo puntero al Heap sin un contador de referencias,
 * se puede evitar.
 */

class GestorPeligroso {
public:
    int* datos;

    // Constructor base
    GestorPeligroso() {
        datos = new int[10];
        std::cout << "[NACE] Memoria reservada en: " << (void*)datos << std::endl;
    }
    
    /**
     * LA SOLUCIÓN: Constructor de Copia Personalizado.
     * En lugar de copiar la dirección (shallow), reservamos nuevo espacio (deep).
     */
    GestorPeligroso(const GestorPeligroso& otro) {
        datos = new int[10]; // PEDIMOS NUEVA MEMORIA propia
        for(int i=0; i<10; i++) {
            datos[i] = otro.datos[i]; // Clonamos el contenido, no la dirección
        }
        std::cout << "[COPIA PROFUNDA] Nueva memoria en: " << (void*)datos << std::endl;
    }

    ~GestorPeligroso() {
        std::cout << "[MUERE] Intentando liberar Heap en: " << (void*)datos << std::endl;
        // Gracias a la copia profunda, cada objeto libera su propio bloque.
        delete[] datos; 
    }
};

void causarDesastre() {
    // 1. Nace objetoA y reserva su propia memoria.
    GestorPeligroso objetoA; 
    
    // 2. LA SOLUCIÓN EN ACCIÓN: 
    // Al haber definido un constructor de copia, C++ ya no usa la copia bit a bit.
    // objetoB tendrá su propia dirección de memoria con los mismos datos.
    GestorPeligroso objetoB = objetoA; 
    
    std::cout << "Puntero A: " << (void*)objetoA.datos << std::endl;
    std::cout << "Puntero B: " << (void*)objetoB.datos << std::endl;

} // 3. FINAL DEL ÁMBITO: 
  // - Muere objetoB: Libera su bloque de memoria sin problemas.
  // - Muere objetoA: Libera SU bloque independiente. 
  // No hay "Double Free" porque las direcciones son distintas.

int main() {
    std::cout << "--- Inicio del Experimento Controlado ---" << std::endl;
    
    causarDesastre();
    
    std::cout << "--- Fin del experimento (Éxito total por Copia Profunda) ---" << std::endl;
    return 0;
}

/**
 * COMENTARIOS
 * ===========
 * 1. LA TRAMPA: Por defecto, C++ realiza una "Copia de Miembros". Si un miembro es
 *    un puntero, se copia la dirección, creando un alias peligroso en lugar de un duplicado.
 * 
 * 2. EL CONFLICTO DE PROPIEDAD: Sin el constructor de copia personalizado, dos destructores
 *    creen ser "dueños" de la misma memoria en el Heap. El primero en morir tiene éxito;
 *    el segundo corrompe el programa (Undefined Behavior).
 * 
 * 3. LA REGLA DE TRES/CINCO: Este snippet demuestra que si necesitas un Destructor 
 *    personalizado para liberar memoria, OBLIGATORIAMENTE necesitas un Constructor de Copia 
 *    y un Operador de Asignación para gestionar dicha copia.
 * 
 * 4. MODERNIZACIÓN: En el C++ real, usar [std::vector](https://en.cppreference.com) 
 *    evita todo este código manual, ya que el contenedor ya implementa la Copia Profunda por ti.
 */

 // Compilar
 // ========
 // g++ PeligroCopiaSuperficial2.cpp -o ./build/PeligroCopiaSuperficial2
