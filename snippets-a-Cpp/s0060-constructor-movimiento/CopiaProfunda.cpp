#include <iostream>

/**
 * SNIPPET #0040 (C): El Peligro de la Copia Superficial (Shallow Copy).
 * 
 * CONCEPTO: Demostrar cómo el fallo catastrófico que ocurre cuando dos objetos 
 * en la Pila comparten el mismo puntero al Heap (sin gestión de referencias),
 * se evita mediante la "Copia Profunda".
 */

class GestorPeligroso {
public:
    int* datos;
    
    // 1. CONSTRUCTOR: Reserva memoria original en el Heap
    GestorPeligroso() {
        datos = new int[10]; 
        for(int i=0; i<10; i++) datos[i] = i; // Inicializamos con basura o datos
        std::cout << "[NACE] Objeto original. Memoria en: " << (void*)datos << std::endl;
    }

    // 2. CONSTRUCTOR DE COPIA (LA SOLUCIÓN):
    // Sin este bloque, C++ copiaría el puntero (Shallow Copy) y el programa "petaría".
    GestorPeligroso(const GestorPeligroso& otro) {
        datos = new int[10]; // PEDIMOS NUEVA MEMORIA propia (Deep Copy)
        for(int i=0; i<10; i++) {
            datos[i] = otro.datos[i]; // Clonamos el contenido, no la dirección
        }
        std::cout << "[COPIA PROFUNDA] Nuevo bloque creado en: " << (void*)datos << std::endl;
    }

    // 3. DESTRUCTOR: Libera la memoria
    ~GestorPeligroso() {
        std::cout << "[MUERE] Intentando liberar Heap en: " << (void*)datos << std::endl;
        // Si no hubiera Copia Profunda, el segundo objeto en morir intentaría
        // borrar memoria ya borrada (Double Free Error).
        delete[] datos; 
    }
};

void demostracion() {
    // A. Nace objetoA y reserva su parcela de memoria.
    GestorPeligroso objetoA; 
    
    // B. Al asignar así, se invoca nuestro "Constructor de Copia".
    // Ahora objetoB tiene SU PROPIA dirección de memoria, independiente de A.
    GestorPeligroso objetoB = objetoA; 
    
    std::cout << "Direccion en A: " << (void*)objetoA.datos << std::endl;
    std::cout << "Direccion en B: " << (void*)objetoB.datos << std::endl;

} // C. FINAL DEL ÁMBITO: 
  // - Muere objetoB: Ejecuta delete[] sobre su memoria.
  // - Muere objetoA: Ejecuta delete[] sobre SU memoria. 
  // ¡Todo en orden! El programa no explota.

int main() {
    std::cout << "--- Inicio del Experimento ---" << std::endl;
    
    demostracion();
    
    std::cout << "--- Fin del experimento (Ejecucion Segura) ---" << std::endl;
    return 0;
}

/**
 * COMENTARIOS TÉCNICOS
 * ===================
 * 1. EL RIESGO: Si comentaras las líneas 21-27, el compilador haría una copia "bit a bit".
 *    objetoA.datos y objetoB.datos valdrían lo mismo (apuntarían al mismo sitio).
 * 
 * 2. EL CRASH: El "Double Free" ocurre porque el sistema operativo no permite que un 
 *    proceso libere dos veces la misma dirección de memoria dinámica.
 * 
 * 3. REGLA DE ORO: En C++ clásico, si una clase gestiona recursos (memoria, archivos, 
 *    sockets), debe implementar el "Big Three": Destructor, Copia y Asignación.
 * 
 * 4. RECOMENDACIÓN: Para evitar este código manual ("boilerplate"), la documentación 
 *    de [cppreference sobre RAII](https://en.cppreference.com) 
 *    sugiere usar siempre contenedores estándar o Smart Pointers.
 */

 // Compilar
 // ========
 // g++ CopiaProfunda.cpp -o ./build/CopiaProfunda


