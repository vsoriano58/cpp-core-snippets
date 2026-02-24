#include <iostream>

/**
 * @file PeligroCopiaSuperficial2.cpp
 * @brief Solución al problema de gestión de memoria mediante Copia Profunda.
 * @author alcón68
 * 
 * @details Implementa el Constructor de Copia para asegurar que cada objeto
 * gestione su propio ciclo de vida de memoria de forma independiente.
 */

class GestorPeligroso {
public:
    int* datos;

    /** @brief Constructor base: Reserva memoria propia. */
    GestorPeligroso() {
        datos = new int[10];
        std::cout << "[NACE] Memoria original en: " << (void*)datos << std::endl;
    }
    
    /**
     * @brief CONSTRUCTOR DE COPIA (Deep Copy).
     * @param otro Referencia constante al objeto origen.
     * @details En lugar de copiar el puntero, se reserva un nuevo bloque de 
     * memoria y se clona el contenido valor por valor.
     */
    GestorPeligroso(const GestorPeligroso& otro) {
        datos = new int[10]; // Nueva reserva en el Heap para el nuevo objeto
        for(int i = 0; i < 10; i++) {
            datos[i] = otro.datos[i]; // Clonación de datos
        }
        std::cout << "[COPIA PROFUNDA] Memoria independiente en: " << (void*)datos << std::endl;
    }

    /** @brief Destructor: Cada objeto libera su propio bloque sin conflictos. */
    ~GestorPeligroso() {
        std::cout << "[MUERE] Liberando bloque en: " << (void*)datos << std::endl;
        delete[] datos; 
    }
};

void operacionSegura() {
    GestorPeligroso objetoA; 
    
    // Se invoca el constructor de copia personalizado.
    GestorPeligroso objetoB = objetoA; 
    
    std::cout << "Dirección A: " << (void*)objetoA.datos << std::endl;
    std::cout << "Dirección B: " << (void*)objetoB.datos << std::endl;

} // Al salir, ambos objetos se destruyen limpiamente.

int main() {
    std::cout << "--- Inicio del Experimento: Implementación RAII ---" << std::endl;
    
    operacionSegura();
    
    std::cout << "--- Éxito: Gestión de memoria segura y determinista ---" << std::endl;
    return 0;
}

/**
 * RECOMENDACIONES DE INGENIERÍA:
 * 1. REGLA DE LOS TRES: Si defines un Destructor, debes definir el Constructor de 
 *    Copia y el Operador de Asignación. Consulta la [Regla de los Tres en C++](https://en.cppreference.com).
 * 2. SMART POINTERS: En C++ moderno, se recomienda usar `std::unique_ptr` o `std::shared_ptr` 
 *    para evitar la gestión manual de `new` y `delete`.
 * 3. ABSTRACCIÓN: El uso de [std::vector](https://en.cppreference.com) 
 *    automatiza todo este proceso de copia profunda de forma eficiente.
 */

 // Compilar
 // ========
 // g++ PeligroCopiaSuperficial2.cpp -o ./build/PeligroCopiaSuperficial2
