#include <iostream>
#include <memory> // REQUISITO: Para usar Smart Pointers

/**
 * @file unique_ptr.cpp
 * @brief El fin de la limpieza manual con std::unique_ptr.
 * @author alcón68
 * 
 * @details Demostración de RAII moderno. unique_ptr garantiza que un recurso 
 * tenga un ÚNICO dueño, eliminando fugas de memoria y eliminando la 
 * necesidad de llamar a 'delete' manualmente.
 */

class RecursoSeguro {
public:
    RecursoSeguro() { std::cout << "[NACE] Recurso asignado en el Heap.\n"; }
    ~RecursoSeguro() { std::cout << "[MUERE] Limpieza automática por Smart Pointer.\n"; }
    
    void trabajar() { std::cout << "[INFO] Trabajando con datos protegidos...\n"; }
};

void demostracionRAII() {
    // 1. CREACIÓN MODERNA (C++14): 
    // std::make_unique es preferible a 'new' porque es más seguro frente a excepciones.
    std::unique_ptr<RecursoSeguro> ptr1 = std::make_unique<RecursoSeguro>();

    ptr1->trabajar();

    // 2. PROTECCIÓN CONTRA COPIA: 
    // El desastre del snippet s0050 es imposible aquí. El compilador prohíbe la copia.
    // std::unique_ptr<RecursoSeguro> copia = ptr1; // <-- ERROR: No se permite duplicar el dueño.

    // 3. TRANSFERENCIA DE PROPIEDAD (Semántica de Movimiento):
    // Movemos el recurso de ptr1 a ptr2. ptr1 quedará invalidado (nullptr).
    std::unique_ptr<RecursoSeguro> ptr2 = std::move(ptr1); 
    
    if (!ptr1) {
        std::cout << "[ESTADO] ptr1 ahora es null. La propiedad pasó a ptr2.\n";
    }

    if (ptr2) {
        ptr2->trabajar();
    }

} // 4. FINAL DEL ÁMBITO: 
  // Al salir de la función, ptr2 se destruye y libera el Heap AUTOMÁTICAMENTE.
  // No hay riesgo de "Double Free" ni de "Memory Leak".

int main() {
    std::cout << "--- Inicio de Gestión Moderna (Smart Pointers) ---" << std::endl;
    
    demostracionRAII();
    
    std::cout << "--- Fin (Memoria gestionada correctamente) ---" << std::endl;
    return 0;
}

/**
 * ANÁLISIS TÉCNICO:
 * 1. PROPIEDAD EXCLUSIVA: [std::unique_ptr](https://en.cppreference.com) 
 *    implementa internamente el Constructor de Movimiento y elimina el de Copia.
 * 
 * 2. RENDIMIENTO: Tiene "Zero Overhead". El ejecutable resultante es idéntico 
 *    en velocidad al uso de punteros crudos (*), pero con seguridad garantizada.
 * 
 * 3. SEGURIDAD EXCEPCIONAL: Si el programa lanza una excepción, el Smart Pointer 
 *    libera la memoria igualmente, algo que con 'delete' manual es difícil de asegurar.
 * 
 * 4. CONSEJO: Usa siempre `std::make_unique` (C++14) para evitar fugas potenciales 
 *    durante la construcción de objetos complejos.
 */

// Compilar: g++ unique_ptr.cpp -o ./build/unique_ptr
