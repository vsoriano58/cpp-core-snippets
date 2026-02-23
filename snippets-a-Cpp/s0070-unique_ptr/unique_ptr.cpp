#include <iostream>
#include <memory> // REQUISITO: Para usar Smart Pointers

/**
 * SNIPPET #0070: El fin de la Limpieza Manual (std::unique_ptr).
 * 
 * CONCEPTO: Demostrar cómo el RAII (Resource Acquisition Is Initialization)
 * elimina la necesidad de destructores manuales complejos y evita fugas 
 * de memoria por diseño.
 */

class RecursoSeguro {
public:
    RecursoSeguro() { std::cout << "[NACE] Recurso asignado en el Heap.\n"; }
    ~RecursoSeguro() { std::cout << "[MUERE] Limpieza automatica por Smart Pointer.\n"; }
    
    void saludar() { std::cout << "Trabajando con datos seguros...\n"; }
};

void demostracionRAII() {
    // 1. CREACIÓN: std::make_unique es la forma segura y eficiente (C++14).
    // No usamos 'new', por lo tanto, no buscaremos un 'delete'.
    std::unique_ptr<RecursoSeguro> ptr = std::make_unique<RecursoSeguro>();

    ptr->saludar();

    // 2. SEGURIDAD DE COPIA: 
    // unique_ptr NO se puede copiar. Esto evita el desastre del Snippet #0040.
    // std::unique_ptr<RecursoSeguro> ptr2 = ptr; // <-- ERROR DE COMPILACIÓN (Protección)

    // 3. TRANSFERENCIA DE PROPIEDAD:
    // Si queremos moverlo, debemos ser explícitos.
    std::unique_ptr<RecursoSeguro> ptr2 = std::move(ptr); 
    
    if (!ptr) {
        std::cout << "[INFO] ptr ahora es null, la propiedad paso a ptr2.\n";
    }

} // 4. FINAL DEL ÁMBITO: 
  // ptr2 sale de alcance y libera la memoria AUTOMÁTICAMENTE. 
  // No hay riesgo de olvido ni de doble liberación.

int main() {
    std::cout << "--- Inicio de Gestion Moderna ---" << std::endl;
    
    demostracionRAII();
    
    std::cout << "--- Fin (Sin fugas de memoria detectadas) ---" << std::endl;
    return 0;
}

/**
 * COMENTARIOS TÉCNICOS
 * ===================
 * 1. PROPIEDAD ÚNICA: [std::unique_ptr](https://en.cppreference.com) garantiza 
 *    que solo un puntero sea dueño del objeto en el Heap en un momento dado.
 * 
 * 2. CERO OVERHEAD: En tiempo de ejecución, un `unique_ptr` es tan rápido como un puntero 
 *    crudo (`*`). No consume memoria extra ni ciclos de CPU adicionales.
 * 
 * 3. ADIÓS AL DELETE: El destructor del smart pointer llama automáticamente a `delete` 
 *    cuando el objeto sale del ámbito (scope), incluso si ocurre una excepción inesperada.
 * 
 * 4. SEMÁNTICA DE MOVIMIENTO: Al usar `std::move`, transferimos el control sin copiar 
 *    los datos, lo cual es extremadamente eficiente para objetos pesados.
 */

 // Compilar
 // ========
 // g++ unique_ptr.cpp -o ./build/unique_ptr
