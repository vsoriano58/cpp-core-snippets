#include <iostream>
#include <memory> // REQUISITO: Para la familia de Smart Pointers

/**
 * @file shared_ptr.cpp
 * @brief Gestión de memoria mediante Conteo de Referencias (Reference Counting).
 * @author alcón68
 * 
 * CONCEPTO: A diferencia de unique_ptr, el recurso en el Heap tiene múltiples 
 * dueños. El objeto solo se destruye cuando el contador de referencias llega a 0.
 */

class RecursoCompartido {
public:
    RecursoCompartido() { std::cout << "[NACE] Recurso creado en el Heap.\n"; }
    ~RecursoCompartido() { std::cout << "[MUERE] Contador llegó a 0. Memoria liberada.\n"; }
    
    void accion() { std::cout << "[INFO] Usando recurso compartido...\n"; }
};

void demostracionShared() {
    // 1. CREACIÓN: std::make_shared reserva el objeto y el "Bloque de Control" 
    // en una sola asignación (más eficiente y seguro). Contador = 1.
    std::shared_ptr<RecursoCompartido> p1 = std::make_shared<RecursoCompartido>();
    
    std::cout << "[ESTADO] p1 creado. Referencias: " << p1.use_count() << std::endl;

    {
        std::cout << "\n--- Entrando en ámbito interno ---" << std::endl;
        
        // 2. COPIA PERMITIDA: Al copiar p1 en p2, ambos comparten la propiedad.
        // El bloque de control interno incrementa el contador a 2.
        std::shared_ptr<RecursoCompartido> p2 = p1; 
        
        std::cout << "[COPIA] p2 creado. Referencias totales: " << p1.use_count() << std::endl;
        p2->accion();
        
        std::cout << "--- Saliendo del ámbito interno (muere p2) ---" << std::endl;
        
    } // 3. P2 SALE DE ÁMBITO: El contador baja de 2 a 1. 
      // El recurso NO se libera porque p1 todavía lo mantiene vivo.
    
    std::cout << "\n[ESTADO] p2 ha muerto. Referencias actuales: " << p1.use_count() << std::endl;
    p1->accion();

} // 4. FINAL DEL ÁMBITO: 
  // p1 sale de alcance, el contador llega a 0 y el destructor se ejecuta automáticamente.

int main() {
    std::cout << "--- Inicio de Propiedad Compartida (std::shared_ptr) ---" << std::endl;
    
    demostracionShared();
    
    std::cout << "--- Fin del programa (Limpieza garantizada) ---" << std::endl;
    return 0;
}

/**
 * ANÁLISIS TÉCNICO:
 * 1. EL BLOQUE DE CONTROL: [std::shared_ptr](https://en.cppreference.com) 
 *    gestiona un puntero al objeto y otro al contador de forma atómica (seguro para hilos).
 * 
 * 2. SOBRECOSTE (Overhead): A diferencia de `unique_ptr`, tiene un ligero impacto 
 *    en rendimiento debido a la gestión del contador y la indirección doble.
 * 
 * 3. MAKE_SHARED: Usa siempre [std::make_shared](https://en.cppreference.com/make_shared) 
 *    para evitar fragmentación de memoria y mejorar la localidad de datos.
 * 
 * 4. EL TALÓN DE AQUILES: Las referencias circulares (Objeto A apunta a B y B a A). 
 *    Esto causa Memory Leaks porque el contador nunca llega a 0. Se soluciona con 
 *    `std::weak_ptr`.
 */

// Compilar: g++ shared_ptr.cpp -o ./build/shared_ptr
