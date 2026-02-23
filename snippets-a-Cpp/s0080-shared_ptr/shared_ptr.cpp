#include <iostream>
#include <memory> // REQUISITO: Para la familia de Smart Pointers

/**
 * SNIPPET #0080: Propiedad Compartida (std::shared_ptr).
 * 
 * CONCEPTO: Gestión de memoria mediante "Contador de Referencias". 
 * El objeto en el Heap solo se destruye cuando el ÚLTIMO puntero que lo
 * apuntaba deja de existir.
 */

class RecursoCompartido {
public:
    RecursoCompartido() { std::cout << "[NACE] Recurso creado en el Heap.\n"; }
    ~RecursoCompartido() { std::cout << "[MUERE] Contador llego a 0. Memoria liberada.\n"; }
};

void demostracionShared() {
    // 1. CREACIÓN: Se crea el objeto y el "Bloque de Control" (contador = 1).
    std::shared_ptr<RecursoCompartido> p1 = std::make_shared<RecursoCompartido>();
    
    {
        // 2. COPIA PERMITIDA: A diferencia de unique_ptr, aquí sí podemos copiar.
        // Al copiar, el contador de referencias sube a 2.
        std::shared_ptr<RecursoCompartido> p2 = p1; 
        
        std::cout << "[INFO] Referencias actuales: " << p1.use_count() << std::endl;
        
    } // 3. P2 SALE DE ÁMBITO: El contador baja a 1, pero NO se libera la memoria.
    
    std::cout << "[INFO] p2 ha muerto. Referencias: " << p1.use_count() << std::endl;
    std::cout << "[INFO] El recurso sigue vivo porque p1 aun lo mantiene.\n";

} // 4. FINAL DEL ÁMBITO: 
  // p1 muere, el contador llega a 0 y ¡BUM!, el destructor se ejecuta por fin.

int main() {
    std::cout << "--- Inicio de Propiedad Compartida ---" << std::endl;
    
    demostracionShared();
    
    std::cout << "--- Fin del programa ---" << std::endl;
    return 0;
}

/**
 * COMENTARIOS TÉCNICOS
 * ===================
 * 1. EL CONTADOR (Reference Counting): [std::shared_ptr](https://en.cppreference.com) 
 *    utiliza un bloque de control interno que rastrea cuántos dueños tiene el objeto.
 * 
 * 2. SEGURIDAD: Evita el error de "Puntero Colgante" (Dangling Pointer), ya que es 
 *    imposible que el objeto desaparezca si todavía tienes un shared_ptr hacia él.
 * 
 * 3. SOBRECOSTE (Overhead): A diferencia de unique_ptr, este tiene un pequeño coste 
 *    de rendimiento porque el contador debe actualizarse de forma atómica (Thread-safe).
 * 
 * 4. EL PELIGRO (Ciclos): Si dos objetos tienen un shared_ptr el uno hacia el otro, 
 *    el contador nunca llegará a 0 (Memory Leak). Para solucionar esto se usa:
 *    [std::weak_ptr](https://en.cppreference.com).
 */

 // Compilar
 // ========
 // g++ shared_ptr.cpp -o ./build/shared_ptr
