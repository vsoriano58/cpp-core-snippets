#include <iostream>
#include <memory>
#include <string>

/**
 * @file weak_ptr.cpp
 * @brief El Observador Débil: Rompiendo Ciclos de Referencia.
 * @author alcón68
 * 
 * CONCEPTO: Un weak_ptr "observa" un recurso gestionado por shared_ptr 
 * SIN aumentar el contador de referencias. Es la cura para el "Ciclo de la Muerte".
 */

class Persona {
public:
    std::string nombre;
    // Si usáramos shared_ptr aquí, crearíamos una referencia circular.
    // weak_ptr dice: "Sé quién es, pero no soy su dueño".
    std::weak_ptr<Persona> amigo; 

    Persona(std::string n) : nombre(n) { 
        std::cout << "[NACE] " << nombre << " ha entrado en escena.\n"; 
    }
    
    ~Persona() { 
        std::cout << "[MUERE] " << nombre << " ha sido liberado del Heap.\n"; 
    }

    /**
     * @brief Intenta interactuar con el recurso observado.
     * @details Como el objeto puede haber muerto, debemos "bloquearlo" (lock)
     * para obtener un shared_ptr temporal y seguro.
     */
    void saludarAmigo() {
        if (auto compartido = amigo.lock()) {
            std::cout << "  > " << nombre << " dice: ¡Hola, " << compartido->nombre << "!\n";
        } else {
            std::cout << "  > " << nombre << " dice: Mi amigo ya no existe (puntero expirado).\n";
        }
    }
};

void demostracionCiclo() {
    // 1. CREACIÓN: Dos dueños independientes.
    auto paco = std::make_shared<Persona>("Paco");
    auto maria = std::make_shared<Persona>("Maria");

    // 2. CONEXIÓN CRUZADA: 
    // Paco observa a Maria y Maria observa a Paco.
    // Al ser weak_ptr, el use_count() de ambos SIGUE SIENDO 1.
    paco->amigo = maria;
    maria->amigo = paco;

    std::cout << "[ESTADO] Referencias de Paco: " << paco.use_count() << "\n";
    paco->saludarAmigo();

} // 3. FINAL DEL ÁMBITO: 
  // Al salir, los contadores bajan a 0. Ambos mueren correctamente.
  // Si 'amigo' fuera shared_ptr, el contador sería 2 y NUNCA morirían.

int main() {
    std::cout << "--- Inicio: Rompiendo el Ciclo de la Muerte ---" << std::endl;
    
    demostracionCiclo();
    
    std::cout << "--- Fin (Memoria limpia y sin ciclos) ---" << std::endl;
    return 0;
}

/**
 * ANÁLISIS TÉCNICO:
 * 1. EL BLOQUEO (Lock): El método [lock()](https://en.cppreference.com) 
 *    es atómico. Crea un shared_ptr temporal si el objeto existe.
 * 
 * 2. EXPIRACIÓN: Puedes usar `amigo.expired()` para verificar si el objeto 
 *    ha sido borrado sin intentar acceder a él.
 * 
 * 3. CASOS DE USO: Crucial en caches de datos, sistemas de observadores (Observer Pattern) 
 *    y estructuras de datos cíclicas como grafos o árboles con punteros al padre.
 * 
 * 4. SEGURIDAD: Evita el "Dangling Pointer" (puntero colgado) porque el sistema 
 *    sabe si la memoria a la que apunta el weak_ptr ha sido liberada.
 */

// Compilar: g++ weak_ptr.cpp -o ./build/weak_ptr
