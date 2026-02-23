#include <iostream>
#include <memory>

/**
 * SNIPPET #0090: El Observador Débil (std::weak_ptr).
 * 
 * CONCEPTO: Romper ciclos de referencia. Un weak_ptr "observa" un recurso
 * gestionado por shared_ptr pero SIN aumentar el contador de referencias.
 * Evita que dos objetos se mantengan vivos el uno al otro eternamente.
 */

class Persona {
public:
    std::string nombre;
    // Si usamos shared_ptr aquí y en la otra clase, crearíamos un ciclo.
    // Usamos weak_ptr para decir: "Lo conozco, pero no soy su dueño".
    std::weak_ptr<Persona> amigo; 

    Persona(std::string n) : nombre(n) { std::cout << "[NACE] " << nombre << "\n"; }
    ~Persona() { std::cout << "[MUERE] " << nombre << "\n"; }

    void saludarAmigo() {
        // Un weak_ptr no se puede usar directamente (podría estar muerto).
        // Hay que "bloquearlo" (lock) para obtener un shared_ptr temporal.
        if (auto compartido = amigo.lock()) {
            std::cout << nombre << " dice: Hola, " << compartido->nombre << "!\n";
        } else {
            std::cout << nombre << " dice: Mi amigo ya no existe...\n";
        }
    }
};

void demostracionCiclo() {
    auto paco = std::make_shared<Persona>("Paco");
    auto maria = std::make_shared<Persona>("Maria");

    // Creamos la conexión
    paco->amigo = maria;
    maria->amigo = paco;

    std::cout << "Referencias de Paco: " << paco.use_count() << "\n";
    paco->saludarAmigo();

} // FINAL DEL ÁMBITO: 
  // Gracias a weak_ptr, el contador de referencias real es 1 para cada uno.
  // Al salir de aquí, ambos mueren correctamente.

int main() {
    std::cout << "--- Inicio de Relaciones Peligrosas ---" << std::endl;
    demostracionCiclo();
    std::cout << "--- Fin (Sin fugas de memoria) ---" << std::endl;
    return 0;
}

/**
 * COMENTARIOS TÉCNICOS
 * ===================
 * 1. EL CICLO DE LA MUERTE: Si `amigo` fuera [std::shared_ptr](https://en.cppreference.com), 
 *    Paco no moriría hasta que Maria soltara su puntero, y Maria no moriría hasta que Paco 
 *    soltara el suyo. Resultado: Fuga de memoria (Memory Leak).
 * 
 * 2. EL LOCK: El método `lock()` de [std::weak_ptr](https://en.cppreference.com) 
 *    es atómico. Verifica si el objeto aún existe y, si es así, nos da un shared_ptr 
 *    para trabajar seguros.
 * 
 * 3. CADUCIDAD: Puedes verificar si el objeto observado ha sido borrado usando 
 *    el método `expired()`.
 * 
 * 4. CASO DE USO: Ideal para sistemas de caché, observadores de eventos o estructuras 
 *    de datos cíclicas como grafos o listas doblemente enlazadas.
 */

 // Compilar
 // ========
 // g++ weak_ptr.cpp -o ./build/weak_ptr
