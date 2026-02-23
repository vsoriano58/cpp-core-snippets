#include <iostream>
#include <memory>
#include <vector>

/**
 * SNIPPET #0100: EL ECOSISTEMA FINAL (Resumen de Smart Pointers).
 * 
 * ESCENARIO: 
 * 1. Un 'Motor' (Recurso pesado) gestionado por un Servidor (Shared).
 * 2. Un 'Usuario' que tiene su propia 'Llave' privada (Unique).
 * 3. Un 'Monitor' que vigila si el Motor sigue vivo (Weak).
 */

class Motor {
public:
    Motor() { std::cout << "[MOTOR] Rugiendo... (Memoria reservada)\n"; }
    ~Motor() { std::cout << "[MOTOR] Apagado y liberado.\n"; }
    void estado() { std::cout << " -> Motor funcionando al 100%\n"; }
};

class Usuario {
    std::unique_ptr<std::string> ID; // Propiedad exclusiva del ID
public:
    Usuario(std::string nombre) : ID(std::make_unique<std::string>(nombre)) {}
    void identificar() { std::cout << "[USUARIO] Mi ID privado es: " << *ID << "\n"; }
};

int main() {
    std::cout << "=== SNIPPET #0100: EL GRAN RESUMEN ===\n\n";

    // 1. UNIQUE_PTR: Propiedad privada e intransferible.
    // Solo el objeto 'paco' puede leer su ID. Nadie más lo comparte.
    Usuario paco("PACO_99");
    paco.identificar();

    // 2. SHARED_PTR: El recurso compartido.
    // El Servidor A y el Servidor B comparten el mismo motor.
    std::shared_ptr<Motor> motorCompartido = std::make_shared<Motor>();
    std::cout << "[INFO] Usuarios del motor: " << motorCompartido.use_count() << "\n";

    {
        std::shared_ptr<Motor> servidorAuxiliar = motorCompartido; 
        std::cout << "[INFO] Entra Servidor Auxiliar. Usuarios: " << motorCompartido.use_count() << "\n";
    } // Aquí muere el auxiliar, pero el motor NO.

    // 3. WEAK_PTR: El observador pasivo.
    // El monitor sabe que el motor existe, pero no "tira" de él para mantenerlo vivo.
    std::weak_ptr<Motor> monitor = motorCompartido;

    std::cout << "\n--- Simulando Apagado del Sistema ---\n";
    
    // Verificamos el motor antes de que muera
    if (auto tmp = monitor.lock()) {
        std::cout << "[MONITOR] El motor sigue online.";
        tmp->estado();
    }

    motorCompartido.reset(); // Forzamos la muerte del motor (Contador -> 0)

    // El monitor intenta mirar ahora
    if (monitor.expired()) {
        std::cout << "[MONITOR] Alerta: El motor ha sido destruido legalmente.\n";
    }

    std::cout << "\n=== FIN DEL ECOSISTEMA SEGURO ===\n";
    return 0;
}

/**
 * REFLEXIÓN FINAL PARA TU DOCUMENTO:
 * =================================
 * - El UNIQUE_PTR es tu "cepillo de dientes": no se comparte (eficiencia pura).
 * - El SHARED_PTR es "el coche de la familia": todos tienen llave y el último 
 *   que lo usa cierra el garaje (gestión automática).
 * - El WEAK_PTR es "el retrovisor": miras lo que hay atrás, pero no controlas 
 *   el movimiento de los otros coches (evita dependencias circulares).
 * 
 * Con este arsenal, los errores de memoria manual del Snippet #0040 son 
 * ahora cosa del pasado.
 */

 // Compilar
 // ========
 // g++ resumen.cpp -o ./build/resumen
