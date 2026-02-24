#include <iostream>
#include <memory>
#include <string>
#include <vector>

/**
 * @file resumen.cpp
 * @brief Ecosistema Final: Integración de la familia Smart Pointers.
 * @author alcón68
 * 
 * ESCENARIO DE DISEÑO:
 * 1. UNIQUE_PTR: Propiedad privada e intransferible (El ID del Usuario).
 * 2. SHARED_PTR: Recurso compartido por múltiples entidades (El Motor).
 * 3. WEAK_PTR: Observador pasivo que no retiene el recurso (El Monitor).
 */

class Motor {
public:
    Motor() { std::cout << "[MOTOR] Reserva de Hardware: Rugiendo... ✅\n"; }
    ~Motor() { std::cout << "[MOTOR] Hardware Liberado: Apagado seguro. 🛑\n"; }
    void check() { std::cout << "  -> Telemetría: Funcionando al 100%.\n"; }
};

class Usuario {
    // El ID es propiedad EXCLUSIVA del usuario. Nadie más puede ser dueño de este string.
    std::unique_ptr<std::string> id_privado; 
public:
    Usuario(std::string nombre) : id_privado(std::make_unique<std::string>(nombre)) {}
    
    void identificar() { 
        std::cout << "[USUARIO] Accediendo a identidad segura: " << *id_privado << "\n"; 
    }
};

int main() {
    std::cout << "=== SNIPPET #0100: EL ECOSISTEMA SMART POINTERS ===\n\n";

    // 1. UNIQUE_PTR: Eficiencia y Privacidad.
    // Solo 'paco' gestiona este recurso. No hay sobrecoste de contadores.
    Usuario paco("PACO_99");
    paco.identificar();

    // 2. SHARED_PTR: Gestión Colectiva.
    // Creamos el motor. Contador = 1.
    std::shared_ptr<Motor> motorPrincipal = std::make_shared<Motor>();
    
    {
        std::cout << "\n[SISTEMA] Entra Servidor Auxiliar en el clúster...\n";
        std::shared_ptr<Motor> servidorAux = motorPrincipal; // Contador = 2.
        std::cout << "[INFO] Dueños del motor: " << motorPrincipal.use_count() << "\n";
        servidorAux->check();
        std::cout << "[SISTEMA] Servidor Auxiliar se desconecta.\n";
    } // Aquí muere servidorAux, pero el motor sigue vivo. Contador = 1.

    // 3. WEAK_PTR: Vigilancia No Invasiva.
    // El monitor "mira" el motor pero no impide que se apague.
    std::weak_ptr<Motor> monitorLocal = motorPrincipal;

    std::cout << "\n--- Simulando Apagado del Sistema Central ---\n";
    
    // Verificamos salud del motor a través del monitor
    if (auto temporal = monitorLocal.lock()) {
        std::cout << "[MONITOR] Motor detectado online.";
        temporal->check();
    }

    // El dueño principal suelta el motor. Contador -> 0.
    std::cout << "[INFO] Liberando puntero principal...\n";
    motorPrincipal.reset(); 

    // El monitor intenta acceder ahora al recurso destruido
    if (monitorLocal.expired()) {
        std::cout << "[MONITOR] Alerta: El recurso ha sido destruido legalmente. Acceso denegado.\n";
    }

    std::cout << "\n=== CIERRE DEL ECOSISTEMA SEGURO ===\n";
    return 0;
}

/**
 * RESUMEN FILOSÓFICO PARA TU "CHULETA":
 * 1. UNIQUE_PTR: El "Cepillo de Dientes". No se presta. Si lo pierdes, se tira (Destruye).
 * 2. SHARED_PTR: El "Coche de Empresa". Varios tienen llave. El último que lo usa, cierra el garaje.
 * 3. WEAK_PTR: El "Espejo Retrovisor". Miras lo que hay, pero no controlas su existencia.
 */

// Compilar: g++ resumen.cpp -o ./build/resumen
