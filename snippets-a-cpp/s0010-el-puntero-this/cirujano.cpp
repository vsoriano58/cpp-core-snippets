#include <iostream>
#include <string>

/**
 * SNIPPET S0010-B: El Puntero 'this' - Simil del Cirujano.
 * 
 * Este programa demuestra cómo una única función (Lógica) actúa sobre 
 * diferentes direcciones de memoria (Datos) usando punteros y 'this'.
 */

class Paciente {
public:
    std::string nombre;
    
    Paciente(std::string n) : nombre(n) {}

    void serOperado() {
        // [REF-01] 'this' identifica qué paciente está en la camilla
        std::cout << "Paciente: " << nombre << " | Dirección (this): " << this << std::endl;
        std::cout << "Operación finalizada con éxito." << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
    }
};

class Cirujano {
public:
    void realizarCirugia(Paciente* p) { // [REF-02] Recibe la dirección del paciente
        std::cout << "\n>>> El cirujano entra en el quirófano..." << std::endl;
        std::cout << "Dirección del Cirujano (this): " << this << std::endl; // [REF-03]
        std::cout << "Dirección del Paciente a operar: " << p << std::endl;
        
        p->serOperado(); // [REF-04] Salto al método del paciente
    }
};

int main() {
    std::cout << "=== SNIPPET S0010-B: EL CIRUJANO Y EL PACIENTE ===\n" << std::endl;

    Paciente p1("Juan Perez");
    Paciente p2("Maria Garcia");
    Cirujano medico;

    // Caso 1: Operando a Juan
    medico.realizarCirugia(&p1);

    // Caso 2: Operando a Maria
    medico.realizarCirugia(&p2);

    return 0;
}

/*
    Compilar
    ========
    mkdir build   (si no existe build)
    g++ Cirujano.cpp -o ./build/Cirujano 
*/ 
