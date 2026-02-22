#include <iostream>
#include <string>

/**
 * SNIPPET S0010-(A): El puntero 'this', Aritmética de Memoria y Method Chaining.
 * 
 * Este código demuestra la estructura física de un objeto en memoria 
 * y el uso de 'this' para el encadenamiento de métodos.
 */

class Persona {        // [REF-07]
public:
    int dni;           // 4 bytes
    int edad;          // 4 bytes
    std::string nombre; 

    Persona(int d, int e, std::string n) : dni(d), edad(e), nombre(n) {}

    // --- ANÁLISIS FÍSICO DE LA MEMORIA ---

    void mostrarMapaMemoria() {
        std::cout << "--- Mapa de Memoria [" << nombre << "] ---" << std::endl;
        std::cout << "Direccion base (this): " << this << std::endl; // [REF-01]

        unsigned char* base = (unsigned char*)this; // [REF-02]

        // Cálculo manual de direcciones mediante offsets
        int* pDni  = (int*)(base + 0); // [REF-03]
        int* pEdad = (int*)(base + 4); 

        std::cout << "Atributo 'dni'  (base + 0): " << pDni  << " -> Valor: " << *pDni  << std::endl;
        std::cout << "Atributo 'edad' (base + 4): " << pEdad << " -> Valor: " << *pEdad << std::endl;
        std::cout << "------------------------------------------" << std::endl;
    }

    // --- INTERFAZ FLUIDA (ENCADENAMIENTO) ---

    Persona& setEdad(int edad) {
        this->edad = edad; // [REF-04]
        return *this;      // [REF-05]
    }

    Persona& setNombre(std::string nombre) {
        this->nombre = nombre;
        return *this;
    }
};

int main() {
    std::cout << "=== SNIPPET S0010: EL PUNTERO THIS ===\n" << std::endl;

    // 1. Instancia y encadenamiento de métodos
    Persona persona(0, 0, "Provisional");
    persona.setEdad(40).setNombre("Alcon68"); // [REF-06]

    // 2. Verificación de memoria
    persona.mostrarMapaMemoria();   // [REF-07]

    // 3. Encadenamiento con objeto temporal
    Persona(999, 18, "Temporal").setEdad(25).mostrarMapaMemoria();

    return 0;
}

/*
    Opción 1. Con un click en VS Code
        Explicado en la teoría: 5. Guía de Configuración (VS Code)

    Opción 2. Con g++ (permite indicar directorio de destino del ejecutable)
        mkdir build
        g++ Persona.cpp -o ./build/Persona
 */


