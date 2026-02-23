#include <iostream>
#include <string>

/**
 * @title: El Símil del Cirujano (Lógica vs Datos)
 * @tags: @ThisPointer, @OOP, @MemoryAddresses, @Collaboration
 * @description: Demuestra cómo los métodos (lógica) son compartidos por todas 
 * las instancias, pero 'this' permite que operen sobre datos específicos.
 */

class Paciente {
public:
    std::string nombre;
    
    Paciente(std::string n) : nombre(n) {}

    void serOperado() {
        // [REF-01] El método es el mismo para todos, pero 'this' cambia según quién esté en la camilla.
        std::cout << "Paciente: " << nombre << " | Dirección (this): " << this << std::endl;
        std::cout << "Operación finalizada con éxito." << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
    }
};

class Cirujano {
public:
    // [REF-02] El cirujano no posee al paciente, solo recibe su ubicación (puntero).
    void realizarCirugia(Paciente* p) { 
        std::cout << "\n>>> El cirujano entra en el quirófano..." << std::endl;
        // [REF-03] El cirujano también tiene su propia dirección en la memoria.
        std::cout << "Dirección del Cirujano (this): " << this << std::endl; 
        std::cout << "Dirección del Paciente a operar: " << p << std::endl;
        
        // [REF-04] Invocamos el comportamiento del paciente usando la dirección recibida.
        p->serOperado(); 
    }
};

int main() {
    std::cout << "=== SNIPPET S0010-B: EL CIRUJANO Y EL PACIENTE ===\n" << std::endl;

    Paciente p1("Juan Perez");
    Paciente p2("Maria Garcia");
    Cirujano medico;

    // [REF-05] Pasamos la dirección (&) para que el cirujano sepa a quién operar.
    medico.realizarCirugia(&p1);
    medico.realizarCirugia(&p2);

    return 0;
}

/**
 * GPS DEL PROGRAMA [REF]
 * ======================
 * [REF-01] CONTEXTO DE EJECUCIÓN: 'this' es el parámetro invisible que C++ pasa 
 *           a cada método para que la función sepa qué atributos 'nombre' debe leer.
 * [REF-02] PASO POR PUNTERO: Al pasar 'Paciente*', el cirujano evita cargar con 
 *           el paciente entero (copia); solo maneja la "tarjeta de visita" con su dirección.
 * [REF-03] MULTIPLES INSTANCIAS: Demuestra que cada objeto (medico, p1, p2) ocupa 
 *           su propio "camarote" único en la memoria RAM.
 * [REF-04] OPERADOR FLECHA (->): Se usa para acceder a miembros de un objeto a través 
 *           de su puntero. Equivale a (*p).serOperado().
 * [REF-05] OPERADOR DE DIRECCIÓN (&): Extraemos la ubicación física de la variable 
 *           en la pila para enviarla al quirófano.
 */

 // Compilar
 // ========
 // g++ cirujano.cpp -o ./build/cirujano
