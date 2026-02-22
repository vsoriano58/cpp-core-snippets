#include <iostream>
#include <string>

/**
 * @title: El Puntero 'this' y Anatomía de Objetos
 * @tags: @MemoryLayout, @ThisPointer, @MethodChaining, @PointerArithmetic
 * @description: Radiografía de un objeto en memoria: cómo 'this' actúa como 
 * la dirección base y cómo encadenar métodos para crear interfaces fluidas.
 */

class Persona {        
public:
    int dni;           // 4 bytes (en sistemas de 32/64 bits estándar para int)
    int edad;          // 4 bytes
    std::string nombre; 

    Persona(int d, int e, std::string n) : dni(d), edad(e), nombre(n) {}

    // --- ANÁLISIS FÍSICO DE LA MEMORIA ---

    void mostrarMapaMemoria() {
        std::cout << "--- Mapa de Memoria [" << nombre << "] ---" << std::endl;
        // [REF-01] 'this' es la dirección donde comienza el bloque del objeto.
        std::cout << "Direccion base (this): " << this << std::endl; 

        // [REF-02] Convertimos a char* para poder movernos byte a byte (aritmética de punteros).
        unsigned char* base = (unsigned char*)this; 

        // [REF-03] Cálculo manual de offsets. El compilador hace esto por nosotros habitualmente.
        int* pDni  = (int*)(base + 0); 
        int* pEdad = (int*)(base + 4); 

        std::cout << "Atributo 'dni'  (base + 0): " << pDni  << " -> Valor: " << *pDni  << std::endl;
        std::cout << "Atributo 'edad' (base + 4): " << pEdad << " -> Valor: " << *pEdad << std::endl;
        std::cout << "------------------------------------------" << std::endl;
    }

    // --- INTERFAZ FLUIDA (ENCADENAMIENTO) ---

    // [REF-04] La referencia permite devolver el objeto real, no una copia.
    Persona& setEdad(int edad) {
        this->edad = edad; // Resolvemos ambigüedad entre parámetro y atributo.
        return *this;      // [REF-05] Retornamos el contenido de 'this'.
    }

    Persona& setNombre(std::string nombre) {
        this->nombre = nombre;
        return *this;
    }
};

int main() {
    std::cout << "=== SNIPPET S0010: EL PUNTERO THIS ===\n" << std::endl;

    // [REF-06] Method Chaining: Al retornar referencia, podemos encadenar llamadas.
    Persona persona(0, 0, "Provisional");
    persona.setEdad(40).setNombre("Alcon68"); 

    persona.mostrarMapaMemoria();

    // [REF-07] Objeto temporal (R-value): Se crea, se modifica y muere en la misma línea.
    Persona(999, 18, "Temporal").setEdad(25).mostrarMapaMemoria();

    return 0;
}

/**
 * GPS DEL PROGRAMA [REF]
 * ======================
 * [REF-01] DIRECCIÓN BASE: Todo objeto es un bloque de memoria contiguo. 'this' apunta al inicio.
 * [REF-02] CASTING A CHAR: En C++, sumar 1 a un puntero lo mueve según el tamaño del tipo. 
 *           Usamos 'char' (1 byte) para desplazarnos con precisión quirúrgica de byte en byte.
 * [REF-03] OFFSETS FÍSICOS: Demuestra que los atributos están uno tras otro. El 'dni' está 
 *           en el byte 0 y la 'edad' empieza en el 4 (justo después del int de 4 bytes).
 * [REF-04] REFERENCIA DE RETORNO: Es vital devolver 'Persona&'. Si devolviéramos 'Persona' 
 *           (por valor), el encadenamiento trabajaría sobre copias y no sobre el objeto original.
 * [REF-05] DESREFERENCIA DE THIS: 'this' es la dirección (puntero), '*this' es el objeto. 
 *           Al devolverlo, permitimos que el siguiente método se ejecute sobre el mismo barco.
 * [REF-06] INTERFAZ FLUIDA: Técnica estética que hace el código más legible y expresivo.
 * [REF-07] CICLO DE VIDA TEMPORAL: El objeto vive lo que dura la expresión. Es útil para 
 *           operaciones "usar y tirar" sin ensuciar la pila con nombres de variables.
 */
