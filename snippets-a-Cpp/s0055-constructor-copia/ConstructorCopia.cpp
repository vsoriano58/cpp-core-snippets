#include <iostream>
#include <cstring>

/**
 * SNIPPET #0110: El Constructor de Copia (Deep Copy)
 * 
 * CONCEPTO: Cuando clonamos un objeto que tiene memoria en el HEAP,
 * C++ por defecto hace una "Copia Superficial" (copia el puntero, no los datos).
 * El Constructor de Copia es nuestra herramienta para realizar una "Copia Profunda".
 * 
 * ESCENARIO: Un lienzo de dibujo (Canvas) que reserva memoria dinámica.
 */

class Canvas {
public:
    int* pixeles;
    int tamaño;

    // Constructor estándar
    Canvas(int t) : tamaño(t) {
        pixeles = new int[t]; // Reserva en el HEAP [M-01]
        for(int i=0; i<t; i++) pixeles[i] = 255; // Blanco
        std::cout << "Constructor: Memoria reservada en " << pixeles << std::endl;
    }

    // --- EL CONSTRUCTOR DE COPIA ---
    // Se invoca cuando hacemos: Canvas c2 = c1; o pasamos por valor.
    Canvas(const Canvas& otro) {
        this->tamaño = otro.tamaño;
        
        // ¡CLAVE!: No copiamos el puntero (otro.pixeles). 
        // Solicitamos NUEVA memoria para el clon. [M-02]
        this->pixeles = new int[otro.tamaño];
        
        // Copiamos el CONTENIDO bit a bit del original al nuevo
        std::memcpy(this->pixeles, otro.pixeles, sizeof(int) * otro.tamaño);
        
        std::cout << "Constructor de COPIA: Nueva memoria en " << this->pixeles 
                  << " (Clonando datos de " << otro.pixeles << ")" << std::endl;
    }

    ~Canvas() {
        std::cout << "Destructor: Liberando " << pixeles << std::endl;
        delete[] pixeles; // [M-03]
    }
};

int main() {
    std::cout << "--- Creando original ---" << std::endl;
    Canvas original(10);

    std::cout << "\n--- Creando copia (Clonación) ---" << std::endl;
    Canvas copia = original; // Aquí se invoca el Constructor de Copia [M-04]

    std::cout << "\n--- Fin del programa (Limpieza automática) ---" << std::endl;
    return 0;
}

/**
 * COMENTARIOS TÉCNICOS
 * ===================
 * 1. LA FIRMA: Debe ser 'Clase(const Clase& otro)'. El 'const' asegura que no 
 *    modificamos el original, y la '&' evita una recursión infinita de copias.
 * 
 * 2. EL PELIGRO EVITADO: Si no definiéramos este constructor, 'copia.pixeles' 
 *    apuntaría a la misma dirección que 'original.pixeles'. Al cerrar el programa, 
 *    el primer destructor liberaría la memoria y el segundo intentaría liberar 
 *    memoria ya borrada (Dangling Pointer / Double Free), provocando un CRASH.
 * 
 * 3. INDEPENDENCIA: Con la Copia Profunda, si cambias un píxel en 'original', 
 *    el objeto 'copia' no se entera. Son entidades físicas distintas.
 */

 /*
    COMENTARIO FINAL DE LA IA   
    =========================
    ¿Cómo encaja esto en tu estudio?
    Antes de subir el S0060 (Move Semantics), este snippet te servirá para explicar por qué a veces copiar es caro (tienes que pedir memoria y copiar datos). Eso te dará el "gancho" perfecto para decir en el siguiente: "¿Y si en vez de copiar la memoria, simplemente le robamos el puntero al objeto que va a morir?".
    ¿Te parece que este nivel de detalle en el Constructor de Copia ayuda a entender por qué el S0050 (Copia Superficial) era peligroso?
 */

 // Compilar
 // ========
 // g++ ConstructorCopia.cpp -o ./build/ConstructorCopia