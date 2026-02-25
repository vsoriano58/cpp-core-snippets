#include <iostream>
#include <cstring>

/**
 * @file ConstructorCopia.cpp
 * @brief Implementación de Copia Profunda (Deep Copy).
 * @author alcón68
 * 
 * CONCEPTO: El Constructor de Copia es el mecanismo de defensa contra la 
 * duplicidad de punteros en el Heap. Asegura la independencia de datos.
 */

class Canvas {
public:
    int* pixeles;
    int size;

    /** @brief Constructor estándar con reserva de Heap. */
    Canvas(int s) : size(s) {
        pixeles = new int[s]; 
        for(int i = 0; i < s; i++) pixeles[i] = 255; // Inicializado a blanco
        std::cout << "[NACE] Canvas original. Heap en: " << (void*)pixeles << std::endl;
    }

    /**
     * @brief CONSTRUCTOR DE COPIA (Deep Copy)
     * @details Evita que dos objetos compartan el mismo puntero.
     * Invocado en: Canvas c2 = c1; o al pasar el objeto por valor a una función.
     */
    Canvas(const Canvas& otro) {
        this->size = otro.size;
        
        // SOLUCIÓN: Solicitamos NUEVA memoria propia en el Heap.
        this->pixeles = new int[otro.size];
        
        // Clonamos físicamente el contenido del original al clon.
        // Se usa memcpy de <cstring> para máxima eficiencia en bloques de datos.
        std::memcpy(this->pixeles, otro.pixeles, sizeof(int) * otro.size);
        
        std::cout << "[COPIA] Nueva memoria en " << (void*)this->pixeles 
                  << " (Clon de " << (void*)otro.pixeles << ")" << std::endl;
    }

    /** @brief Destructor seguro. */
    ~Canvas() {
        std::cout << "[MUERE] Liberando Heap en: " << (void*)pixeles << std::endl;
        delete[] pixeles; 
    }
};

int main() {
    std::cout << "--- Inicio s0045: Demostración de Independencia ---" << std::endl;
    
    Canvas original(10);
    
    // Aquí se invoca el Constructor de Copia personalizado.
    Canvas copia = original; 

    std::cout << "\nPuntero Original: " << (void*)original.pixeles << std::endl;
    std::cout << "Puntero Copia:    " << (void*)copia.pixeles << std::endl;

    std::cout << "\n--- Fin del programa: Cada objeto limpia su propia memoria ---" << std::endl;
    return 0;
}

/**
 * NOTAS DE INGENIERÍA
 * ==================
 * 1. LA FIRMA: Usamos 'const Clase&' para evitar modificar el original y prevenir
 *    una recursión infinita (copiar la copia de la copia).
 * 
 * 2. PREVENCIÓN: Este código es la cura para el "Double Free Error" que se 
 *    analiza en el snippet s0050-peligro-copia-superficial.
 * 
 * 3. EFICIENCIA: Aunque [std::memcpy](https://en.cppreference.com) 
 *    es rápido, en el siguiente tema (Move Semantics) veremos cómo evitar 
 *    esta copia costosa si el objeto original ya no se va a usar.
 */

// Compilar: g++ ConstructorCopia.cpp -o ./build/ConstructorCopia
