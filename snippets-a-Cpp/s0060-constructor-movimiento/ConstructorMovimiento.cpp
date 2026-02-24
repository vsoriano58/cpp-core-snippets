#include <iostream>
#include <utility> // Para std::move

/**
 * @file ConstructorMovimiento.cpp
 * @brief Implementación de Semántica de Movimiento (Move Semantics).
 * @author alcón68
 * 
 * CONCEPTO: En lugar de copiar memoria (caro), "robamos" el puntero de un 
 * objeto temporal que va a ser destruido. Es una transferencia de propiedad.
 */

class Canvas {
public:
    int* pixeles;
    size_t size;

    /** @brief Constructor estándar. */
    Canvas(size_t s) : size(s) {
        pixeles = new int[s];
        std::cout << "[NACE] Canvas original. Heap en: " << (void*)pixeles << std::endl;
    }

    /**
     * @brief CONSTRUCTOR DE MOVIMIENTO (Move Constructor)
     * @param otro Referencia R-value (&&). Indica que 'otro' es un objeto temporal.
     * 
     * @details IMPORTANTE: No pedimos nueva memoria con 'new'. 
     * Simplemente redirigimos nuestro puntero al Heap del objeto original.
     */
    Canvas(Canvas&& otro) noexcept {
        // 1. "Robamos" el recurso
        this->pixeles = otro.pixeles;
        this->size = otro.size;

        // 2. DEJAMOS AL ORIGINAL EN UN ESTADO SEGURO
        // Si no ponemos a nullptr el original, su destructor borraría NUESTRA memoria.
        otro.pixeles = nullptr;
        otro.size = 0;

        std::cout << "[MOVIMIENTO] Recurso transferido desde: " << (void*)this->pixeles << std::endl;
    }

    /** @brief Destructor: Solo libera si el puntero no es nulo. */
    ~Canvas() {
        if (pixeles != nullptr) {
            std::cout << "[MUERE] Liberando Heap en: " << (void*)pixeles << std::endl;
            delete[] pixeles;
        } else {
            std::cout << "[MUERE] Objeto 'vacío' (recurso ya movido). Nada que liberar." << std::endl;
        }
    }
};

/** @brief Función que devuelve un objeto temporal (R-value). */
Canvas crearCanvasGrande() {
    Canvas temp(1000); 
    return temp; // Aquí el compilador usará MOVIMIENTO automáticamente.
}

int main() {
    std::cout << "--- Inicio s0060: Semántica de Movimiento ---" << std::endl;

    // Caso A: Movimiento automático desde una función
    std::cout << "\n1. Creando desde función:" << std::endl;
    Canvas c1 = crearCanvasGrande(); 

    // Caso B: Forzar movimiento de un objeto existente
    std::cout << "\n2. Forzando movimiento de c1 a c2:" << std::endl;
    Canvas c2 = std::move(c1); // c1 queda invalidado (puntero a null)

    std::cout << "\n--- Fin del programa ---" << std::endl;
    return 0;
}

/**
 * REFLEXIÓN DE INGENIERÍA:
 * 1. ¿POR QUÉ ES MÁS RÁPIDO?: Porque evitamos 'new', 'memcpy' y 'delete'. 
 *    Solo reasignamos un puntero (una operación de nanosegundos).
 * 
 * 2. EL OPERADOR &&: Indica una "Referencia R-value". Representa objetos 
 *    que no tienen nombre o que están a punto de ser destruidos.
 * 
 * 3. NOEXCEPT: Es vital marcarlo como 'noexcept' para que los contenedores 
 *    como std::vector prefieran mover en lugar de copiar durante un resize.
 */

// Compilar: g++ ConstructorMovimiento.cpp -o ./build/ConstructorMovimiento
