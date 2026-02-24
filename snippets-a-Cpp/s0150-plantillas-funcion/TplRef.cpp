#include <iostream>
#include <string>

/**
 * @file TplRef.cpp
 * @brief Plantillas de Función: El paso por Referencia Constante.
 * @author alcón68
 * 
 * CONCEPTO: El "molde" trabaja sobre la DIRECCIÓN de los datos originales. 
 * Es la forma profesional de evitar copias pesadas en memoria.
 */

template <typename T>
const T& maximo(const T& a, const T& b) {
    return (a > b) ? a : b;
}

int main() {
    std::cout << "=== TEMPLATES POR REFERENCIA (Direccion) ===\n";

    // Funciona igual para tipos pequeños...
    std::cout << "Maximo int: " << maximo(10, 20) << std::endl;

    // ...pero es VITAL para objetos grandes (std::string, clases)
    std::string s1 = "Alpha";
    std::string s2 = "Zeta";
    std::cout << "Maximo string: " << maximo(s1, s2) << std::endl;

    return 0;
}

/**
 * COMENTARIOS TÉCNICOS
 * ===================
 * 1. RENDIMIENTO: No se copia el contenido; solo se pasa un puntero interno (8 bytes).
 * 2. SEGURIDAD: El uso de `const` impide que la función modifique accidentalmente 
 *    los datos originales que nos han prestado.
 * 3. RETORNO: Devolvemos `const T&` para que el resultado también sea una 
 *    referencia al ganador, evitando una copia al salir de la función.
 */

// Compilar: g++ TplRef.cpp -o ./build/TplRef