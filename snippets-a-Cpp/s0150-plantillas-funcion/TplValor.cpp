#include <iostream>
#include <string>

/**
 * @file TplValor.cpp
 * @brief Plantillas de Función: El paso por Valor.
 * @author alcón68
 * 
 * CONCEPTO: El "molde" crea una COPIA de los argumentos. Es ideal para tipos 
 * primitivos (int, double) donde copiar es muy barato y rápido.
 */

template <typename T>
T maximo(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    std::cout << "=== TEMPLATES POR VALOR (Copia) ===\n";

    // Eficiente para tipos pequeños (4-8 bytes)
    std::cout << "Maximo int: " << maximo(10, 20) << std::endl;
    std::cout << "Maximo double: " << maximo(5.7, 2.1) << std::endl;

    return 0;
}

/**
 * COMENTARIOS TÉCNICOS
 * ===================
 * 1. COPIA: Al llamar a `maximo(a, b)`, se duplican los valores en la pila (stack).
 * 2. SOBRECARGA: Si T es un objeto pesado (ej. un vector de 1 millón de datos), 
 *    este código sería extremadamente lento por la duplicación innecesaria.
 */

// Compilar: g++ TplValor.cpp -o ./build/TplValor