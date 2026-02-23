#include <iostream>
#include <string>

/**
 * SNIPPET #0110: Programación Genérica (Templates de Función).
 * 
 * CONCEPTO: Escribir una lógica "universal" que funcione con cualquier 
 * tipo de dato sin tener que repetir código.
 */

// El "Molde": T es un marcador de posición para el tipo de dato.
template <typename T>
T maximo(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    std::cout << "=== SNIPPET #0110: EL PODER DE LOS TEMPLATES ===\n\n";

    // 1. El compilador crea una versión para ENTEROS automáticamente.
    std::cout << "Maximo entre 10 y 20: " << maximo(10, 20) << std::endl;

    // 2. El compilador crea una versión para DECIMALES (double).
    std::cout << "Maximo entre 5.7 y 2.1: " << maximo(5.7, 2.1) << std::endl;

    // 3. ¡Incluso funciona con TEXTO (std::string)!
    std::string s1 = "Alpha";
    std::string s2 = "Zeta";
    std::cout << "Maximo alfabetico entre Alpha y Zeta: " << maximo(s1, s2) << std::endl;

    std::cout << std::endl;

    return 0;
}

/**
 * COMENTARIOS TÉCNICOS
 * ===================
 * 1. ABSTRACCIÓN: No programamos para un dato, sino para un COMPORTAMIENTO 
 *    (en este caso, la comparación con el signo '>').
 * 
 * 2. TIPADO FUERTE: Aunque parece magia, C++ sigue siendo estricto. Si intentas 
 *    usar `maximo` con una clase que no tiene definido el operador '>', el 
 *    error saltará en tiempo de compilación.
 * 
 * 3. EFICIENCIA: No hay penalización de velocidad. El compilador genera el 
 *    código específico para cada tipo que uses detrás de las cámaras.
 */

 // Compilar
 // ========
 // g++ TemplatesFfuncion.cpp -o ./build/TemplatesFfuncion
