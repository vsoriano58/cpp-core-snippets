#include <iostream>
#include <string>

/**
 * @title: Paso por Valor (La Fotocopia)
 * @description: Demuestra cómo el paso por valor crea una copia independiente cuyos
 * cambios no afectan al dato original.
 */

// [REF-001] La firma recibe 'int x'. El compilador reserva un nuevo espacio en la pila.
void intentarCambiarValor(int x) {
    x = 99;      // [REF-002] Solo modificamos la copia local.
}

int main() {
    int dato = 10;

    std::cout << "Original antes: " << dato << std::endl;  // 10

    // [REF-003] Se envía el valor, no la variable.
    intentarCambiarValor(dato);

    std::cout << "Original despues: " << dato << " (Inalterado)" << std::endl;  // 10

    return 0;
}

/**
 * GPS DEL PROGRAMA [REF]
 * ======================
 * [REF-001] PARÁMETRO POR VALOR: Al no llevar '&', la función exige una copia. 
 *           Si el dato fuera un objeto grande, esto penalizaría el rendimiento.
 * [REF-002] SCOPE LOCAL: La variable 'x' nace cuando la función es llamada y 
 *           muere al llegar a su llave de cierre. Es "volátil".
 * [REF-003] AISLAMIENTO: 'dato' y 'x' viven en direcciones de memoria distintas. 
 *           Son dos barcos navegando en océanos diferentes.
 */

 // Compilar
 // ========
 // g++ imprimir.cpp -o ./build/imprimir
