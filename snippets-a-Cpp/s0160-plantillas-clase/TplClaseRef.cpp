/**
 * @file TplClaseRef.cpp
 * @brief Plantillas de Clase: El paso por Referencia Constante.
 * @author alcón68
 * 
 * CONCEPTO: El "molde" recibe la DIRECCIÓN del dato original.
 * Se usa 'const' para garantizar que la caja no pueda modificar el 
 * valor original, solo leerlo para almacenarlo eficientemente.
 */

#include <iostream>
#include <string>

/**
 * SNIPPET #0121: Clases Genéricas (Templates de Clase) - Versión Referencia.
 * 
 * CONCEPTO: Al usar 'const T&', evitamos crear copias temporales en 
 * el constructor. Es la forma óptima de programar en C++.
 */

template <typename T>
class CajaMagica {
private:
    T contenido;

public:
    /**
     * Constructor: Recibe 'algo' por referencia constante.
     * No se crea una copia del argumento al llamar a la función.
     * El dato se copia directamente al atributo 'contenido' vía lista de inicialización.
     */
    CajaMagica(const T& algo) : contenido(algo) {}

    // Marcamos el método como 'const' porque no altera los atributos de la clase.
    void mostrar() const {
        std::cout << "[CAJA REF] Contenido guardado: " << contenido << std::endl;
    }

    /**
     * Retorno por Referencia Constante:
     * Al devolver el dato, tampoco creamos una copia de salida. 
     * El receptor recibe una "ventana" al dato almacenado.
     */
    const T& obtener() const {
        return contenido;
    }
};

int main() {
    std::cout << "=== SNIPPET #0121: CLASES GENÉRICAS (REFERENCIA) ===\n\n";

    // 1. Uso con tipos simples (Aquí la mejora es despreciable, pero funciona igual)
    CajaMagica<int> cajaEntera(777);
    cajaEntera.mostrar();

    // 2. Uso con tipos complejos (Aquí es donde brilla el rendimiento)
    // No hay copia pesada del string al entrar al constructor.
    std::string textoLargo = "Este es un texto que podria ser muy pesado en memoria";
    CajaMagica<std::string> cajaTexto(textoLargo);
    
    cajaTexto.mostrar();

    // 3. Verificación de seguridad
    std::cout << "Valor obtenido por ref: " << cajaTexto.obtener() << std::endl;

    return 0;
}

/**
 * COMENTARIOS TÉCNICOS (PASO POR REFERENCIA)
 * ==========================================
 * 1. EFICIENCIA: 'const T&' evita que el compilador ejecute el constructor 
 *    de copia del tipo T al pasar el argumento. Crucial en sistemas de alto rendimiento.
 * 
 * 2. READ-ONLY: Al ser 'const', protegemos la integridad del dato. No podemos 
 *    hacer 'algo = nuevo_valor' dentro del constructor.
 * 
 * 3. MÉTODOS CONST: El uso de 'const' al final de mostrar() y obtener() es 
 *    parte de la "Const Correctness", permitiendo usar la clase con objetos constantes.
 */

 // COMPILACIÓN:  g++ TplClaseRef.cpp -o ./build/TplClaseRef

