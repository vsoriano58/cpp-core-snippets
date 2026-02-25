/**
 * @file TplClaseValor.cpp
 * @brief Plantillas de Clase: El paso por Valor.
 * @author alcón68
 * 
 * CONCEPTO: El "molde" crea una COPIA de los datos al ingresarlos en la clase.
 * Útil para tipos de datos básicos (int, char, float) o cuando queremos
 * asegurar que la clase sea dueña de su propia copia independiente.
 */

#include <iostream>
#include <string>

/**
 * SNIPPET #0120: Clases Genéricas (Templates de Clase) - Versión Paso por Valor.
 * 
 * CONCEPTO: Crear una estructura (una "Caja") que puede contener 
 * y gestionar CUALQUIER tipo de dato, definido solo en el momento 
 * de crear el objeto.
 */

// T representa el tipo de contenido que la caja guardará.
template <typename T>
class CajaMagica {
private:
    T contenido;

public:
    // Constructor: Recibe el dato 'algo' por valor (crea una copia local).
    CajaMagica(T algo) : contenido(algo) {}

    // Método mostrar: Imprime el valor actual.
    void mostrar() {
        std::cout << "[CAJA VALOR] Contenido guardado: " << contenido << std::endl;
    }

    // Método obtener: Retorna una copia del contenido.
    T obtener() {
        return contenido;
    }
};

int main() {
    std::cout << "=== SNIPPET #0120: CLASES GENÉRICAS (VALOR) ===\n\n";

    // 1. Una caja para números (El compilador genera la versión 'int')
    CajaMagica<int> cajaEntera(500);
    cajaEntera.mostrar();

    // 2. Una caja para texto (El compilador genera la versión 'string')
    // Nota cómo pasamos el tipo entre ángulos < >
    CajaMagica<std::string> cajaTexto("Diamante");
    cajaTexto.mostrar();

    // 3. Demostración de independencia (Paso por valor)
    int numero = 10;
    CajaMagica<int> cajaDinamica(numero);
    numero = 20; // Cambiamos el original
    std::cout << "Original: " << numero << " | En caja: " << cajaDinamica.obtener() << std::endl;
    // La caja mantiene el 10 porque guardó una COPIA.

    std::cout << "\nValor final extraído: " << cajaTexto.obtener() << std::endl;

    return 0;
}

/**
 * COMENTARIOS TÉCNICOS (PASO POR VALOR)
 * =====================================
 * 1. COPIA EN MEMORIA: Al usar `CajaMagica(T algo)`, si 'T' es un objeto 
 *    grande (como un string de 1GB), el programa duplicará ese GB en RAM.
 * 
 * 2. INSTANCIACIÓN: Se requiere especificar el tipo `NombreClase<Tipo>` 
 *    para que el compilador sepa qué tamaño de "hueco" reservar para 'T'.
 * 
 * 3. SEGURIDAD: El paso por valor es más seguro si el dato original va a 
 *    desaparecer o cambiar y queremos que la caja conserve el estado inicial.
 */
 
 // COMPILACIÓN: g++ TplClaseValor.cpp -o ./build/TplClaseValor
 
