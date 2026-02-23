#include <iostream>
#include <string>

/**
 * SNIPPET #0120: Clases Genéricas (Templates de Clase).
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
    CajaMagica(T algo) : contenido(algo) {}

    void mostrar() {
        std::cout << "[CAJA] Contenido guardado: " << contenido << std::endl;
    }

    T obtener() { return contenido; }
};

int main() {
    std::cout << "=== SNIPPET #0120: CLASES GENÉRICAS ===\n\n";

    // 1. Una caja para números (El compilador genera la versión 'int')
    CajaMagica<int> cajaEntera(500);
    cajaEntera.mostrar();

    // 2. Una caja para texto (El compilador genera la versión 'string')
    // Nota cómo pasamos el tipo entre ángulos < >
    CajaMagica<std::string> cajaTexto("Diamante");
    cajaTexto.mostrar();

    // 3. ¿Qué pasa si intentamos algo que no tiene sentido?
    // CajaMagica<int> error = "Hola"; // <--- Esto daría error de compilación.
    
    std::cout << "\nValor extraído de la caja de texto: " << cajaTexto.obtener() << std::endl;

    return 0;
}

/**
 * COMENTARIOS TÉCNICOS
 * ===================
 * 1. INSTANCIACIÓN: A diferencia de las funciones, al crear un objeto de clase 
 *    genérica debemos especificar el tipo entre ángulos: `NombreClase<Tipo>`.
 * 
 * 2. REUTILIZACIÓN: Fíjate que el código de `mostrar()` y `obtener()` es idéntico 
 *    para ambos casos. Hemos ahorrado escribir dos clases completas.
 * 
 * 3. STL CONNECTION: Así es exactamente como funciona el [std::vector<T>](https://en.cppreference.com). 
 *    Es una clase genérica diseñada para manejar una lista de 'T'.
 * 
 * 4. COMPILACIÓN: El código del template debe estar disponible en el archivo de 
 *    cabecera (.h), ya que el compilador necesita el "molde" completo para 
 *    fabricar la clase específica cada vez que la usas.
 */

 // Compilar
 // ========
 // g++ TemplatesClase.cpp -o ./build/TemplatesClase
