#include <iostream>
#include <string>

/**
 * SNIPPET #0040: Seguimiento del Ciclo de Vida del Objeto.
 * 
 * CONCEPTO: Visualizar los momentos exactos en los que el compilador 
 * invoca al Constructor, al Constructor de Copia y al Destructor.
 */

class Rastreador {
private:
    std::string etiqueta;

public:
    // 1. CONSTRUCTOR POR DEFECTO / PARAMETRIZADO
    Rastreador(std::string nombre) : etiqueta(nombre) {
        std::cout << "[NACE] Objeto '" << etiqueta << "' creado en la Pila.\n";
    }

    // 2. CONSTRUCTOR DE COPIA (Se activa en pasos por VALOR)
    Rastreador(const Rastreador& otro) {
        etiqueta = "Copia de " + otro.etiqueta;
        std::cout << "[COPIA] Se crea una '" << etiqueta << "' (fotocopia).\n";
    }

    // 3. DESTRUCTOR
    ~Rastreador() {
        std::cout << "[MUERE] Objeto '" << etiqueta << "' desaparece de la Pila.\n";
    }

    void saludar() {
        std::cout << "   Hola desde " << etiqueta << std::endl;
    }
};

// Función que fuerza un paso por VALOR (fotocopia)
void funcionQueCopia(Rastreador r) {
    r.saludar();
}

int main() {
    std::cout << "--- Inicio del Main ---\n";

    {
        Rastreador original("Original_01"); // Invocación del Constructor
        
        std::cout << "\nLlamando a funcionQueCopia...\n";
        funcionQueCopia(original); // Invocación del Constructor de Copia
        std::cout << "Regreso al Main...\n";

    } // Invocación del Destructor de 'original' al salir del ámbito

    std::cout << "\n--- Fin del Main ---\n";
    return 0;
}

/**
 * COMENTARIOS
 * ===========
 * 1. EL "ORDEN DE LAS COSAS":
 *    Observa la consola. Verás que la 'Copia' muere ANTES que el 'Original'. 
 *    Esto ocurre porque la copia vive en la pila de la función, y esa pila 
 *    se limpia en cuanto la función termina. El original espera a que se 
 *    cierre su propia llave '}'.
 * 
 * 2. EL CONSTRUCTOR DE COPIA (La Fotocopiadora):
 *    Cuando pasamos 'original' a 'funcionQueCopia(Rastreador r)', C++ no usa 
 *    el puntero 'this' del original. En su lugar, fabrica un objeto nuevo 
 *    basado en el primero. Este es el "peaje" de rendimiento del que hablábamos.
 * 
 * 3. EL DESTRUCTOR (El enterrador):
 *    Es la única función que se llama sola. No hay un 'miObjeto.destruir()'. 
 *    El compilador inserta el código de limpieza al final de cada bloque. 
 *    Es el mecanismo que hace que RAII (S0020) funcione.
 * 
 * 4. CONCLUSIÓN DE INGENIERÍA:
 *    Si este objeto gestionara 1GB de memoria, el mensaje [COPIA] significaría 
 *    que acabamos de gastar 1GB extra de RAM y tiempo de CPU. Al ver este 
 *    "rastreo" en consola, entiendes por qué las referencias (&) son 
 *    cruciales: evitan el disparo del Constructor de Copia.
 */

 // Compilar
 // ========
 // g++ ConstructoresDestructor.cpp -o ./build/ConstructoresDestructor

