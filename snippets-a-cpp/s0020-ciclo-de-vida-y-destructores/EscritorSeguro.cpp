#include <iostream>
#include <fstream>
#include <string>

/**
 * SNIPPET #0020: Gestión de Recursos mediante el Ciclo de Vida (RAII).
 * 
 * CONCEPTO: Vincular la existencia de un recurso crítico (un archivo) 
 * a la vida de una variable en la Pila (Stack).
 */

class EscritorSeguro {
private:
    std::ofstream archivo; // El recurso físico (identificador de flujo en disco)

public:
    // 1. ADQUISICIÓN: El constructor vincula el objeto al recurso externo.
    EscritorSeguro(std::string nombre) {
        archivo.open(nombre); 
        std::cout << "[RAII] Recurso adquirido: Archivo abierto en disco.\n";
    }

    // 2. USO: Interfaz para operar sobre el recurso protegido.
    void escribir(std::string texto) {
        if (archivo.is_open()) archivo << texto << std::endl;
    }

    // 3. LIBERACIÓN: El destructor garantiza la limpieza sin intervención del usuario.
    ~EscritorSeguro() {
        if (archivo.is_open()) {
            archivo.close(); // Liberación forzosa del "handle" del S.O.
        }
        std::cout << "[RAII] Recurso liberado: Archivo cerrado por el destructor.\n";
    }
};

int main() {
    // Bloque de ámbito (scope) artificial para controlar la vida de la variable
    {
        // La variable 'miNota' se reserva en la PILA.
        EscritorSeguro miNota("diario.txt"); 
        
        miNota.escribir("Hola, esto es C++ solido.");
        
        // Al llegar a la llave de cierre '}', el 'frame' de la pila se limpia.
        // El compilador inserta automáticamente la llamada al destructor ~EscritorSeguro().
    } 

    std::cout << "El programa continua, pero el recurso ya ha sido devuelto al sistema.\n";

    std::cout << std::endl;
    return 0;
}

/**
 * COMENTARIOS
 * ===========
 * 1. EL CONCEPTO RAII:
 *    Resource Acquisition Is Initialization. Es el pilar de C++ para evitar fugas.
 *    La idea es que el programador no tenga que acordarse de hacer ".close()" o 
 *    "free()". Si la variable muere en la Pila, el recurso muere con ella.
 * 
 * 2. SEGURIDAD ANTE EXCEPCIONES:
 *    Si el programa fallara o lanzara un error dentro del bloque {}, el destructor 
 *    se ejecutaría igualmente mientras la pila se "desenrolla". Esto garantiza que 
 *    el archivo no quede bloqueado por el Sistema Operativo.
 * 
 * 3. PILA VS HEAP EN ESTE SNIPPET:
 *    - 'miNota' reside en la PILA (gestión automática).
 *    - El "buffer" interno de 'std::ofstream' suele pedir memoria al HEAP para 
 *      gestionar el flujo de datos, pero como 'EscritorSeguro' contiene a 'archivo', 
 *      la limpieza es en cascada.
 * 
 * 4. DIFERENCIA CON PUNTEROS:
 *    Si hubiéramos hecho 'EscritorSeguro* miNota = new EscritorSeguro(...)', el destructor 
 *    NUNCA se llamaría al salir del bloque. El archivo quedaría abierto y la memoria 
 *    perdida hasta un 'delete' explícito. Por eso, en C++ sólido, preferimos objetos 
 *    directos en la pila siempre que sea posible.
 */
