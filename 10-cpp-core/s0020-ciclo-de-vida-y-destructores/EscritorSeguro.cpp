#include <iostream>
#include <fstream>
#include <string>

/**
 * @title: Gestión de Recursos mediante el Ciclo de Vida (RAII).
 * @description: Vincular la existencia de un recurso crítico (un archivo) 
 *               a la vida de una variable en la Pila (Stack).
 */

class EscritorSeguro {
private:
    std::ofstream archivo;  // El recurso físico (identificador de flujo en disco)

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
        
        // Al llegar a la llave de cierre '}' del main, el 'frame' de la pila se limpia.
        // La memoria que ocupaba la variable 'miNota' se libera.
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
 *    Resource Acquisition Is Initialization. Es el pilar de C++ para evitar fugas de memoria.
 *    La idea es que el programador no tenga que acordarse de hacer ".close()" o "free()". 
 *    Si la variable muere en la Pila, el recurso asociado muere con ella.
 * 
 * 2. SEGURIDAD ANTE EXCEPCIONES:
 *    Si ocurre un error, el sistema no se detiene en seco; inicia un efecto dominó donde todas 
 *    las piezas (objetos) en la pila caen en orden inverso, ejecutando sus destructores hasta 
 *    limpiar el escenario
 * 
 * 3. PILA VS HEAP EN ESTE SNIPPET:
 *    - 'miNota' reside en la PILA (gestión automática).
 *    - El "buffer" interno de 'std::ofstream' suele pedir memoria al HEAP para 
 *      gestionar el flujo de datos, pero como 'EscritorSeguro' contiene a 'archivo', 
 *      la limpieza es en cascada.
 * 
 * 4. DIFERENCIA CON PUNTEROS:
 *    Si hubiéramos hecho 'EscritorSeguro* miNota = new EscritorSeguro(...)', el destructor 
 *    NUNCA se llamaría automáticamente; el puntero se perdería, pero el objeto seguiría 'vivo'
 *    en el Heap, dejando el archivo abierto (Leak de recurso)."
 */

 // Compilar
 // ========
 // g++ EscritorSeguro.cpp -o ./build/EscritorSeguro
