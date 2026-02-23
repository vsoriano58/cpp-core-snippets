#include <iostream>

/**
 * SNIPPET #0040 (B): El Peligro de la Copia Superficial (Shallow Copy).
 * 
 * CONCEPTO: Demostrar el fallo catastrófico que ocurre cuando dos objetos 
 * en la Pila comparten el mismo puntero al Heap sin un contador de referencias.
 */

class GestorPeligroso {
public:
    int* datos;
    
    GestorPeligroso() {
        datos = new int[10]; // Asignación de recursos en el Heap
        std::cout << "[NACE] Objeto creado. Memoria Heap en: " << (void*)datos << std::endl;
    }

    ~GestorPeligroso() {
        std::cout << "[MUERE] Intentando liberar Heap en: " << (void*)datos << std::endl;
        // El problema: si ya fue liberado por otro objeto, aquí el programa "peta".
        delete[] datos; 
    }
};

void causarDesastre() {
    // 1. Nace objetoA y reserva memoria.
    GestorPeligroso objetoA; 
    
    // 2. EL ERROR: Copia por defecto (Shallow Copy).
    // C++ copia el valor del puntero 'datos' bit a bit. 
    // Ahora objetoB.datos apunta a la MISMA camilla que objetoA.datos.
    GestorPeligroso objetoB = objetoA; 
    
    std::cout << "Puntero A: " << (void*)objetoA.datos << std::endl;
    std::cout << "Puntero B: " << (void*)objetoB.datos << std::endl;

} // 3. FINAL DEL ÁMBITO: 
  // - Primero muere objetoB: Llama a delete[] y el Heap queda libre.
  // - Luego muere objetoA: ¡Intenta llamar a delete[] sobre memoria YA LIBERADA! 
  //   Esto se llama "Double Free Error" y suele colapsar el programa.

int main() {
    std::cout << "--- Inicio del Experimento Peligroso ---" << std::endl;
    
    // Ejecutamos en un entorno controlado (función aparte)
    causarDesastre();
    
    std::cout << "--- Fin del experimento (Si ves esto, el S.O. fue indulgente) ---" << std::endl;
    return 0;
}

/**
 * COMENTARIOS
 * ===========
 * 1. LA COPIA BIT A BIT:
 *    Por defecto, si no escribes un "Constructor de Copia", C++ hace una 
 *    copia superficial. Copia la "nota con la dirección" (el puntero), 
 *    pero no clona el contenido del almacén (el Heap).
 * 
 * 2. EL DESASTRE DEL "DOUBLE FREE":
 *    Es uno de los errores más temidos en C++. Al salir de 'causarDesastre', 
 *    se ejecutan los destructores en orden inverso. El segundo destructor 
 *    intenta limpiar una basura que ya no le pertenece, provocando una 
 *    violación de segmento o un cierre abrupto.
 * 
 * 3. SOLUCIÓN PROFESIONAL (RAII Avanzado):
 *    Para evitar esto, un ingeniero tiene tres caminos:
 *    - Prohibir la copia (usando delete en el constructor de copia).
 *    - Implementar una "Copia Profunda" (Deep Copy) que pida nuevo Heap.
 *    - Usar Contadores de Referencia (como hace cv::Mat de OpenCV).
 * 
 * 4. REFLEXIÓN DE INGENIERÍA:
 *    Este snippet explica por qué 'this' es tan importante. Aquí tenemos 
 *    dos 'this' distintos en la pila, pero ambos "creen" que son dueños 
 *    de la misma dirección de memoria. La falta de comunicación entre 
 *    objetos sobre la propiedad del Heap es el origen del caos.
 */

 /*
    No había leido los COMENTARIOS y le he pedido exlicaciones por la excepción.
    ============================================================================
    Me ha dado una solución reescribiendo GestorPeligroso(const GestorPeligroso& otro)
    y la voy a poner en otro fichero.

    Me repitió los motivos del fallo y de momento los pongo aquí:
    - Es totalmente lógico que el programa "pete" ahí. Como buen Ingeniero, acabas de provocar 
      un fallo de sistema controlado para entender el diagnóstico.
    - Esa excepción es el famoso "Double Free Error" (Error de doble liberación). Aquí tienes la e
      xplicación de por qué ocurre exactamente en esa línea:
    - La trampa: Al hacer GestorPeligroso objetoB = objetoA;, ambos objetos tienen una copia idéntica 
      del puntero. Ambos dicen: "Mi dirección de datos es 0x123".
    - Primer cierre: Al terminar la función, se destruye objetoB. Su destructor ejecuta delete[] 
      datos sobre 0x123. El Sistema Operativo marca esa memoria como libre.
    - El desastre: Acto seguido, se destruye objetoA. Su destructor intenta ejecutar delete[] datos 
      sobre... ¡la misma dirección 0x123!
    - La excepción: El Sistema Operativo salta y dice: "¡Alto! Estás intentando liberar una memoria 
      que ya no te pertenece o que ya está libre". Y el programa lanza la excepción para evitar corromper el resto de la RAM.
 */

 // Compilar
 // ========
 // g++ PeligroCopiaSuperficial.cpp -o ./build/PeligroCopiaSuperficial
