#include <iostream>
#include <string>

/** 
 * PROGRAMA: 02_Memoria_Dinamica
 * OBJETIVO: Diferenciar visualmente la persistencia en el Stack vs el Heap.
 * CLAVE: Uso de punteros (*) y el operador 'new'.
 */

int main() {
    // 1. VARIABLE EN EL STACK (La Pila)
    // Es automática. Se crea y se destruye sola al llegar al final de las llaves {}.
    int edadStack = 30; 

    // 2. VARIABLE EN EL HEAP (El Montón)
    // Usamos 'new'. No estamos creando un entero, sino "pidiendo sitio" para uno.
    // El puntero 'p_edadHeap' solo guarda la DIRECCIÓN de ese sitio.
    int* p_edadHeap = new int(45); 

    std::cout << "Valor en el Stack: " << edadStack << std::endl;
    std::cout << "Direccion de memoria en el Stack: " << &edadStack << std::endl;
    std::cout << "Valor en el Heap (via puntero): " << *p_edadHeap << std::endl;
    std::cout << "Direccion de memoria en el Heap: " << p_edadHeap << std::endl;

    // 3. LA SUTILEZA TÉCNICA
    // Si estuviéramos en una función de CimaStudio, al terminar, 'edadStack' 
    // moriría. Pero el entero de 'p_edadHeap' seguiría vivo en la RAM.

    // 4. LIMPIEZA MANUAL
    // En C++ puro, lo que creas con 'new' debes borrarlo con 'delete'.
    // (Nota: En Qt, muchas veces el sistema lo hace por nosotros, pero es 
    // vital entender que alguien debe "limpiar la mesa").
    delete p_edadHeap;

    return 0;
}