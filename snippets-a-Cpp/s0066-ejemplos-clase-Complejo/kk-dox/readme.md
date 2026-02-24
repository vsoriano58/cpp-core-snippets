# Guía Técnica: Clase Complejo en C++

Este documento detalla la implementación de una clase `Complejo` diseñada para el manejo de números imaginarios, cubriendo la gestión de memoria y la sobrecarga de operadores según los estándares de la [ISO C++](https://isocpp.org).

## 1. Código Fuente Completo

```cpp
#include <iostream>
#include <cmath>     // Para std::sqrt
#include <utility>   // Para std::move

class Complejo {
private:
    double real;
    double img;

public:
    // --- CONSTRUCTORES ---

    // 1. Por defecto
    Complejo() : real(0.0), img(0.0) {
        std::cout << "[LOG] Constructor por defecto\n";
    }

    // 2. Parametrizado (Normal)
    Complejo(double r, double i) : real(r), img(i) {    //[REF-01]
        std::cout << "[LOG] Constructor normal\n";
    }

    // 3. Constructor de Copia
    // Sintaxis: Complejo(const Complejo& otro)
    Complejo(const Complejo& otro) : real(otro.real), img(otro.img) {
        std::cout << "[LOG] Constructor de copia\n";
    }

    // 4. Constructor de Movimiento (Move)
    // Sintaxis: Complejo(Complejo&& otro)
    Complejo(Complejo&& otro) noexcept : real(otro.real), img(otro.img) {
        otro.real = 0; 
        otro.img = 0;
        std::cout << "[LOG] Constructor de movimiento\n";
    }

    // --- MÉTODOS ---

    double modulo() const {
        return std::sqrt(real * real + img * img);
    }

    Complejo sumar(const Complejo& otro) const {
        return Complejo(this->real + otro.real, this->img + otro.img);
    }

    // --- SOBRECARGA DE OPERADORES ---

    // Operador +
    Complejo operator+(const Complejo& otro) const {
        return Complejo(this->real + otro.real, this->img + otro.img);
    }

    // Operador = (Asignación por copia)
    Complejo& operator=(const Complejo& otro) {
        if (this != &otro) {
            real = otro.real;
            img = otro.img;
        }
        std::cout << "[LOG] Asignacion por copia\n";
        return *this;
    }

    // Operador << para impresión directa
    friend std::ostream& operator<<(std::ostream& os, const Complejo& c) {
        os << "(parte real: " << c.real 
           << ", parte imaginaria: " << c.img 
           << ", modulo: " << c.modulo() << ")";
        return os;
    }
};

int main() {
    // 1. Creación con inicialización directa [REF-01]
    Complejo c1(3.0, 4.0); 
    
    // 2. Uso del operador << y método modulo
    std::cout << "C1: " << c1 << " | Modulo: " << c1.modulo() << std::endl;

    // 3. Prueba rápida de suma y asignación
    Complejo c2;
    c2 = c1 + c1; 
    std::cout << "C1 + C1 = " << c2 << std::endl;

    // 4. Prueba de movimiento (eficiencia)
    Complejo c3 = std::move(c1);
    std::cout << "C3 (movido): " << c3 << std::endl;

    return 0;
}

```

# 2. Guía de Conceptos para Ingeniería

## A. La Lista de Inicialización
En lugar de asignar valores dentro del cuerpo del constructor `{ real = r; }`, usamos la sintaxis : `real(r), img(i)`. [REF-01]

* **Por qué**: En ingeniería de software, esto se conoce como inicialización directa. Es más eficiente porque el objeto se crea con los valores correctos desde el primer ciclo de reloj, evitando una asignación posterior innecesaria en la memoria.

## B. Análisis del Constructor de Copia (&)

```cpp
Complejo(const Complejo& otro) : real(otro.real), img(otro.img)
```
* **const**: Es un contrato de solo lectura. Garantizamos que el objeto original no sufrirá cambios accidentales durante la duplicación.
* **Referencia `(&)`**: Evita el problema de la "recursividad infinita". Si no usáramos la referencia, C++ intentaría crear una copia para pasarla al constructor de copia, llamándose a sí mismo infinitamente.
* **Uso**: Se dispara en `Complejo c2 = c1;`.

## C. Análisis del Constructor de Movimiento (&&)

```cpp
Complejo(Complejo&& otro) noexcept : real(otro.real), img(otro.img)
```
* **Semántica de Movimiento**: Introducida en C++11, permite "robar" los datos de un objeto temporal (rvalue).
* **&&**: Identifica que el objeto otro es temporal y que sus recursos pueden ser transferidos para evitar duplicaciones costosas.
* **noexcept**: Es una garantía para el compilador de que la operación es segura, fundamental para optimizar el uso de std::vector.

## D. La función friend y el Operador `<<`

El operador de flujo `<<` requiere acceder a la parte privada de la clase, pero no puede ser un método de la propia clase porque el primer operando es `std::cout` (un objeto externo de tipo `std::ostream`).

* **Solución**: Declararlo como friend. Esto otorga permisos de "invitado" a una función externa para que la sintaxis `std::cout << objeto` sea posible.

## E. Precisión Matemática

El método `modulo()` utiliza la función `std::sqrt` de la librería `cmath`. Es fundamental que este método sea `const` para asegurar la **const-correctness**, permitiendo que el objeto sea leído pero no modificado durante el cálculo.

## Resumen de lo logrado

* **Gestión de memoria**: Tenemos cubiertos todos los casos (`copia y movimiento`).
* **Encadenamiento:** El `operator=` está correctamente implementado devolviendo `Complejo&`.
* **Const-Correctness:** El uso de `const` en los métodos de lectura `(modulo, sumar)` es el correcto.










