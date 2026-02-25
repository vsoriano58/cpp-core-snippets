#include <iostream>
#include <utility>
#include <cmath> // Necesario para sqrt()

class Complejo {
private:
    double real;
    double img;

public:
    // 1. Constructor por defecto
    Complejo() : real(0.0), img(0.0) {
        std::cout << "[LOG] [Default Constructor]\n";
    }

    // 2. Constructor normal (parametrizado)
    Complejo(double r, double i) : real(r), img(i) {    // [REF-01]
        std::cout << "[LOG] Constructor]\n";
    }

    // 3. Constructor de copia
    Complejo(const Complejo& otro) : real(otro.real), img(otro.img) {
        std::cout << "[LOG] Copy Constructor]\n";
    }

    // 4. Constructor de movimiento (move)
    Complejo(Complejo&& otro) noexcept : real(otro.real), img(otro.img) {
        // En tipos primitivos no hay "limpieza" que hacer, pero se marcan los
        // valores originales por buena práctica.
        otro.real = 0;
        otro.img = 0;
        std::cout << "[Move Constructor]\n";
    }

    // --- MÉTODOS ---

    // Método sumar complejos
    Complejo sumar(const Complejo& otro) const {
        return Complejo(this->real + otro.real, this->img + otro.img);
    }

    // --- NUEVO MÉTODO MÓDULO ---
    // El módulo es la raíz cuadrada de (a² + b²)
    double modulo() const {
        return std::sqrt(real * real + img * img);
    }


    // --- OPERADORES ---

    // Redefinir operador = (Asignación de copia)
    Complejo& operator=(const Complejo& otro) {
        if (this != &otro) {
            real = otro.real;
            img = otro.img;
        }
        std::cout << "[Copy Assignment]\n";
        // Retornamos una referencia al objeto actual (*this)
        // Esto permite el encadenamiento: a = b = c;
        return *this;
    }

    // Redefinir operador +
    Complejo operator+(const Complejo& otro) const {
        return Complejo(this->real + otro.real, this->img + otro.img);
    }

    // --- REDEFINICIÓN DEL OPERADOR << ---
    // Se declara como friend para acceder a 'real' e 'img' siendo privado
    friend std::ostream& operator<<(std::ostream& os, const Complejo& c) {
        os << "(parte real: " << c.real 
           << ", parte imaginaria: " << c.img 
           << ", modulo: " << c.modulo() << ")";
        return os;
    }

    // --- REDEFINICIÓN DEL OPERADOR > ---
    bool operator>(const Complejo& otro) const {
        return this->modulo() > otro.modulo();
    }

    // Método auxiliar para imprimir
    void mostrar() const {
        std::cout << real << " + " << img << "i" << std::endl;
    }
};

int main() {
    // 1. Prueba de Constructor Parametrizado y Operador <<
    std::cout << "--- 1. Creación y Visualización ---" << std::endl;
    Complejo c1(3.0, 4.0); 
    std::cout << "c1: " << c1 << std::endl;

    // 2. Prueba del método modulo() directamente
    std::cout << "\n--- 2. Prueba de modulo() ---" << std::endl;
    double m = c1.modulo();
    std::cout << "El valor del modulo de c1 es: " << m << std::endl;

    // 3. Prueba de operaciones aritméticas
    std::cout << "\n--- 3. Operaciones (+, =) ---" << std::endl;
    Complejo c2(1.0, 2.0);
    Complejo c3; // Constructor por defecto
    
    c3 = c1 + c2; // Suma y asignación
    std::cout << "c1 + c2 = " << c3 << std::endl;

    // 4. Prueba de Semántica de Movimiento
    std::cout << "\n--- 4. Movimiento (std::move) ---" << std::endl;
    // Movemos los datos de c3 a un nuevo objeto c4
    Complejo c4 = std::move(c3);
    
    std::cout << "c4 (movido desde c3): " << c4 << std::endl;
    std::cout << "c3 (tras el movimiento): " << c3 << std::endl;

    // 5. Prueba de encadenamiento del operador <<
    std::cout << "\n--- 5. Encadenamiento ---" << std::endl;
    std::cout << "Listado: " << c1 << " | " << c2 << std::endl;

    // XX. Para el operador =
    Complejo c10(5.0, 8.0);
    Complejo c20(1.0, 1.0);
    Complejo c30;

    std::cout << "Antes de asignar:" << std::endl;
    std::cout << "c1: " << c10 << std::endl;
    std::cout << "c2: " << c20 << std::endl;

    // --- USO DEL OPERADOR = ---
    
    // Asignación simple
    c20 = c10; 
    
    // Asignación encadenada (posible gracias a que retornamos Complejo&)
    c30 = c20 = c10;

    std::cout << "\nDespues de asignar (c2 = c1):" << std::endl;
    std::cout << "c2: " << c20 << std::endl;
    std::cout << "c3: " << c30 << std::endl;


    return 0;
}

// Compilar: g++ Complejo.cpp -o ./build/Complejo
