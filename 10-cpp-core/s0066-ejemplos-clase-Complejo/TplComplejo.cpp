/**
 * @file TplComplejo.cpp
 * @brief Integración de Templates con Clases Personalizadas.
 * @author alcón68
 * 
 * CONCEPTO: El template 'maximo' es agnóstico al tipo. Al definir el 
 * operador '>' en Complejo, la plantilla puede procesarlos sin cambios.
 */

#include <iostream>
#include <cmath>
#include <utility>

class Complejo {
private:
    double real;
    double img;

public:
    Complejo(double r, double i) : real(r), img(i) {}

    double modulo() const {
        return std::sqrt(real * real + img * img);
    }

    // VITAL: El template 'maximo' requiere este operador
    bool operator>(const Complejo& otro) const {
        return this->modulo() > otro.modulo();
    }

    friend std::ostream& operator<<(std::ostream& os, const Complejo& c) {
        os << "(" << c.real << " + " << c.img << "i, mod: " << c.modulo() << ")";
        return os;
    }
};

// Nuestro Template profesional (paso por referencia)
template <typename T>
const T& maximo(const T& a, const T& b) {
    return (a > b) ? a : b;
}

int main() {
    Complejo c1(3.0, 4.0); // Modulo 5
    Complejo c2(1.0, 2.0); // Modulo 2.23

    std::cout << "Comparando complejos con Template:\n";
    std::cout << "c1: " << c1 << "\n";
    std::cout << "c2: " << c2 << "\n\n";

    // El compilador instancia 'maximo<Complejo>'
    Complejo ganador = maximo(c1, c2);

    std::cout << "El complejo con mayor modulo es: " << ganador << std::endl;

    return 0;
}

// Compilar: g++ TplComplejo.cpp -o ./build/TplComplejo
