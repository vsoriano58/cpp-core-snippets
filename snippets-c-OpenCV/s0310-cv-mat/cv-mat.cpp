#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

/**
 * SNIPPET S0205: cv::Mat a fondo.
 * Concepto: Gestión de cabeceras, copias superficiales y ROI.
 */

int main() {
    // [REF-01] Lectura e inicialización
    cv::Mat A = cv::imread("../assets/lena.jpg", cv::IMREAD_COLOR);
    if (A.empty()) return -1;

    // [REF-02] Copia Superficial (Solo cabecera)
    // B y C comparten los mismos píxeles que A. ¡Peligro!
    cv::Mat B(A); 
    cv::Mat C = A;

    // [REF-03] ROI (Region of Interest)
    // D apunta a una zona interna de A. No hay copia de datos.
    cv::Mat D(A, cv::Rect(10, 10, 100, 100));

    // [REF-04] Copia Profunda (Duplicado real)
    // F es independiente de A. Tiene su propia memoria.
    cv::Mat F = A.clone();

    // [REF-05] Tipos de datos (Anatomía de CV_8UC3)
    cv::Mat M(2, 2, CV_8UC3, cv::Scalar(0,0,255));

    // [REF-06] Impresión segura de estructuras
    cv::Point2f P(5, 1);
    std::cout << "Punto 2D: " << P.x << ", " << P.y << std::endl;
    
    // Para vectores, mejor usar el constructor de Mat para visualizar
    std::vector<float> v = {1.1f, 2.2f, 3.3f};
    std::cout << "Vector via Mat:\n" << cv::Mat(v) << std::endl;

    return 0;
}

// COMPILAR
// g++ cv_Mat.cpp -o ./build/cv_Mat `pkg-config --cflags --libs opencv4`