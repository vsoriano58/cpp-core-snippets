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

/**
 * REFERENCIAS TÉCNICAS (S0205):
 * 
 * [REF-01] cv::imread: Carga la imagen. Si falla, Mat.empty() es true. 
 *          IMPORTANTE: OpenCV usa BGR (Blue-Green-Red) por defecto, no RGB.
 * 
 * [REF-02] COPIA SUPERFICIAL: 'B(A)' y 'C = A' no duplican los píxeles. 
 *          Si modificas un píxel en B, también cambiará en A y C. 
 *          Es una gestión de memoria ultra-eficiente basada en contadores de referencia.
 * 
 * [REF-03] ROI (Region of Interest): Permite trabajar sobre una sub-ventana. 
 *          D no es una imagen nueva, es una "vista" de A. 
 *          Ideal para procesar solo caras o matrículas sin gastar RAM.
 * 
 * [REF-04] CLONE / COPYTO: Es la única forma de crear una independencia total. 
 *          F tiene su propio bloque de memoria en el Heap.
 * 
 * [REF-05] CV_8UC3: La nomenclatura clave de OpenCV.
 *          8U = 8 bits Unsigned (0-255).
 *          C3 = 3 Canales (B, G, R).
 * 
 * [REF-06] VISUALIZACIÓN: 'cv::Mat(v)' es un truco brillante para usar el 
 *          operador '<<' de OpenCV y formatear vectores de C++ de forma legible.
 */

// COMPILAR
// g++ cv-mat.cpp -o ./build/cv-mat `pkg-config --cflags --libs opencv4`