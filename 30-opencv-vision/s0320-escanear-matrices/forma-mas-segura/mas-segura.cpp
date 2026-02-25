#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

/**
 * SNIPPET S0210-C: Escaneo mediante Acceso Aleatorio (.at)
 * 
 * CONCEPTO: El método más seguro y legible. 
 * Ideal para algoritmos que no siguen un orden lineal (saltos entre píxeles).
 */

int main() {
    // [REF-01] Carga de imagen en color (BGR)
    cv::Mat img = cv::imread("../assets/lena.jpg", cv::IMREAD_COLOR);
    if (img.empty()) return -1;

    // [REF-02] Recorrido mediante coordenadas (i, j)
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            
            // [REF-03] El método .at requiere especificar el tipo de dato.
            // Para imágenes en color de 8 bits usamos cv::Vec3b (vector de 3 bytes).
            cv::Vec3b &pixel = img.at<cv::Vec3b>(i, j);

            // [REF-04] Modificación directa de los canales Blue, Green, Red
            // Ejemplo: Poner a cero el canal azul (B=0, G=1, R=2)
            pixel[0] = 0; 
            
            // También podríamos hacer:
            // img.at<cv::Vec3b>(i, j)[1] = 255; // Forzar verde al máximo
        }
    }

    cv::imshow("Escaneo con .at - Seguro pero Lento", img);
    cv::waitKey(0);

    return 0;
}

/**
 * REFERENCIAS TÉCNICAS (S0210-C):
 * 
 * [REF-01] CARGA BGR: Al cargar con IMREAD_COLOR, OpenCV organiza los canales 
 *          como Blue (0), Green (1) y Red (2).
 * 
 * [REF-02] RECORRIDO (i, j): A diferencia de los punteros, aquí usamos 
 *          coordenadas (fila, columna) naturales. Es mucho más legible 
 *          para implementar algoritmos geométricos o de rotación.
 * 
 * [REF-03] MÉTODO .at<type>: Es una función de plantilla (template). 
 *          Debes decirle exactamente qué hay en esa coordenada. 
 *          'cv::Vec3b' indica un vector de 3 bytes (unsigned char).
 *          Uso de REFERENCIA (&): Al usar '&pixel', cualquier cambio en la 
 *          variable local afecta directamente a la imagen original.
 * 
 * [REF-04] MANIPULACIÓN DE CANALES: Al poner pixel[0] = 0, estamos eliminando 
 *          toda la información del canal Azul. El resultado visual es una 
 *          imagen con predominancia de amarillos (Rojo + Verde).
 */
