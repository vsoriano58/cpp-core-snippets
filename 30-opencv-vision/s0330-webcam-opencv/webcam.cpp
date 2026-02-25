#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
/** 
 * PROGRAMA: 04_HolaMundo_OpenCV
 * OBJETIVO: Abrir la cámara y mostrar el flujo de video en una ventana.
 * CLAVE: El bucle while y el objeto cv::Mat (La Matriz).
 */

int main() {
    // 1. EL OJO DEL PROGRAMA
    // Creamos el objeto capturador. El '0' indica la cámara por defecto.
    cv::VideoCapture cap(0);

    // 2. EL SEGURO DE VIDA
    // Siempre debemos comprobar si el hardware ha respondido.
    if (!cap.isOpened()) {
        std::cerr << "Error: No se pudo acceder a la camara." << std::endl;
        return -1;
    }

    // 3. EL CONTENEDOR (La Sombra y el Cuerpo)
    // Reservamos la cabecera en el Stack para la matriz de píxeles.
    cv::Mat frame;

    std::cout << "Presiona cualquier tecla para salir..." << std::endl;

    // 4. EL BUCLE DE CAPTURA
    // Mientras la cámara esté abierta, "robamos" luz del sensor.
    while (true) {
        cap >> frame; // Volcamos el flujo de la cámara en la matriz 'frame'

        if (frame.empty()) break; // Si el frame viene vacío, salimos.

        // 5. LA VENTANA NATIVA
        // Mostramos la matriz en una ventana creada por OpenCV (HighGUI).
        cv::imshow("Ventana Nativa de OpenCV", frame);

        // Esperamos 30ms. Si el usuario pulsa una tecla, salimos del bucle.
        if (cv::waitKey(30) >= 0) break;
    }

    // Al salir del ámbito, 'cap' se libera automáticamente (Destructor).
    return 0;
}

/**
 * REFERENCIAS TÉCNICAS (S0330):
 * 
 * [REF-01] VideoCapture(0): El índice '0' abre la cámara predeterminada del sistema. 
 *          Si tuvieras una cámara USB externa, podrías probar con el índice '1'.
 * 
 * [REF-02] cap.isOpened(): Paso crítico de seguridad. Si el driver de la cámara 
 *          está ocupado por otra app o no existe, el programa fallará aquí.
 * 
 * [REF-03] OPERADOR >>: Es un alias de 'cap.read(frame)'. Extrae el siguiente 
 *          cuadro del flujo de vídeo y lo decodifica en una matriz BGR.
 * 
 * [REF-04] cv::waitKey(30): Vital por dos razones:
 *          1. Genera un retraso de 30ms (aprox. 33 FPS) para que el ojo humano 
 *             perciba fluidez.
 *          2. Permite que 'cv::imshow' procese los eventos de la ventana (dibujado). 
 *             Sin esta línea, la ventana se quedaría congelada en blanco.
 */
