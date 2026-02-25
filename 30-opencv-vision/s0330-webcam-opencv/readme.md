# OpenCV: s0330 - Captura de Vídeo en Tiempo Real (WebCam)

Este módulo marca el inicio del procesamiento dinámico. Aprendemos a gestionar el hardware de captura y a tratar un flujo de vídeo como una secuencia infinita de matrices `cv::Mat`.

---

## 📽️ El Concepto de Flujo (Stream)

A diferencia de cargar una imagen del disco, el vídeo es un **recurso compartido**. El objeto `cv::VideoCapture` actúa como un puente entre el driver del sistema operativo y nuestro código C++.

### El Bucle de Captura (`while`)
Para ver vídeo, necesitamos "engañar" al ojo humano. El bucle realiza tres acciones cíclicas:
1.  **Grabbing**: Solicita un nuevo cuadro al sensor.
2.  **Decoding**: Convierte la señal del sensor en una matriz de píxeles (BGR).
3.  **Displaying**: Dibuja la matriz en una ventana mediante `cv::imshow`.

---

## 🛑 Control de Ejecución: `cv::waitKey`

En este programa, `cv::waitKey(30)` es el director de orquesta:
*   Si el valor es muy bajo (e.g., `1`), el programa consumirá el 100% de la CPU intentando ir más rápido que la propia cámara.
*   Si el valor es `30`, limitamos la ejecución a unos **33 cuadros por segundo**, lo cual es ideal para un procesado fluido y eficiente.

---

## 🛠️ Requisitos de Hardware y Software

*   **Hardware**: Una cámara web integrada o USB conectada.
*   **Linux**: Asegúrate de tener permisos para acceder al dispositivo (usualmente en `/dev/video0`).
*   **Librerías**: OpenCV compilada con soporte para **FFMPEG** o **V4L2** (Video for Linux).

```bash
# Compilación rápida
g++ webcam.cpp -o webcam `pkg-config --cflags --libs opencv4`
./webcam
