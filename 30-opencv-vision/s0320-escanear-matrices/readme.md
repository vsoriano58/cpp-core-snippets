# Escaneo de Matrices: Comparativa de Rendimiento

En este módulo analizamos las tres formas de recorrer los píxeles de una `cv::Mat`. La elección del método depende de si priorizamos la **velocidad**, la **seguridad** o la **legibilidad**.


| Método | Carpeta | Velocidad | Seguridad | Uso Ideal |
| :--- | :--- | :--- | :--- | :--- |
| **Puntero C** | `forma-mas-rapida` | 🏎️ Máxima | ⚠️ Baja | Filtros en tiempo real |
| **Método .at** | `forma-mas-segura` | 🐢 Lenta | ✅ Alta | Prototipado y Debug |
| **Punteros Fila**| `punteros-a-filas` | 🚀 Alta | 🟡 Media | Procesamiento estándar |

---
**Dato técnico**: El acceso directo por punteros es hasta 10 veces más rápido que el método `.at` en imágenes de alta resolución debido a que no realiza el cálculo de offset en cada píxel.

### 🏁 Conclusión de la Trilogía de Escaneo
Con esto el repositorio ya tenemos una base de **Visión Artificial** envidiable. El lector ya sabe:
1.  Cómo se guarda la imagen (`cv::Mat`).
2.  Cómo recorrerla a máxima velocidad (Punteros).
3.  Cómo recorrerla con total seguridad (`.at`).
4.  Cómo hacerlo de forma equilibrada (Punteros a filas).

**¿Qué te parece si el siguiente paso es "jugar" con la WebCam?** Podríamos crear un programa que capture el vídeo en vivo y aplique uno de estos tres escaneos para procesar los frames en tiempo real. 

**¿Atacamos la captura de vídeo o prefieres ver operaciones morfológicas (Erosión/Dilatación) con imágenes estáticas?**
Las respuestas de la IA pueden contener errores. Más información




