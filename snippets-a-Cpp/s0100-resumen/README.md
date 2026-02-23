# 🏔️ CimaStudio v1.0
> **Exploraciones en C++ Moderno: Del Puntero Crudo a la Gestión Inteligente.**

Bienvenido a **CimaStudio**, un repositorio educativo diseñado para documentar la evolución técnica del lenguaje C++. Este proyecto no busca la complejidad innecesaria, sino la claridad conceptual a través de "píldoras" de código o **snippets**.

## 🎯 Propósito del Proyecto
Este repositorio es un diario de aprendizaje técnico que recorre los hitos críticos del desarrollo en C++, cubriendo desde la gestión manual de memoria hasta las abstracciones modernas de la STL.

## 🗂️ Estructura de Snippets
Los módulos están organizados de forma incremental para facilitar la comprensión del "porqué" detrás de cada evolución:

*   **S0040:** El peligro de la **Copia Superficial** (Shallow Copy) y el error *Double Free*.
*   **S0070:** La llegada de la propiedad exclusiva con `std::unique_ptr`.
*   **S0080:** Gestión de recursos compartidos mediante `std::shared_ptr`.
*   **S0090:** Resolución de ciclos de referencia con el observador `std::weak_ptr`.
*   **S0100:** **RESUMEN FINAL**: Ecosistema completo de Smart Pointers.
*   **S0110:** Introducción a la **Programación Genérica** (Templates de función).

## 🚀 Cómo utilizar este material
Cada snippet es un programa independiente y funcional.
1. Navega a la carpeta del snippet deseado.
2. Compila usando `g++` (se recomienda evitar paréntesis en las rutas de directorio).
3. Lee los **Comentarios Técnicos** al final de cada archivo para entender la teoría aplicada.

```bash
# Ejemplo de compilación segura
g++ -std=c++17 main.cpp -o programa_ejecutable
