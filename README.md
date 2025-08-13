# CEIA-AprendizajeRefuerzoI
## Desafío Práctico: Aprendizaje por Refuerzo en FrozenLake con Q-Learning

### Autores

 - Agustín López Fredes

### Descripción del Problema
Este proyecto resuelve el entorno FrozenLake-v1 de Gymnasium utilizando el algoritmo de Q-Learning, una técnica de Aprendizaje por Refuerzo basada en Diferencia Temporal (Temporal Difference).
El problema consiste en un agente que debe navegar por un lago congelado representado como una grilla 4x4 (por defecto en FrozenLake-v1). El agente comienza en la posición inicial 'S' y debe llegar al objetivo 'G' evitando caer en agujeros 'H'. Las casillas 'F' son seguras al deshabilitar el parámetro is_slippery.

### Recompensas originales:

- +1 por llegar al objetivo.
- 0 por caer en un agujero o por pasos intermedios.

Se modifica con un wrapper para agregar:

- Penalización por paso: -0.01 (para incentivar caminos cortos).
- Penalización por agujero: -0.3 (para penalizar fallos).

El objetivo es aprender una política óptima que maximice la recompensa acumulada, alcanzando el objetivo en el menor número de pasos posible mientras minimiza caídas.

### Enfoque de Solución

Se implementa Q-Learning off-policy con:
- Tabla Q inicializada como defaultdict para estados no visitados.
- Política ε-greedy con decaimiento exponencial (de 1.0 a 0.01) para balancear exploración-explotación.
- Tasa de aprendizaje α con decaimiento (de 0.2 a 0.01).
- Factor de descuento γ = 0.99.
- 200.000 episodios de entrenamiento.

Se trackean métricas como recompensas, tasa de éxito (basada en llegada al objetivo), caídas en agujeros y longitud de episodios.

### Inconvenientes encontrados:

- Convergencia lenta inicial debido a alta exploración: Solucionado con decaimiento de ε y α.
- Recompensas ruidosas: Usada media móvil (ventana=1000) para suavizar gráficos.
- Detección de éxito: Inicialmente basada en recompensa, pero ajustada a variable reached_goal para precisión.


### Resultados

- Tasa de éxito final: ~99% (basado en pruebas cada 10k episodios).

https://github.com/Agustinlopezf/CEIA-AprendizajeRefuerzoI/blob/main/frozenlake_qlearning.png

### Análisis de los Gráficos

Los gráficos generados muestran la evolución del entrenamiento a lo largo de 200,000 episodios. Se utilizan curvas de media móvil (ventana de 1,000 episodios) para suavizar las variaciones y destacar tendencias, junto con los datos crudos en opacidad baja para contexto.

- Convergencia de Recompensas: Esta gráfica ilustra cómo la recompensa promedio por episodio aumenta progresivamente. Inicialmente negativa debido a exploración aleatoria y penalizaciones, converge hacia valores cercanos a 1 (recompensa máxima por objetivo menos penalizaciones por pasos mínimos). La media móvil resalta la mejora estable, indicando que el agente aprende a maximizar recompensas evitando caminos largos y fallos.
- Convergencia de Tasa de Éxito: Muestra el porcentaje de episodios donde el agente alcanza el objetivo ('G'). Comienza bajo (~0) y asciende rápidamente a ~1, demostrando que la política Q-learning se optimiza para el éxito consistente. Las oscilaciones iniciales reflejan la exploración, pero la media móvil confirma la convergencia.
- Convergencia de Caídas en Agujeros: Representa la tasa de episodios donde el agente cae en un agujero. Parte alta (~1) y desciende a casi 0, evidenciando que el algoritmo aprende a evitar estados terminales negativos. La media móvil suaviza el ruido, destacando la reducción efectiva de errores.


### Instrucciones de Ejecución

Instalar dependencias: pip install gymnasium numpy matplotlib tqdm.
Ejecutar el script en Python o Colab.
El entorno no requiere instalación adicional (Gymnasium maneja FrozenLake internamente).

