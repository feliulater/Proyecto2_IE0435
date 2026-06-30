# Proyecto 2 IE0435 - Predicción de Demanda Eléctrica

Proyecto desarrollado para el curso **IE0435 Inteligencia Artificial Aplicada a la Ingeniería Eléctrica**.

## Descripción

Este proyecto implementa un agente para el análisis y predicción de series de tiempo aplicadas a Ingeniería Eléctrica, usando como caso de estudio la demanda eléctrica nacional. El sistema trabaja con datos del CENCE y permite descargar, procesar, visualizar, analizar y predecir la demanda eléctrica en MW.

El proyecto toma como referencia un agente previo desarrollado para predicción de demanda eléctrica, pero agrega herramientas propias enfocadas en detección de anomalías, corrección de datos y predicción mediante árboles de regresión.

## Herramientas implementadas

Las herramientas principales seleccionadas fueron:

1. **Detección y corrección de anomalías**
2. **Predicción con árboles de regresión**

Además, se incorporaron herramientas de apoyo:

* Descarga automática de datos del CENCE.
* Preprocesamiento de datos.
* Visualización de demanda real y programada.
* Comparación contra demanda programada del CENCE.
* Comparación contra promedio histórico horario.
* Cálculo de métricas de error.
* Selección del mejor modelo según RMSE.
* Agente interactivo por consola.

## Estructura del proyecto

```text
Proyecto2_IE0435/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── src/
│   ├── agent.py
│   ├── anomaly_detection.py
│   ├── data_downloader.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Flujo general

El flujo implementado es:

```text
Descarga de datos
→ Carga de datos
→ Preprocesamiento
→ Visualización inicial
→ Detección de anomalías
→ Corrección de anomalías
→ Ingeniería de características
→ Entrenamiento de modelos
→ Evaluación
→ Selección del mejor modelo
```

## Modelos evaluados

Se evaluaron cuatro enfoques:

* Árbol de regresión con demanda programada.
* Árbol de regresión sin demanda programada.
* Demanda programada del CENCE.
* Promedio histórico horario.

## Resultados principales

Para el periodo analizado, el mejor modelo fue:

```text
Árbol sin demanda programada
MAE: 39.017 MW
RMSE: 45.701 MW
MAPE: 2.906 %
```

Este resultado indica que el modelo basado en variables históricas, especialmente la demanda con rezago de 15 minutos, logró el menor error de predicción.

## Ejecución

Primero se crea y activa un entorno virtual:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Luego se instalan las dependencias:

```powershell
pip install -r requirements.txt
```

Para ejecutar el agente:

```powershell
py main.py
```

El agente muestra un menú con las siguientes opciones:

```text
1. Ejecutar flujo completo
2. Descargar datos del CENCE
3. Ejecutar detección/corrección de anomalías
4. Entrenar y evaluar modelos
5. Mostrar mejor modelo guardado
0. Salir
```

## Archivos principales

* `main.py`: ejecuta el agente.
* `src/agent.py`: agente orquestador del proyecto.
* `src/data_downloader.py`: descarga datos del CENCE.
* `src/preprocessing.py`: limpia y prepara los datos.
* `src/anomaly_detection.py`: detecta y corrige anomalías.
* `src/feature_engineering.py`: crea variables para el modelo.
* `src/models.py`: entrena y evalúa modelos de predicción.
* `src/visualization.py`: genera gráficas.

## Oportunidades de mejora

Como trabajo futuro se podrían implementar:

* Predicción multihorizonte.
* Modelos adicionales como KNN, SVR o Random Forest.
* Integración con un framework de agentes como LangChain o CrewAI.
* Interfaz gráfica o dashboard.
* Uso de datos meteorológicos o calendario de feriados.
* Evaluación con periodos más extensos de datos.
