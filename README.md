# Documentacion - Taller 2 IA: prediccion de mora (`defaulted_loan`)

Clasificacion binaria: predecir si un cliente **no pagara** su credito (`1`) o **si pagara** (`0`).
Se entrena con `train_dataset` (36000 clientes, con etiqueta) y se predice `test_dataset`
(24000 clientes, sin etiqueta). Metrica de evaluacion: **F1-score**.

## Resultado en una linea

Regresion logistica con splines sobre 7 variables -> **F1 en validacion cruzada 5-fold = 0.706**
(AUC 0.883), sin sobreajuste (F1 train = 0.706).

## Indice

| Documento | Contenido |
|---|---|
| [ejecucion.md](ejecucion.md) | Instalacion, orden de ejecucion, tiempos, solucion de problemas |
| [funcionamiento.md](funcionamiento.md) | Que hace el pipeline paso a paso y por que se tomo cada decision |
| [implementacion.md](implementacion.md) | Detalle de cada script y funcion, entradas y salidas |
| [pruebas.md](pruebas.md) | Pruebas automaticas, validaciones internas y como verificar la entrega |

Para el informe escrito ver [../GUIA_INFORME.md](../GUIA_INFORME.md) (estructura + cifras).

## Estructura del proyecto

```
taller 2nd 50/
├── Taller 2 IA 2026-2.md        enunciado
├── train_dataset.numbers        datos originales (Apple Numbers)
├── test_dataset.numbers
├── train_dataset.csv            generados por convertir.py
├── test_dataset.csv
├── requirements.txt             dependencias con version fija
├── convertir.py                 1. .numbers -> .csv
├── eda.py                       2. analisis exploratorio -> resultados/eda_*
├── modelo.py                    3. entrenamiento, CV, metricas, CSV de entrega
├── test_pipeline.py             4. pruebas
├── GUIA_INFORME.md              guia para el informe
├── resultados/                  metricas y graficas
├── docs/                        esta documentacion
└── Nombre Apellido 05-11-2026 Proyecto- 1-machine learning.csv   entrega
```

## Inicio rapido

```bash
python -m pip install -r requirements.txt
python convertir.py        # solo si faltan los .csv (~15 min)
python eda.py
python modelo.py
python test_pipeline.py
```
