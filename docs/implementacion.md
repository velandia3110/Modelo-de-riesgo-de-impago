# Implementacion

Tres scripts independientes, sin clases ni configuracion externa. Las constantes compartidas
viven en `modelo.py` y `eda.py` las importa.

## `convertir.py`

Convierte `train_dataset.numbers` y `test_dataset.numbers` en `.csv`.

| Funcion | Que hace |
|---|---|
| `lineas_desde_numbers(nombre)` | Abre el `.numbers` con `numbers_parser.Document`, toma la primera tabla de la primera hoja y devuelve el encabezado unido con comas mas el contenido de la columna A de cada fila |
| `lineas_desde_csv_crudo(nombre)` | Igual, pero a partir del CSV intermedio. Si detecta que el CSV ya esta convertido (hay datos fuera de la 1a columna), aborta para no destruirlo |

Flujo principal: obtiene las lineas -> `"\n".join(lineas)` -> `pd.read_csv(io.StringIO(...))`
(une los registros partidos por el salto de linea entre comillas) -> `to_csv(index=False)`.

Argumento opcional: `--crudo` (usa `lineas_desde_csv_crudo`).

## `modelo.py`

### Constantes

| Nombre | Valor | Uso |
|---|---|---|
| `SEED` | 42 | semilla de CV y modelos (reproducibilidad) |
| `NUCLEO` | 7 variables | conjunto final de variables |
| `SALIDA` | `Path("resultados")` | carpeta de salidas (se crea si no existe) |
| `NOMBRE_ENTREGA` | `"Nombre Apellido 05-11-2026 ..."` | nombre del CSV de entrega - **cambiar antes de entregar** |

### Funciones

**`features(df) -> DataFrame`**
Recibe el DataFrame crudo (train o test) y devuelve la matriz numerica de variables:
7 numericas originales, 3 booleanas convertidas a 0/1 y 12 meta-variables (ver
[funcionamiento.md](funcionamiento.md#3-ingenieria-de-variables-features)). Es una funcion pura:
se aplica igual a train y test, por lo que no hay fuga de informacion (no usa la etiqueta ni
estadisticas del conjunto).

**`mejor_umbral(y, p) -> (umbral, f1)`**
Recorre umbrales de 0.10 a 0.89 en pasos de 0.01 y devuelve el que maximiza `f1_score(y, p >= u)`.

**`curva_lift(y, p, ax, nombre)`**
Ordena a los clientes por probabilidad descendente y grafica
`lift(k) = tasa de mora en el top k / tasa de mora global`.

**`main()`**
1. Carga los CSV; mapea `defaulted_loan` a 0/1 con `if/else` (acepta `True` booleano o texto).
2. `X = features(train)`, `X_test = features(test)`.
3. Define el diccionario `modelos = {nombre: (estimador, columnas)}`.
4. Por cada modelo:
   - `cross_val_predict(..., method="predict_proba")` -> probabilidades fuera de fold.
   - `mejor_umbral` -> umbral y F1 de CV.
   - F1 por fold con ese umbral -> desviacion estandar.
   - `fit` con todo el train -> F1 en train (diagnostico de sobreajuste).
   - Guarda precision, recall, AUC y matriz de confusion.
5. Escribe `metricas.json`, `roc_lift.png`, `matriz_confusion.png` e `importancia_variables.*`.
6. Elige `mejor = argmax F1 CV`, predice el test, aplica el umbral, valida con `assert` la forma
   (24000 x 2) y los valores {0, 1}, y escribe el CSV de entrega.

### Hiperparametros

| Modelo | Parametros | Criterio |
|---|---|---|
| LogisticRegression | `max_iter=2000`, escalado previo | por defecto (C=1) |
| RandomForest | `n_estimators=400, min_samples_leaf=5` | hojas minimas para limitar sobreajuste |
| LightGBM | `n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=80, subsample=0.8, colsample_bytree=0.8` | arboles pequeños y regularizados |
| Spline + logistica | `SplineTransformer(n_knots=6)`, `C=1` | se probaron C = 0.1 / 1 / 10 (0.704 / 0.706 / 0.705) y se eligio 1 |

Las variantes descartadas (interacciones polinomicas, hijos como categoria, LightGBM monotono) se
evaluaron en experimentos aparte; sus resultados estan en [GUIA_INFORME.md](../GUIA_INFORME.md).

## `eda.py`

Importa `NUCLEO` y `SALIDA` de `modelo.py`. Genera, en orden:

1. `eda_resumen.csv` - `describe(include="all")`; imprime nulos, tasa de mora y valores negativos.
2. `eda_tasa_mora.png` - tasa de mora por decil (o por valor si hay menos de 10 distintos).
3. `eda_distribuciones.png` - histogramas por clase de las variables numericas.
4. `eda_correlaciones.{csv,png}` - correlacion de Pearson entre `NUCLEO` y la etiqueta.
5. `eda_auc_variables_texto.csv` - AUC de CV (5 folds) de una logistica con `TargetEncoder` para
   cada variable de texto. El `TargetEncoder` esta dentro del pipeline, asi que la codificacion se
   aprende solo con los folds de entrenamiento (sin fuga).

## Decisiones de diseño

- **Sin informacion externa:** todas las variables derivan de las columnas del dataset.
- **Sin fuga de datos:** las transformaciones que aprenden de los datos (escalado, splines,
  target encoding) van dentro de `Pipeline`, de modo que en CV se ajustan solo con los folds de
  entrenamiento.
- **Umbral elegido en CV, no en test:** el test no tiene etiquetas y no se usa para ajustar nada.
- **Modelo simple frente a complejo:** a igual F1 se prefiere el que generaliza (train ≈ CV).
