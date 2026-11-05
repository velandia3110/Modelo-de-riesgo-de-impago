# Funcionamiento

## Vision general

```
 .numbers ──► convertir.py ──► .csv ──┬──► eda.py ────► resultados/eda_*  (analisis)
                                      │
                                      └──► modelo.py ─► features() ─► 4 modelos x CV 5-fold
                                                         │
                                                         ├─► umbral optimo de F1 por modelo
                                                         ├─► metricas.json, ROC, lift, matriz
                                                         └─► mejor modelo ─► CSV de entrega
```

## 1. Datos de entrada

- **Train:** 36000 filas x 23 columnas (22 predictoras + `defaulted_loan`).
- **Test:** 24000 filas x 22 columnas (sin `defaulted_loan`).
- Tipos de columnas:
  - numericas: `age`, `salary`, `number_of_credit_cards`, `years_in_current_job`,
    `credit_score`, `number_of_children`, `years_of_education`.
  - booleanas: `has_criminal_records`, `is_married`, `smoker`.
  - texto: `name`, `company`, `current_job`, `email_domain`, `credit_card_provider`, `address`,
    `car_licence_plate`.
  - fecha: `application_date`, `credit_card_expire` (mm/aa).
  - identificadores: `application_id`, `phone_number`, `credit_card_number`.

### Problema de formato encontrado

Los archivos llegaron en formato **Apple Numbers**. Ademas, dentro del Numbers cada registro era
una linea de CSV crudo en la columna A, y como `address` contiene un salto de linea (calle /
ciudad) entre comillas, **cada registro quedaba partido en 2 filas** (72000 en vez de 36000).

Solucion (`convertir.py`): tomar todas las lineas, unirlas con `\n` y volver a parsearlas con el
lector CSV de pandas, que respeta los saltos de linea dentro de comillas. Asi se recuperan
exactamente 36000 y 24000 registros sin nulos.

## 2. Analisis exploratorio (`eda.py`)

Hallazgos que guiaron el modelado:

- Clase positiva (mora) = **29.7%** -> desbalance moderado; por eso se usa F1 y validacion
  estratificada.
- **7 variables con senal**, todas con relacion monotona con la mora:
  `salary` (−), `credit_score` (−), `number_of_children` (+, con salto fuerte en 4+),
  `years_of_education` (−), `years_in_current_job` (−), `age` (−), `has_criminal_records` (+).
- **Variables sin senal** (AUC ≈ 0.50 con target encoding en CV): estado, profesion, empresa,
  dominio de correo, proveedor de tarjeta, formato de placa, hora/dia de solicitud, nombre,
  estado civil, fumador, numero de tarjetas.
- 2 registros con `-1` en `years_in_current_job` y otros 2 en `years_of_education` (invalidos,
  impacto despreciable; se dejaron).

## 3. Ingenieria de variables (`features()`)

Se construyen las variables originales utiles y meta-variables derivadas **solo** de columnas del
propio dataset (el enunciado prohibe informacion externa):

| Meta-variable | Formula / idea |
|---|---|
| `salario_por_hijo` | salary / (hijos + 1) |
| `salario_por_tarjeta` | salary / (tarjetas + 1) |
| `score_x_salario` | credit_score · salary / 1e6 |
| `edu_por_edad`, `trabajo_por_edad` | años de educacion o trabajo / edad |
| `anios_sin_estudio_ni_trabajo` | edad − educacion − años en el trabajo |
| `anio_solicitud`, `mes_solicitud` | de `application_date` |
| `meses_a_expirar_tarjeta` | `credit_card_expire` − fecha de solicitud |
| `telefono_con_ext` | el telefono tiene extension (`x`) |
| `direccion_militar` | direccion APO/FPO/DPO |

**Resultado experimental:** las meta-variables no agregaron informacion; con ellas el F1 bajo
(0.694 frente a 0.706) y el Random Forest sobreajusto. Por eso el modelo final usa solo el
conjunto **NUCLEO** (las 7 variables con senal). Las meta-variables se mantienen en el codigo
porque forman parte del experimento comparativo.

## 4. Modelado y validacion (`modelo.py`)

### Modelos comparados

| Modelo | Variables | Por que se incluye |
|---|---|---|
| Regresion logistica + escalado | todas | linea base lineal |
| Random Forest (400 arboles, hoja min. 5) | todas | no lineal, robusto; sirve para ver importancias |
| LightGBM (500 arboles, 15 hojas, lr 0.02) | nucleo | boosting, estado del arte en datos tabulares |
| Logistica + `SplineTransformer` (6 nodos) | nucleo | lineal en splines: capta curvaturas sin sobreajustar |

### Validacion cruzada

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`: cada fold conserva el 29.7% de
  mora.
- `cross_val_predict` obtiene, para cada cliente, la probabilidad predicha por un modelo que **no
  lo vio** durante el entrenamiento (prediccion fuera de fold). Todas las metricas de CV se
  calculan sobre esas probabilidades.

### Umbral de decision

Un clasificador da probabilidades; por defecto se corta en 0.5. Como F1 premia el recall de la
clase minoritaria, se busca el umbral que maximiza F1 entre 0.10 y 0.89 (paso 0.01) sobre las
predicciones fuera de fold. El optimo queda en ≈ 0.35: conviene marcar como mora a clientes con
probabilidad algo menor que 0.5.

### Deteccion de sobre/sub-ajuste

Para cada modelo se reentrena con todo el train y se mide F1 en train con el mismo umbral:

- F1 train ≫ F1 CV -> sobreajuste (Random Forest: 0.859 frente a 0.693).
- F1 train ≈ F1 CV y ambos razonables -> buena generalizacion (modelo elegido: 0.706 / 0.706).
- Desviacion estandar entre folds baja (≈ 0.006) -> estable.

### Seleccion y prediccion final

Se elige automaticamente el modelo con mayor F1 de CV. Ese modelo, ya entrenado con los 36000
registros, predice probabilidades en el test, se aplica su umbral y se mapea a 0/1 con `if/else`
(`1 if p >= u else 0`), como pide el enunciado.

Chequeo de coherencia: el modelo marca como mora el 35.3% del train y el 36.0% del test, y las
medias de las variables son iguales en ambos conjuntos -> no hay cambio de distribucion.

## 5. Salidas

| Archivo | Contenido |
|---|---|
| `resultados/metricas.json` | por modelo: umbral, F1 CV ± std, F1 train, precision, recall, AUC, matriz de confusion |
| `resultados/roc_lift.png` | curvas ROC y lift acumulada de los 4 modelos |
| `resultados/matriz_confusion.png` | matriz de confusion CV del mejor modelo |
| `resultados/importancia_variables.{csv,png}` | importancia de variables (Random Forest) |
| `resultados/eda_*.{csv,png}` | resumen, tasas de mora, distribuciones, correlaciones, AUC de variables de texto |
| `<NOMBRE_ENTREGA>.csv` | `application_id,defaulted_loan` - 24000 filas + encabezado |

## 6. Limitaciones

- El F1 se estanca en ≈ 0.70-0.71 con todos los modelos probados: parece ser ruido propio de los
  datos y no una limitacion del modelo.
- El umbral se optimiza para F1; en un uso real deberia elegirse segun el costo de un falso
  negativo frente a un falso positivo.
- La conversion depende de que el `.numbers` mantenga la estructura de "CSV en la columna A".
