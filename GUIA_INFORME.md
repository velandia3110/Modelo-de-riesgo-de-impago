# Guia para el informe (estructura + datos)

> Esto es solo el esqueleto y los hallazgos en bruto. La redaccion, la interpretacion y las
> conclusiones van por tu cuenta. Todos los numeros salen de `eda.py` y `modelo.py`
> (carpeta `resultados/`).

---

## 0. Portada y resumen
- Titulo, autor, curso, profesor, fecha.
- Resumen (≈150-250 palabras): problema, datos, metodo, F1 obtenido, recomendacion principal.
- Palabras clave.

## 1. Introduccion
- Contexto: riesgo crediticio, costo de la mora para la entidad.
- Objetivo general / especificos (predecir `defaulted_loan`, identificar factores de riesgo,
  proponer acciones).
- Por que F1 y no accuracy: clases desbalanceadas (29.7% mora). Un modelo que diga "siempre paga"
  tiene 70% de accuracy y F1 = 0.

## 2. Materiales y metodos
### 2.1 Datos
- Train: 36000 x 23 (22 predictoras + objetivo). Test: 24000 x 22.
- Tabla de variables con tipo (numerica, booleana, texto, fecha, identificador).
- Nota de preparacion: los archivos venian en formato Apple Numbers con cada registro partido en
  2 filas (salto de linea dentro de `address`); se reconstruyeron con `convertir.py`.
### 2.2 Herramientas
- Python 3.13, pandas, scikit-learn 1.9, LightGBM, matplotlib.
### 2.3 Metodologia (diagrama sugerido: datos -> limpieza -> EDA -> features -> modelos -> CV -> umbral -> prediccion)
- Validacion cruzada estratificada 5-fold (`StratifiedKFold`, shuffle, semilla 42).
  Investigar y explicar: por que estratificada, por que 5 folds, que es `cross_val_predict`.
- Ajuste del umbral de decision para maximizar F1 (0.10 a 0.90 en pasos de 0.01) sobre las
  predicciones fuera de fold. Explicar por que el umbral optimo (0.35) es menor que 0.5.
- Deteccion de sobre/sub-ajuste: comparar F1 en train vs F1 en CV.
- Definir precision, recall, F1, ROC/AUC, curva lift, matriz de confusion (ecuaciones del enunciado).

## 3. Analisis exploratorio (figuras en `resultados/eda_*`)
### 3.1 Calidad de datos
- 0 valores nulos.
- `years_in_current_job` y `years_of_education`: 2 registros con valor -1 (dato invalido).
  Se dejaron (impacto despreciable); discutir alternativa (imputar).
- Identificadores unicos sin valor predictivo: `application_id`, `phone_number`,
  `credit_card_number`, `address` (36000 valores distintos cada uno).
### 3.2 Variable objetivo
- 10695 mora (29.7%) vs 25305 paga (70.3%).
### 3.3 Variables con senal (`eda_tasa_mora.png`, `eda_distribuciones.png`)
| Variable | Correlacion con mora | Tasa de mora (decil bajo -> alto) |
|---|---|---|
| salary | -0.359 | 0.55 -> 0.04 |
| credit_score | -0.332 | 0.51 -> 0.04 |
| number_of_children | +0.305 | 2 hijos 0.18 · 3 hijos 0.31 · 4 hijos 0.68 · 5+ hijos ~0.99 |
| years_of_education | -0.259 | 0.47 -> 0.09 |
| years_in_current_job | -0.218 | 0.46 -> 0.14 |
| age | -0.144 | 0.39 -> 0.18 |
| has_criminal_records | — | sin registro 0.29 · con registro 0.43 (3.6% de clientes) |
- Relaciones monotonas; observa la no linealidad en `number_of_children` (salto en 4+).
### 3.4 Variables sin senal (`eda_auc_variables_texto.csv`, AUC ≈ 0.50 = azar)
- estado de la direccion, current_job, company, email_domain, credit_card_provider, formato de
  placa, hora y dia de solicitud, apellido, is_married, smoker, number_of_credit_cards.
- Discutir: por que es bueno descartarlas (ruido, sobreajuste, posible sesgo/discriminacion
  por nombre o ubicacion).
### 3.5 Correlacion entre predictoras (`eda_correlaciones.png`)
- Comentar si hay multicolinealidad.

## 4. Ingenieria de variables (meta-variables)
- Creadas: salario_por_hijo, salario_por_tarjeta, score_x_salario, edu_por_edad,
  trabajo_por_edad, anios_sin_estudio_ni_trabajo, anio/mes de solicitud, meses hasta expirar la
  tarjeta, telefono con extension, direccion militar.
- Resultado: con todas, el F1 CV fue 0.694 (logistica) / 0.693 (RF); con solo las 7 variables
  nucleo subio a 0.701-0.706. Las meta-variables no aportaron informacion nueva y el RF con
  todas sobreajusto (ver tabla 5). Ver `importancia_variables.png`.
- Conjunto final ("nucleo"): age, salary, years_in_current_job, credit_score,
  number_of_children, years_of_education, has_criminal_records.

## 5. Resultados (`resultados/metricas.json`)
| Modelo | Variables | F1 CV (±std) | F1 train | AUC | Umbral |
|---|---|---|---|---|---|
| Regresion logistica | todas | 0.694 ± 0.006 | 0.696 | 0.875 | 0.34 |
| Random Forest | todas | 0.693 ± 0.005 | **0.859** | 0.875 | 0.35 |
| LightGBM | nucleo | 0.701 ± 0.004 | 0.714 | 0.882 | 0.38 |
| **Logistica + splines** | nucleo | **0.706 ± 0.006** | 0.706 | **0.883** | 0.35 |

- Otros experimentos (para la discusion): interacciones polinomicas grado 2 (0.705), hijos como
  categoria (0.704), LightGBM con restricciones monotonas (0.702), C de regularizacion 0.1/1/10
  (0.704/0.706/0.705). Todos tocan el mismo techo de ≈0.70-0.71.
- Modelo elegido: logistica + splines (6 nodos). Matriz de confusion CV (`matriz_confusion.png`):

  |  | Pred. paga | Pred. mora |
  |---|---|---|
  | Real paga | 20854 (TN) | 4451 (FP) |
  | Real mora | 2437 (FN) | 8258 (TP) |

  Precision 0.650 · Recall 0.772 · F1 0.706.
- ROC (`roc_lift.png`): todas las curvas casi superpuestas, AUC 0.875-0.883.
- Lift (modelo elegido):
  | % clientes mas riesgosos | Tasa de mora | Lift | % de morosos capturados |
  |---|---|---|---|
  | 10% | 88.8% | 2.99 | 29.9% |
  | 20% | 78.5% | 2.64 | 52.8% |
  | 30% | 69.4% | 2.34 | 70.1% |
- Prediccion en test: 36.0% marcados como mora (en train, con el mismo umbral, 35.3%). Las
  medias de las variables en train y test son practicamente iguales -> no hay drift.

## 6. Discusion
- Sobreajuste: RF con todas (train 0.859 vs CV 0.693). Ajuste correcto: el modelo elegido
  (train 0.706 ≈ CV 0.706, std baja entre folds).
- Por que un modelo simple gana: relaciones monotonas y casi aditivas; los splines capturan la
  curvatura sin memorizar ruido.
- Techo de ≈0.71: discutir ruido irreducible / informacion que no esta en los datos.
- Trade-off precision/recall y como el umbral cambia segun el costo de negocio
  (FN = credito perdido, FP = cliente bueno rechazado).
- Limitaciones.

## 7. Soporte a la toma de decisiones (seccion prioritaria segun el enunciado)
Ideas a desarrollar con tus palabras:
- Segmentacion por riesgo usando la probabilidad del modelo (p. ej. bajo / medio / alto con la
  tabla de lift): el 20% mas riesgoso concentra el 53% de la mora.
- Reglas operativas: atencion especial a 4+ hijos con salario bajo, credit_score bajo,
  registros criminales.
- Ajustar el umbral segun la politica (mas conservador -> mas recall; mas comercial -> mas precision).
- Acciones: garantias o codeudor en riesgo alto, montos y plazos menores, seguimiento temprano.
- Etica: no usar nombre, ubicacion ni estado civil (ademas no predicen).
- Monitoreo del modelo en produccion (recalcular F1 y drift periodicamente).

## 8. Conclusiones
## 9. Referencias
- scikit-learn docs (F1, StratifiedKFold, SplineTransformer), LightGBM docs, los enlaces de
  Kaggle/GitHub del enunciado, bibliografia de riesgo crediticio.
## Anexos
- Codigo: `convertir.py`, `eda.py`, `modelo.py`.
