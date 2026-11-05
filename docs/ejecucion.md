# Ejecucion

## Requisitos

- Python 3.11 o superior (probado con 3.13.14 en Windows 11).
- ~3 GB de RAM libres para `convertir.py` (leer `.numbers` es pesado). El resto usa < 1 GB.
- Dependencias (versiones exactas en [`requirements.txt`](../requirements.txt)):

| Paquete | Uso |
|---|---|
| pandas | carga y manipulacion de datos |
| numpy | calculo numerico |
| scikit-learn | modelos, validacion cruzada, metricas |
| lightgbm | modelo de gradient boosting (comparacion) |
| matplotlib | graficas |
| numbers-parser | leer archivos Apple Numbers |

```bash
python -m pip install -r requirements.txt
```

Todos los comandos se ejecutan **desde la carpeta raiz del proyecto** (los scripts usan rutas
relativas).

## Orden de ejecucion

| Paso | Comando | Tiempo aprox. | Genera |
|---|---|---|---|
| 1 | `python convertir.py` | ~15 min | `train_dataset.csv`, `test_dataset.csv` |
| 2 | `python eda.py` | ~20 s | `resultados/eda_*` |
| 3 | `python modelo.py` | ~40 s | `resultados/metricas.json`, graficas, CSV de entrega |
| 4 | `python test_pipeline.py` | ~5 s | salida en consola |

- El paso 1 solo hace falta si no existen los `.csv` convertidos.
- Los pasos 2 y 3 son independientes entre si; ambos necesitan el paso 1.
- El paso 4 necesita el paso 3 (verifica el CSV de entrega).
- Para ocultar los avisos de scikit-learn/LightGBM: `python -W ignore modelo.py`.

### Variante de conversion `--crudo`

`python convertir.py --crudo` repara un CSV **intermedio** (el que deja numbers-parser con todo el
registro en la primera columna) sin volver a leer el `.numbers`. Tarda segundos. Si los CSV ya
estan bien convertidos, el script se detiene con un mensaje y no los toca.

## Salida esperada de `modelo.py`

```
Regresion logistica (todas)  F1cv=0.6941±0.0062 F1train=0.6959 AUC=0.8748 umbral=0.34
Random Forest (todas)        F1cv=0.6934±0.0049 F1train=0.8590 AUC=0.8747 umbral=0.35
LightGBM (nucleo)            F1cv=0.7014±0.0039 F1train=0.7141 AUC=0.8815 umbral=0.38
Logistica + splines (nucleo) F1cv=0.7057±0.0065 F1train=0.7056 AUC=0.8835 umbral=0.35
Mejor: Logistica + splines (nucleo) | tasa predicha en test=0.360 (train=0.297) -> ...
```

Los resultados son reproducibles: todas las fuentes de aleatoriedad usan `SEED = 42`.

## Antes de entregar

1. Cambiar `NOMBRE_ENTREGA` en [`modelo.py`](../modelo.py) por el nombre real
   (formato del enunciado: `Nombre Apellido dd-mm-aaaa ...csv`) y volver a ejecutar
   `python modelo.py`, o renombrar el archivo y actualizar la constante (la prueba la usa).
2. Ejecutar `python test_pipeline.py` y confirmar `5 pruebas pasaron`.

## Solucion de problemas

| Sintoma | Causa | Solucion |
|---|---|---|
| `pd.read_csv` da basura o error de codificacion | el `.csv` es en realidad un `.numbers` renombrado (empieza con `PK`) | ejecutar `python convertir.py` |
| CSV con 72000 / 48000 filas y solo la 1a columna llena | conversion intermedia, registros partidos en 2 filas | `python convertir.py --crudo` |
| `ModuleNotFoundError` | falta una dependencia | `python -m pip install -r requirements.txt` |
| `convertir.py` parece congelado | numbers-parser es lento con archivos de 30 MB | esperar (~15 min); no lanzar otra copia |
| `test_archivo_entrega` falla con "ejecuta primero" | no existe el CSV con el nombre de `NOMBRE_ENTREGA` | ejecutar `python modelo.py` |
| `FileNotFoundError` en cualquier script | se ejecuto desde otra carpeta | ejecutar desde la raiz del proyecto |
