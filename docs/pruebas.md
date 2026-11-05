# Pruebas

## Pruebas automaticas - `test_pipeline.py`

Script sin frameworks (asserts de Python). Se ejecuta despues de `modelo.py`:

```bash
python -W ignore test_pipeline.py
```

Salida esperada:

```
OK test_csv_convertidos
OK test_reparseo_direccion_multilinea
OK test_features
OK test_mejor_umbral
OK test_archivo_entrega
5 pruebas pasaron
```

Si una prueba falla, el script se detiene con `AssertionError` y el detalle.

| Prueba | Que verifica | Por que importa |
|---|---|---|
| `test_csv_convertidos` | train 36000x23, test 24000x22, sin `defaulted_loan` en test, sin nulos, `application_id` unico | detecta una conversion fallida (p. ej. registros partidos en 2 filas) |
| `test_reparseo_direccion_multilinea` | un registro con salto de linea dentro de comillas se lee como 1 fila y conserva el `\n` | es la tecnica central de `convertir.py` |
| `test_features` | `features()` devuelve todas las filas, incluye `NUCLEO` y no produce NaN ni infinitos | evita errores silenciosos en los modelos (divisiones por cero, fechas mal parseadas) |
| `test_mejor_umbral` | con datos separables encuentra F1 = 1 y un umbral entre las dos clases | valida la busqueda del umbral |
| `test_archivo_entrega` | columnas exactas `application_id,defaulted_loan`, 24000 filas, valores {0,1}, mismos ids y en el mismo orden que el test | un formato incorrecto se califica con 0 |

## Validaciones internas en `modelo.py`

- `assert entrega.shape == (24000, 2) and set(entrega.defaulted_loan) <= {0, 1}` antes de
  escribir el CSV: si falla, no se genera un archivo invalido.
- Al final se imprime la tasa de mora predicha en el test junto a la del train como chequeo
  rapido de coherencia.

## Validacion del modelo (pruebas experimentales)

No hay etiquetas del test, asi que la calidad del modelo se estima con validacion cruzada:

| Chequeo | Criterio | Resultado del modelo elegido |
|---|---|---|
| F1 fuera de fold | lo mas alto posible | 0.706 |
| Estabilidad entre folds | desviacion estandar baja | ± 0.006 |
| Sobreajuste | F1 train ≈ F1 CV | 0.706 frente a 0.706 |
| Subajuste | supera claramente a los modelos lineales base | 0.706 frente a 0.694 |
| AUC | > 0.5 (azar) | 0.883 |
| Cambio de distribucion train/test | medias iguales; tasa predicha similar | 35.3% (train) frente a 36.0% (test) |

## Verificacion manual de la entrega

```bash
head -3 "<NOMBRE_ENTREGA>.csv"     # application_id,defaulted_loan / ID,0|1
wc -l "<NOMBRE_ENTREGA>.csv"       # 24001
```

Abrirlo en un editor de texto (no en Excel) para confirmar que no tiene columna de indice,
comillas extra ni `True`/`False`.

## Reproducibilidad

Ejecutar `python modelo.py` dos veces debe dar exactamente las mismas metricas y el mismo CSV
(semilla fija `SEED = 42`). Para comprobarlo en PowerShell:

```powershell
Get-FileHash "<NOMBRE_ENTREGA>.csv"   # comparar el hash entre ejecuciones
```
