# Convierte los .numbers a .csv.
# Dentro del Numbers cada fila es una linea de CSV crudo en la columna A, y la direccion
# trae un salto de linea entre comillas, asi que un registro ocupa 2 filas.
# Se unen todas las lineas y se re-parsean como CSV (respeta comillas).
import io, sys
import pandas as pd
from numbers_parser import Document

def lineas_desde_numbers(nombre):
    tabla = Document(f"{nombre}.numbers").sheets[0].tables[0]
    filas = tabla.rows(values_only=True)
    return [",".join(filas[0])] + [str(f[0]) for f in filas[1:]]

def lineas_desde_csv_crudo(nombre):
    # el CSV intermedio (header bien separado, datos en la 1a columna)
    d = pd.read_csv(f"{nombre}.csv", dtype=str)
    if d.iloc[:, 1:].notna().any().any():
        sys.exit(f"{nombre}.csv ya esta convertido; --crudo solo aplica al CSV intermedio")
    return [",".join(d.columns)] + d.iloc[:, 0].tolist()

desde_crudo = "--crudo" in sys.argv
for nombre in ["train_dataset", "test_dataset"]:
    lineas = lineas_desde_csv_crudo(nombre) if desde_crudo else lineas_desde_numbers(nombre)
    df = pd.read_csv(io.StringIO("\n".join(lineas)))
    df.to_csv(f"{nombre}.csv", index=False)
    print(nombre, df.shape)
