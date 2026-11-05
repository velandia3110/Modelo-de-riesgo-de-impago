"""Pruebas rapidas del pipeline (sin frameworks). Uso: python test_pipeline.py
Requiere train_dataset.csv / test_dataset.csv ya convertidos. Tarda unos segundos.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd

import modelo


def test_csv_convertidos():
    tr, te = pd.read_csv("train_dataset.csv"), pd.read_csv("test_dataset.csv")
    assert tr.shape == (36000, 23), tr.shape
    assert te.shape == (24000, 22), te.shape
    assert "defaulted_loan" not in te.columns
    assert tr.isna().sum().sum() == 0 and te.isna().sum().sum() == 0
    assert tr.application_id.is_unique and te.application_id.is_unique


def test_reparseo_direccion_multilinea():
    # misma tecnica de convertir.py: unir lineas y re-parsear respeta saltos entre comillas
    lineas = ['a,address,b', '1,"Calle 1', 'Ciudad, XX 00000",2']
    df = pd.read_csv(io.StringIO("\n".join(lineas)))
    assert df.shape == (1, 3) and df.address[0] == "Calle 1\nCiudad, XX 00000"


def test_features():
    df = pd.read_csv("train_dataset.csv", nrows=500)
    X = modelo.features(df)
    assert len(X) == 500
    assert set(modelo.NUCLEO) <= set(X.columns)
    assert np.isfinite(X.to_numpy(dtype=float)).all(), "features con NaN/inf"


def test_mejor_umbral():
    y = np.array([0, 0, 1, 1])
    u, f1 = modelo.mejor_umbral(y, np.array([0.1, 0.2, 0.7, 0.8]))
    assert f1 == 1.0 and 0.2 < u <= 0.7


def test_archivo_entrega():
    entrega = Path(modelo.NOMBRE_ENTREGA)
    assert entrega.exists(), "ejecuta primero python modelo.py"
    df = pd.read_csv(entrega)
    assert list(df.columns) == ["application_id", "defaulted_loan"]
    assert df.shape == (24000, 2)
    assert set(df.defaulted_loan.unique()) <= {0, 1}
    ids_test = pd.read_csv("test_dataset.csv", usecols=["application_id"]).application_id
    assert (df.application_id.values == ids_test.values).all(), "orden/ids distintos al test"


if __name__ == "__main__":
    pruebas = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in pruebas:
        t()
        print("OK", t.__name__)
    print(f"{len(pruebas)} pruebas pasaron")
