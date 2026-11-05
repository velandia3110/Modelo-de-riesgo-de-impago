"""Taller 2 IA - prediccion de defaulted_loan.
Uso: python modelo.py   (requiere train_dataset.csv / test_dataset.csv; ver convertir.py)
Genera: resultados/ (metricas, graficas) y el CSV de entrega.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

SEED = 42
NUCLEO = ["age", "salary", "years_in_current_job", "credit_score", "number_of_children",
          "years_of_education", "has_criminal_records"]
SALIDA = Path("resultados")
SALIDA.mkdir(exist_ok=True)
NOMBRE_ENTREGA = "Nombre Apellido 05-11-2026 Proyecto- 1-machine learning.csv"  # TODO: poner nombre real


def features(df):
    """Variables originales utiles + meta-variables. Solo usa columnas del propio dataset."""
    X = pd.DataFrame(index=df.index)
    for c in ["age", "salary", "number_of_credit_cards", "years_in_current_job", "credit_score",
              "number_of_children", "years_of_education"]:
        X[c] = df[c]
    for c in ["has_criminal_records", "is_married", "smoker"]:
        X[c] = df[c].astype(int)
    # meta-variables
    X["salario_por_hijo"] = df.salary / (df.number_of_children + 1)
    X["salario_por_tarjeta"] = df.salary / (df.number_of_credit_cards + 1)
    X["score_x_salario"] = df.credit_score * df.salary / 1e6
    X["edu_por_edad"] = df.years_of_education / df.age
    X["trabajo_por_edad"] = df.years_in_current_job / df.age
    X["anios_sin_estudio_ni_trabajo"] = df.age - df.years_of_education - df.years_in_current_job
    fecha = pd.to_datetime(df.application_date)
    X["anio_solicitud"] = fecha.dt.year
    X["mes_solicitud"] = fecha.dt.month
    exp = pd.to_datetime(df.credit_card_expire, format="%m/%y")
    X["meses_a_expirar_tarjeta"] = (exp.dt.year - fecha.dt.year) * 12 + exp.dt.month - fecha.dt.month
    X["telefono_con_ext"] = df.phone_number.str.contains("x").astype(int)
    X["direccion_militar"] = df.address.str.contains(r"\b(?:APO|FPO|DPO)\b").astype(int)
    return X


def mejor_umbral(y, p):
    umbrales = np.arange(0.1, 0.9, 0.01)
    f1s = [f1_score(y, p >= u) for u in umbrales]
    i = int(np.argmax(f1s))
    return float(umbrales[i]), float(f1s[i])


def curva_lift(y, p, ax, nombre):
    orden = np.argsort(-p)
    y_ord = np.asarray(y)[orden]
    frac = np.arange(1, len(y_ord) + 1) / len(y_ord)
    lift = np.cumsum(y_ord) / np.arange(1, len(y_ord) + 1) / y_ord.mean()
    ax.plot(frac, lift, label=nombre)


def main():
    train = pd.read_csv("train_dataset.csv")
    test = pd.read_csv("test_dataset.csv")
    # mapeo [False, True] -> [0, 1] con if/else, como pide el enunciado
    y = train.defaulted_loan.map(lambda v: 1 if v in (True, "True") else 0).values
    X, X_test = features(train), features(test)

    # (modelo, columnas). "Todas" = originales + meta-variables; "nucleo" = las 7 con senal en el EDA
    todas = list(X.columns)
    modelos = {
        "Regresion logistica (todas)": (make_pipeline(StandardScaler(),
                                                      LogisticRegression(max_iter=2000)), todas),
        "Random Forest (todas)": (RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                                         n_jobs=-1, random_state=SEED), todas),
        "LightGBM (nucleo)": (LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=15,
                                             min_child_samples=80, subsample=0.8, subsample_freq=1,
                                             colsample_bytree=0.8, random_state=SEED, verbose=-1),
                              NUCLEO),
        "Logistica + splines (nucleo)": (make_pipeline(SplineTransformer(n_knots=6),
                                                       LogisticRegression(C=1, max_iter=3000)),
                                         NUCLEO),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    resultados, probas = {}, {}
    for nombre, (m, cols) in modelos.items():
        p = cross_val_predict(m, X[cols], y, cv=cv, method="predict_proba")[:, 1]
        u, f1 = mejor_umbral(y, p)
        pred = p >= u
        # F1 por fold con el umbral elegido (estabilidad)
        f1_folds = [f1_score(y[te], pred[te]) for _, te in cv.split(X, y)]
        # F1 en entrenamiento (para detectar sobreajuste: train vs CV)
        m.fit(X[cols], y)
        f1_train = f1_score(y, m.predict_proba(X[cols])[:, 1] >= u)
        resultados[nombre] = dict(umbral=u, f1_cv=f1, f1_cv_std=float(np.std(f1_folds)),
                                  f1_train=float(f1_train),
                                  precision=float(precision_score(y, pred)),
                                  recall=float(recall_score(y, pred)),
                                  auc=float(roc_auc_score(y, p)),
                                  matriz_confusion=confusion_matrix(y, pred).tolist())
        probas[nombre] = p
        print(f"{nombre:22s} F1cv={f1:.4f}±{np.std(f1_folds):.4f} F1train={f1_train:.4f} "
              f"AUC={resultados[nombre]['auc']:.4f} umbral={u:.2f}")

    (SALIDA / "metricas.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))

    # Graficas: ROC, lift, matriz de confusion del mejor
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for nombre, p in probas.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{nombre} (AUC={resultados[nombre]['auc']:.3f})")
        curva_lift(y, p, ax[1], nombre)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(title="Curva ROC (validacion cruzada 5-fold)", xlabel="FPR", ylabel="TPR")
    ax[1].axhline(1, color="k", ls="--", lw=1)
    ax[1].set(title="Curva lift acumulada", xlabel="Fraccion de clientes (ordenados por riesgo)",
              ylabel="Lift")
    ax[0].legend(); ax[1].legend()
    fig.tight_layout(); fig.savefig(SALIDA / "roc_lift.png", dpi=130)

    mejor = max(resultados, key=lambda k: resultados[k]["f1_cv"])
    cm = np.array(resultados[mejor]["matriz_confusion"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Paga (0)", "Mora (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Matriz de confusion CV - {mejor}")
    plt.savefig(SALIDA / "matriz_confusion.png", dpi=130, bbox_inches="tight")

    # Importancia de variables (Random Forest con todas: muestra que las meta-variables aportan poco)
    rf = modelos["Random Forest (todas)"][0]
    imp = pd.Series(rf.feature_importances_, index=todas).sort_values()
    imp.to_csv(SALIDA / "importancia_variables.csv")
    plt.figure(figsize=(7, 7)); imp.plot.barh(); plt.title("Importancia - Random Forest")
    plt.tight_layout(); plt.savefig(SALIDA / "importancia_variables.png", dpi=130)

    # Prediccion final con el mejor modelo (ya entrenado en todo el train)
    u = resultados[mejor]["umbral"]
    m, cols = modelos[mejor]
    p_test = m.predict_proba(X_test[cols])[:, 1]
    entrega = pd.DataFrame({"application_id": test.application_id,
                            "defaulted_loan": [1 if v >= u else 0 for v in p_test]})
    assert entrega.shape == (24000, 2) and set(entrega.defaulted_loan) <= {0, 1}
    entrega.to_csv(NOMBRE_ENTREGA, index=False)
    print(f"Mejor: {mejor} | tasa predicha en test={entrega.defaulted_loan.mean():.3f} "
          f"(train={y.mean():.3f}) -> {NOMBRE_ENTREGA}")


if __name__ == "__main__":
    main()
