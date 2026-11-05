"""Analisis exploratorio: tablas y graficas para el informe -> resultados/eda_*"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from modelo import NUCLEO, SALIDA

d = pd.read_csv("train_dataset.csv")
y = d.defaulted_loan.astype(int)

# 1. Resumen general
d.describe(include="all").T.to_csv(SALIDA / "eda_resumen.csv")
print("filas", d.shape, "nulos", int(d.isna().sum().sum()), "tasa mora", round(y.mean(), 4))
print("valores negativos:", {c: int((d[c] < 0).sum()) for c in ["years_in_current_job", "years_of_education"]})

# 2. Tasa de mora por cuantiles de cada variable numerica
num = ["age", "salary", "years_in_current_job", "credit_score", "years_of_education"]
fig, axs = plt.subplots(2, 4, figsize=(16, 7))
for ax, c in zip(axs.flat, num + ["number_of_children", "has_criminal_records", "number_of_credit_cards"]):
    g = d[c] if d[c].nunique() < 10 else pd.qcut(d[c], 10, duplicates="drop")
    d.groupby(g, observed=True).defaulted_loan.mean().plot.bar(ax=ax)
    ax.set_title(f"Tasa de mora vs {c}"); ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=6)
fig.tight_layout(); fig.savefig(SALIDA / "eda_tasa_mora.png", dpi=130)

# 3. Distribuciones por clase
fig, axs = plt.subplots(1, 5, figsize=(18, 3.5))
for ax, c in zip(axs, num):
    for v, lab in [(False, "Paga"), (True, "Mora")]:
        d.loc[d.defaulted_loan == v, c].plot.hist(ax=ax, bins=30, alpha=0.5, density=True, label=lab)
    ax.set_title(c); ax.legend()
fig.tight_layout(); fig.savefig(SALIDA / "eda_distribuciones.png", dpi=130)

# 4. Correlaciones
corr = d[NUCLEO].astype(float).assign(defaulted_loan=y).corr()
corr.round(3).to_csv(SALIDA / "eda_correlaciones.csv")
plt.figure(figsize=(8, 6)); plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90); plt.yticks(range(len(corr)), corr.columns)
plt.title("Matriz de correlacion"); plt.tight_layout(); plt.savefig(SALIDA / "eda_correlaciones.png", dpi=130)

# 5. Poder predictivo de variables de texto / alta cardinalidad (AUC con target encoding, CV 5)
fecha = pd.to_datetime(d.application_date)
texto = {
    "estado (address)": d.address.str.extract(r"([A-Z]{2}) \d{5}")[0].fillna("NA"),
    "current_job": d.current_job, "company": d.company, "email_domain": d.email_domain,
    "credit_card_provider": d.credit_card_provider, "formato placa":
        d.car_licence_plate.str.replace(r"[A-Z]", "A", regex=True).str.replace(r"\d", "9", regex=True),
    "hora solicitud": fecha.dt.hour.astype(str), "dia semana": fecha.dt.dayofweek.astype(str),
    "apellido": d.name.str.split().str[-1], "is_married": d.is_married.astype(str),
    "smoker": d.smoker.astype(str),
}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
auc = {k: cross_val_score(make_pipeline(TargetEncoder(random_state=0), LogisticRegression()),
                          v.to_frame(), y, cv=cv, scoring="roc_auc").mean() for k, v in texto.items()}
pd.Series(auc, name="auc_cv").round(4).to_csv(SALIDA / "eda_auc_variables_texto.csv")
print(pd.Series(auc).round(4))
