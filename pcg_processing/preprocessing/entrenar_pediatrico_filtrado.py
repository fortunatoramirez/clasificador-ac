# entrenar_pediatrico_filtrado.py

#
# Pregunta: si entrenamos SOLO con ciclos "bien cortados", el modelo mejora?
#  - Se entrena con ciclos filtrados (dev) y se prueba con TODOS los ciclos de los
#    mismos 117 pacientes de siempre (como llegaria audio real a produccion).
#  - Los 3 filtros estan definidos de antemano. No se ajustan viendo la prueba.

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DATASET = "dataset_circor_con_duracion.xlsx"
PACIENTES_PRUEBA = "pacientes_prueba_final.csv"
FEATURES = [f"MFCC_{i+1}" for i in range(13)] + ["RMS"]
UMBRALES = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
N_ESTIMATORS = 200
N_BOOT = 1000

CONFIGS = {
    "F0 sin filtro":                      dict(sin_ultimo=False, lo=None, hi=None),
    "F1 sin ultimo ciclo":                dict(sin_ultimo=True,  lo=None, hi=None),
    "F2 sin ultimo + 0.8-1.25x mediana":  dict(sin_ultimo=True,  lo=0.8,  hi=1.25),
}


def nuevo_modelo():
    return Pipeline([
        ("esc", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced",
                                       random_state=0, n_jobs=-1)),
    ])


def mascara(df, sin_ultimo, lo, hi):
    """True = el ciclo se usa para entrenar."""
    m = np.ones(len(df), dtype=bool)
    if sin_ultimo:
        m &= ~df["es_ultimo"].values
    if lo is not None:
        m &= df["ratio"].values >= lo
    if hi is not None:
        m &= df["ratio"].values <= hi
    return m


def votar(grupo):
    return grupo.value_counts().idxmax()


def por_paciente(prob, y, pacientes):
    """Tabla por paciente: etiqueta real y probabilidad media de sus ciclos."""
    d = pd.DataFrame({"p": pacientes, "prob": prob, "y": y})
    return d.groupby("p").agg(y=("y", "first"), score=("prob", "mean"))


def predecir_pacientes(prob, pacientes, thr):
    d = pd.DataFrame({"p": pacientes, "pred": (prob >= thr).astype(int)})
    return d.groupby("p")["pred"].apply(votar)


def sens_esp(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    sens = ((y == 1) & (pred == 1)).sum() / max((y == 1).sum(), 1)
    esp = ((y == 0) & (pred == 0)).sum() / max((y == 0).sum(), 1)
    return sens, esp


def mejor_umbral(prob, pacientes, pac_tab):
    mejor = (-9, None)
    for thr in UMBRALES:
        pred = predecir_pacientes(prob, pacientes, thr).reindex(pac_tab.index)
        s, e = sens_esp(pac_tab["y"], pred)
        if s + e - 1 > mejor[0]:
            mejor = (s + e - 1, thr)
    return mejor[1], mejor[0]


def oof_probs(X, y, grupos, m_train):
    """Validacion cruzada por paciente: entrena SOLO con ciclos filtrados,
    pero predice TODOS los ciclos del fold de validacion."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    prob = np.zeros(len(y))
    for tr, va in sgkf.split(X, y, groups=grupos):
        tr_f = tr[m_train[tr]]
        mod = nuevo_modelo()
        mod.fit(X[tr_f], y[tr_f])
        prob[va] = mod.predict_proba(X[va])[:, 1]
    return prob


def bootstrap_diff(y, score_a, pred_a, score_b, pred_b, seed=0):
    """Diferencia (B - A) en AUC y Youden J, con intervalo 95% remuestreando pacientes."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); n = len(y)
    d_auc, d_j = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        d_auc.append(roc_auc_score(yy, score_b[idx]) - roc_auc_score(yy, score_a[idx]))
        ja = sum(sens_esp(yy, pred_a[idx])) - 1
        jb = sum(sens_esp(yy, pred_b[idx])) - 1
        d_j.append(jb - ja)
    return np.percentile(d_auc, [2.5, 50, 97.5]), np.percentile(d_j, [2.5, 50, 97.5])


def main():
    df = pd.read_excel(DATASET)
    df["paciente_id"] = df["paciente_id"].astype(str)
    df["es_ultimo"] = df["es_ultimo"].astype(bool)
    df["y"] = (df["Etiqueta"] == 2).astype(int)

    # duracion relativa a la mediana de SU grabacion (sin contar el ultimo intervalo)
    med = df[~df["es_ultimo"]].groupby("archivo")["duracion_ciclo_s"].median()
    df["ratio"] = df["duracion_ciclo_s"] / df["archivo"].map(med)

    pruebas = set(pd.read_csv(PACIENTES_PRUEBA)["paciente_id"].astype(str))
    es_prueba = df["paciente_id"].isin(pruebas)
    dev = df[~es_prueba].reset_index(drop=True)
    test = df[es_prueba].reset_index(drop=True)
    print(f"dev: {len(dev)} ciclos, {dev['paciente_id'].nunique()} pacientes")
    print(f"prueba: {len(test)} ciclos, {test['paciente_id'].nunique()} pacientes (los mismos de siempre)")

    u = dev[dev["es_ultimo"]]["duracion_ciclo_s"].median()
    o = dev[~dev["es_ultimo"]]["duracion_ciclo_s"].median()
    print(f"\nDuracion mediana del ULTIMO ciclo de cada audio: {u:.2f}s  |  del resto: {o:.2f}s")

    Xd, yd, gd = dev[FEATURES].values, dev["y"].values, dev["paciente_id"].values
    Xt, yt, gt = test[FEATURES].values, test["y"].values, test["paciente_id"].values

    res = {}
    print("\n" + "=" * 100)
    print(f"{'filtro':38s} {'ciclos train':>13s} | {'dev AUC':>8s} {'umbral':>7s} | {'TEST sens':>9s} {'esp':>6s} {'J':>6s} {'AUC':>6s}")
    for nombre, cfg in CONFIGS.items():
        m_dev = mascara(dev, **cfg)

        prob_dev = oof_probs(Xd, yd, gd, m_dev)
        tab_dev = por_paciente(prob_dev, yd, gd)
        thr, j_dev = mejor_umbral(prob_dev, gd, tab_dev)
        auc_dev = roc_auc_score(tab_dev["y"], tab_dev["score"])

        mod = nuevo_modelo()
        mod.fit(Xd[m_dev], yd[m_dev])
        prob_t = mod.predict_proba(Xt)[:, 1]
        tab_t = por_paciente(prob_t, yt, gt)
        pred_t = predecir_pacientes(prob_t, gt, thr).reindex(tab_t.index)
        s, e = sens_esp(tab_t["y"], pred_t)
        auc_t = roc_auc_score(tab_t["y"], tab_t["score"])

        res[nombre] = dict(y_dev=tab_dev["y"].values, sc_dev=tab_dev["score"].values,
                           pr_dev=predecir_pacientes(prob_dev, gd, thr).reindex(tab_dev.index).values,
                           y_t=tab_t["y"].values, sc_t=tab_t["score"].values, pr_t=pred_t.values)
        print(f"{nombre:38s} {m_dev.sum():>6d} ({100*m_dev.mean():3.0f}%) | {auc_dev:8.3f} {thr:7.2f} | {s:9.3f} {e:6.3f} {s+e-1:6.3f} {auc_t:6.3f}")

    print("=" * 100)
    print("\nDiferencia contra F0 (sin filtro), con intervalo de confianza 95%.")
    print("Una mejora solo cuenta si el limite INFERIOR del intervalo es mayor que 0.\n")
    base = res["F0 sin filtro"]
    for nombre in list(CONFIGS)[1:]:
        r = res[nombre]
        for etiqueta, y, k in [(f"dev  ({len(base['y_dev'])} pac.)", base["y_dev"], "dev"),
                               (f"TEST ({len(base['y_t'])} pac.)", base["y_t"], "t")]:
            a_auc, a_j = bootstrap_diff(y, base[f"sc_{k}"], base[f"pr_{k}"], r[f"sc_{k}"], r[f"pr_{k}"])
            marca = "MEJORA" if a_auc[0] > 0 else "sin evidencia de mejora"
            print(f"{nombre:36s} {etiqueta}  dAUC {a_auc[1]:+.3f} [{a_auc[0]:+.3f}, {a_auc[2]:+.3f}]   dJ {a_j[1]:+.3f} [{a_j[0]:+.3f}, {a_j[2]:+.3f}]  -> {marca}")
        print()


if __name__ == "__main__":
    main()