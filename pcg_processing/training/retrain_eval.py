# retrain_eval.py
# Recibe: ruta_dataset_xlsx  ruta_salida_modelo
# Entrena con el dataset, evalúa con cross-val, guarda modelo candidato
# Devuelve JSON con métricas completas

import sys
import os
import json
import warnings
import contextlib
import io
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Uso: retrain_eval.py <dataset.xlsx> <modelo_candidato>"}))
        sys.exit(1)

    dataset_path  = sys.argv[1]
    model_out     = sys.argv[2]   # sin .pkl

    try:
        # ── Cargar dataset ──
        df = pd.read_excel(dataset_path)

        required = [f"MFCC_{i+1}" for i in range(13)] + ["RMS", "Etiqueta"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise Exception(f"Columnas faltantes en el dataset: {missing}")

        feature_cols = [f"MFCC_{i+1}" for i in range(13)] + ["RMS"]
        X = df[feature_cols].values
        y = df["Etiqueta"].values

        n_total  = len(X)
        class_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}

        # ── Entrenamiento + evaluación con cross-validation ──
        clf = DecisionTreeClassifier(random_state=42)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X, y, cv=cv)

        acc  = float(accuracy_score(y, y_pred))
        prec = float(precision_score(y, y_pred, average='weighted', zero_division=0))
        rec  = float(recall_score(y, y_pred, average='weighted', zero_division=0))
        f1   = float(f1_score(y, y_pred, average='weighted', zero_division=0))
        cm   = confusion_matrix(y, y_pred, labels=[0, 1, 2]).tolist()

        # Por clase
        prec_cls = precision_score(y, y_pred, average=None, labels=[0,1,2], zero_division=0).tolist()
        rec_cls  = recall_score(y, y_pred, average=None, labels=[0,1,2], zero_division=0).tolist()
        f1_cls   = f1_score(y, y_pred, average=None, labels=[0,1,2], zero_division=0).tolist()

        # ── Entrenar modelo final con TODO el dataset ──
        clf_final = DecisionTreeClassifier(random_state=42)
        clf_final.fit(X, y)

        # Guardar con PyCaret para que classify.py lo pueda cargar
        try:
            with suppress_stdout():
                from pycaret.classification import setup, create_model, save_model
                df_train = df[feature_cols + ["Etiqueta"]].copy()
                setup(data=df_train, target="Etiqueta",
                      session_id=42, verbose=False, html=False)
                model_pc = create_model("dt", verbose=False)
                save_model(model_pc, model_out)
            saved_with = "pycaret"
        except Exception:
            # Fallback: joblib (classify.py necesitaría adaptarse si esto ocurre)
            import joblib
            joblib.dump(clf_final, model_out + ".pkl")
            saved_with = "joblib"

        result = {
            "status":       "success",
            "saved_with":   saved_with,
            "n_total":      n_total,
            "class_counts": class_counts,
            "metrics": {
                "accuracy":  round(acc,  4),
                "precision": round(prec, 4),
                "recall":    round(rec,  4),
                "f1":        round(f1,   4)
            },
            "per_class": {
                "Sano":  {"precision": round(prec_cls[0],4), "recall": round(rec_cls[0],4), "f1": round(f1_cls[0],4)},
                "Click": {"precision": round(prec_cls[1],4), "recall": round(rec_cls[1],4), "f1": round(f1_cls[1],4)},
                "Soplo": {"precision": round(prec_cls[2],4), "recall": round(rec_cls[2],4), "f1": round(f1_cls[2],4)}
            },
            "confusion_matrix": cm   # 3x3 [Sano, Click, Soplo]
        }

    except Exception as e:
        result = {"error": str(e)}

    print(json.dumps(result))


if __name__ == "__main__":
    main()