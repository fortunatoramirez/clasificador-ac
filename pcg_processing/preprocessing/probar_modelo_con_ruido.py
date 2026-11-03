# probar_modelo_con_ruido.py

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, roc_auc_score

BASE_DIR = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
CARPETAS = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']
DATASET_PATH = "dataset_physionet2016.xlsx"
MODEL_PATH = r"..\..\models\modelo_pcg_soplo_adultos_rf.joblib"
METADATA_PATH = r"..\..\models\modelo_pcg_soplo_adultos_rf_metadata.json"

def voto_mayoria(grupo):
    return grupo.value_counts().idxmax()

def main():
    # --- cargar SQI de las 6 carpetas ---
    sqi_frames = []
    for carpeta in CARPETAS:
        ruta_sqi = os.path.join(BASE_DIR, carpeta, 'REFERENCE-SQI.csv')
        sqi = pd.read_csv(ruta_sqi, header=None, names=['nombre', 'label', 'sqi'])
        sqi['archivo'] = carpeta + '_' + sqi['nombre']
        sqi_frames.append(sqi[['archivo', 'sqi']])
    df_sqi = pd.concat(sqi_frames, ignore_index=True)

    # --- cargar dataset completo (sin filtrar) y quedarnos SOLO con sqi=0 ---
    df = pd.read_excel(DATASET_PATH)
    df['paciente_id'] = df['paciente_id'].astype(str)
    df = df.merge(df_sqi, on='archivo', how='left')

    df_ruidosos = df[df['sqi'] == 0].copy()
    print(f"Grabaciones ruidosas (sqi=0), nunca usadas ni para entrenar ni para calibrar: "
          f"{df_ruidosos['paciente_id'].nunique()} sujetos, {len(df_ruidosos)} filas")

    if df_ruidosos.empty:
        print("No hay grabaciones ruidosas en el dataset -- nada que evaluar.")
        return

    # --- cargar modelo y metadata ---
    modelo = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    feature_cols = metadata['feature_cols']
    umbral = metadata['umbral_decision']
    idx_soplo = list(modelo.classes_).index(2)

    # --- predecir ---
    X = df_ruidosos[feature_cols].values
    proba = modelo.predict_proba(X)[:, idx_soplo]
    df_ruidosos['prob_soplo'] = proba
    df_ruidosos['pred_ciclo'] = np.where(proba >= umbral, 2, 0)

    pred_paciente = df_ruidosos.groupby('paciente_id')['pred_ciclo'].apply(voto_mayoria)
    etiquetas_paciente = df_ruidosos.groupby('paciente_id')['Etiqueta'].first()

    m = confusion_matrix(etiquetas_paciente, pred_paciente, labels=[0, 2])
    tn, fp, fn, tp = m.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    esp = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    auc = roc_auc_score(df_ruidosos['Etiqueta'], proba)

    print(f"\n=== MODELO DE ADULTOS, solo grabaciones RUIDOSAS (sqi=0) ===")
    print(f"Sensibilidad: {sens:.3f}   Especificidad: {esp:.3f}   AUC (por ciclo): {auc:.3f}")
    print(m)
    print(f"\nPara comparar: mismo modelo, solo grabaciones LIMPIAS (examen final, sqi=1): "
          f"sens=0.968  esp=0.848  auc=0.971")

if __name__ == "__main__":
    main()