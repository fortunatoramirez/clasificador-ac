# probar_v1_con_segmentacion_v2.py

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, roc_auc_score

def voto_mayoria(grupo):
    return grupo.value_counts().idxmax()

def main():
    df_v2 = pd.read_excel('dataset_circor_v2.xlsx')
    df_v2['paciente_id'] = df_v2['paciente_id'].astype(str)

    pacientes_prueba = pd.read_csv('pacientes_prueba_final.csv')['paciente_id'].astype(str)
    df_prueba_v2 = df_v2[df_v2['paciente_id'].isin(pacientes_prueba)].copy()

    n_encontrados = df_prueba_v2['paciente_id'].nunique()
    print(f"De los 117 pacientes de prueba original, {n_encontrados} siguen presentes en dataset_circor_v2.xlsx")
    print(f"(algunos pudieron perderse si ninguna de sus grabaciones paso el filtro de calidad de v2)")

    modelo = joblib.load('../../models/modelo_pcg_soplo_rf.joblib')
    with open('../../models/modelo_pcg_soplo_rf_metadata.json') as f:
        metadata = json.load(f)
    feature_cols = metadata['feature_cols']
    umbral = metadata['umbral_decision']
    idx_soplo = list(modelo.classes_).index(2)

    X = df_prueba_v2[feature_cols].values
    proba = modelo.predict_proba(X)[:, idx_soplo]
    df_prueba_v2['prob_soplo'] = proba
    df_prueba_v2['pred_ciclo'] = np.where(proba >= umbral, 2, 0)

    pred_paciente = df_prueba_v2.groupby('paciente_id')['pred_ciclo'].apply(voto_mayoria)
    etiquetas_paciente = df_prueba_v2.groupby('paciente_id')['Etiqueta'].first()

    m = confusion_matrix(etiquetas_paciente, pred_paciente, labels=[0, 2])
    tn, fp, fn, tp = m.ravel()
    sens, esp = tp/(tp+fn), tn/(tn+fp)
    auc = roc_auc_score(df_prueba_v2['Etiqueta'], proba)

    print(f"\n=== MODELO V1 + SEGMENTACION V2, mismos {n_encontrados} pacientes de prueba ===")
    print(f"Sensibilidad: {sens:.3f}   Especificidad: {esp:.3f}   AUC (por ciclo): {auc:.3f}")
    print(m)
    print(f"\nPara comparar: modelo v1 + segmentacion v1 (produccion actual): sens=0.735  esp=0.699  auc=0.706")

if __name__ == "__main__":
    main()