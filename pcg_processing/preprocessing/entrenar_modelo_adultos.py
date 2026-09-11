# entrenar_modelo_adultos.py
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

BASE_DIR = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
DATASET_PATH = "dataset_physionet2016.xlsx"
CARPETAS = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']
FEATURE_COLS = [f"MFCC_{i+1}" for i in range(13)] + ["RMS"]

def voto_mayoria(grupo):
    return grupo.value_counts().idxmax()

def cargar_y_filtrar():
    df = pd.read_excel(DATASET_PATH)
    df['paciente_id'] = df['paciente_id'].astype(str)
    sqi_frames = []
    for carpeta in CARPETAS:
        ruta_sqi = os.path.join(BASE_DIR, carpeta, 'REFERENCE-SQI.csv')
        sqi = pd.read_csv(ruta_sqi, header=None, names=['nombre', 'label', 'sqi'])
        sqi['archivo'] = carpeta + '_' + sqi['nombre']
        sqi_frames.append(sqi[['archivo', 'sqi']])
    df_sqi = pd.concat(sqi_frames, ignore_index=True)
    df = df.merge(df_sqi, on='archivo', how='left')
    n_antes = df['paciente_id'].nunique()
    df = df[df['sqi'] == 1]
    print(f"Filtro de calidad (SQI): {df['paciente_id'].nunique()} de {n_antes} sujetos")
    return df

def main():
    df = cargar_y_filtrar()
    X = df[FEATURE_COLS].values
    y = df['Etiqueta'].values
    grupos = df['paciente_id'].values

    sgkf_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    idx_dev, idx_prueba = next(sgkf_split.split(X, y, groups=grupos))
    X_dev, y_dev, g_dev = X[idx_dev], y[idx_dev], grupos[idx_dev]
    X_prueba, y_prueba = X[idx_prueba], y[idx_prueba]

    print(f"repetidos entre dev/prueba: {len(set(g_dev) & set(grupos[idx_prueba]))}")
    print(f"dev: {len(idx_dev)} filas, {len(set(g_dev))} sujetos")
    print(f"prueba: {len(idx_prueba)} filas, {len(set(grupos[idx_prueba]))} sujetos")
    pd.Series(sorted(set(grupos[idx_prueba]))).to_csv('sujetos_prueba_final_adultos_v2.csv', index=False, header=['paciente_id'])

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    modelos = {
        'Regresion Logistica': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=0),
        'Gradient Boosting': GradientBoostingClassifier(random_state=0),
    }
    print("\n=== Comparacion de modelos (dev) ===")
    for nombre, modelo in modelos.items():
        pipe = Pipeline([('escalador', StandardScaler()), ('clf', modelo)])
        scores = cross_val_score(pipe, X_dev, y_dev, cv=sgkf, groups=g_dev, scoring='accuracy')
        print(f"{nombre:22s} exactitud = {scores.mean():.3f} +/- {scores.std():.3f}")

    print("\n=== Barrido de umbral (Random Forest, voto por paciente) ===")
    pipe_rf = Pipeline([('escalador', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=0))])
    proba_dev = cross_val_predict(pipe_rf, X_dev, y_dev, cv=sgkf, groups=g_dev, method='predict_proba')
    np.save('proba_dev_adultos_v2.npy', proba_dev)

    df_dev = df.iloc[idx_dev].copy()
    df_dev['prob_soplo'] = proba_dev[:, 1]
    etiquetas_dev_paciente = df_dev.groupby('paciente_id')['Etiqueta'].first()

    mejor_j, mejor_umbral = -1, None
    print(f"{'umbral':>8} {'sensibilidad':>13} {'especificidad':>15} {'Youden J':>10}")
    for umbral in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]:
        df_dev['pred_ciclo'] = (df_dev['prob_soplo'] >= umbral).astype(int) * 2
        pred_paciente = df_dev.groupby('paciente_id')['pred_ciclo'].apply(voto_mayoria)
        m = confusion_matrix(etiquetas_dev_paciente, pred_paciente)
        tn, fp, fn, vp = m.ravel()
        sens, esp = vp/(vp+fn), tn/(tn+fp)
        j = sens + esp - 1
        print(f"{umbral:>8.2f} {sens:>13.3f} {esp:>15.3f} {j:>10.3f}")
        if j > mejor_j:
            mejor_j, mejor_umbral = j, umbral
    print(f"umbral elegido: {mejor_umbral}")

    print("\n=== Examen final (sujetos nunca vistos) ===")
    modelo_dev = Pipeline([('escalador', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=0))])
    modelo_dev.fit(X_dev, y_dev)
    proba_final = modelo_dev.predict_proba(X_prueba)[:, 1]
    np.save('proba_final_adultos_v2.npy', proba_final)

    df_prueba = df.iloc[idx_prueba].copy()
    df_prueba['prob_soplo'] = proba_final
    df_prueba['pred_ciclo'] = (df_prueba['prob_soplo'] >= mejor_umbral).astype(int) * 2
    pred_paciente_final = df_prueba.groupby('paciente_id')['pred_ciclo'].apply(voto_mayoria)
    etiquetas_paciente_final = df_prueba.groupby('paciente_id')['Etiqueta'].first()

    m_final = confusion_matrix(etiquetas_paciente_final, pred_paciente_final)
    tn, fp, fn, vp = m_final.ravel()
    sens_final, esp_final = vp/(vp+fn), tn/(tn+fp)
    auc_final = roc_auc_score(y_prueba, proba_final)
    print(f"Sensibilidad: {sens_final:.3f}   Especificidad: {esp_final:.3f}   AUC: {auc_final:.3f}")
    print(m_final)

    modelo_final = Pipeline([('escalador', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=0))])
    modelo_final.fit(X, y)
    joblib.dump(modelo_final, 'modelo_pcg_soplo_adultos_rf.joblib')

    metadata = {
        'feature_cols': FEATURE_COLS, 'umbral_decision': mejor_umbral,
        'clases': {'0': 'Sano', '2': 'Soplo'}, 'poblacion': 'adultos',
        'filtro_calidad': 'SQI=1 (PhysioNet2016)',
        'metricas_examen_final': {'sensibilidad': float(sens_final), 'especificidad': float(esp_final),
                                    'auc': float(auc_final), 'n_sujetos_prueba': int(len(etiquetas_paciente_final))},
    }
    with open('modelo_pcg_soplo_adultos_rf_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("\nModelo guardado: modelo_pcg_soplo_adultos_rf.joblib")

if __name__ == "__main__":
    main()