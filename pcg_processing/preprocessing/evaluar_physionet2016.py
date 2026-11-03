# evaluar_physionet2016.py
import os
import json
import numpy as np
import pandas as pd
import joblib
import traceback  # Permite rastrear la línea exacta del error
from sklearn.metrics import confusion_matrix
from extract_features import HeartSignalProcessor

print("VERSION_DE_PRUEBA_2 (Con depuración de errores)")

TRAINING_A_DIR = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0\training-a"
MODEL_PATH = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\clasificador-ac\models\modelo_pcg_soplo_rf.joblib"
METADATA_PATH = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\clasificador-ac\models\modelo_pcg_soplo_rf_metadata.json"

def voto_mayoria(etiquetas):
    if len(etiquetas) == 0:
        return 0
    vals, counts = np.unique(etiquetas, return_counts=True)
    return int(vals[np.argmax(counts)])

def main():
    ref_path = os.path.join(TRAINING_A_DIR, 'REFERENCE.csv')
    ref = pd.read_csv(ref_path, header=None, names=['archivo', 'label'])
    
    print(f"{len(ref)} grabaciones a evaluar en PhysioNet 2016")

    modelo = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    feature_cols = metadata['feature_cols']
    umbral = metadata['umbral_decision']
    idx_soplo = list(modelo.classes_).index(2)

    proc = HeartSignalProcessor()
    resultados = []

    for _, row in ref.iterrows():
        nombre = row['archivo']
        real = 2 if row['label'] == 1 else 0
        ruta = os.path.join(TRAINING_A_DIR, nombre + '.wav')

        try:
            x, fs, t = proc.preprocess_audio(ruta)
            Env = proc.compute_shannon_envelope(x, fs)
            iS1_idx = proc.detect_cycles(Env, x, t, fs)
            
            # POSIBLE SOLUCIÓN AQUÍ: 
            # Si extract_features devuelve 3 valores, usa: features, _, _ = ...
            # Si devuelve solo 1 valor, usa: features = ...
            features = proc.extract_features(x, fs, iS1_idx)

            features = np.asarray(features)
            if features.ndim != 2 or features.shape[1] != 14:
                raise ValueError(f"forma inesperada de features: {features.shape}")

            df = pd.DataFrame(features, columns=[f"MFCC_{i+1}" for i in range(13)] + ["RMS"])
            df = df[feature_cols]
            prob = modelo.predict_proba(df)[:, idx_soplo]
            pred_ciclo = np.where(prob >= umbral, 2, 0)
            pred_final = voto_mayoria(pred_ciclo)
            
            resultados.append({'archivo': nombre, 'real': real, 'predicho': pred_final, 'n_ciclos': len(pred_ciclo)})
            print(f"  {nombre}: real={'Sano' if real==0 else 'Soplo':6s} predicho={'Sano' if pred_final==0 else 'Soplo':6s} ({len(pred_ciclo)} ciclos)")

        except Exception as e:
            print(f"\n[ERROR EN ARCHIVO] {nombre}: {e}")
            # Esto imprimirá la línea exacta de extract_features.py que causó el problema
            traceback.print_exc()
            continue

    df_res = pd.DataFrame(resultados)
    
    # Previene el KeyError si la lista df_res está vacía
    if df_res.empty:
        print("\n[!] CRÍTICO: Todas las grabaciones fallaron. No se puede calcular la matriz de confusión. Revisa el error impreso arriba para arreglar HeartSignalProcessor.")
        return

    df_res.to_csv('resultados_physionet2016.csv', index=False)

    m = confusion_matrix(df_res['real'], df_res['predicho'], labels=[0, 2])
    tn, fp, fn, tp = m.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    esp = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    print(f"\n=== RESULTADO EN PHYSIONET 2016 (training-a) ===")
    print(f"Evaluados: {len(df_res)} de {len(ref)}")
    print(f"Sensibilidad: {sens:.3f}   Especificidad: {esp:.3f}")
    print(f"Matriz de confusion:\n{m}")

if __name__ == "__main__":
    main()