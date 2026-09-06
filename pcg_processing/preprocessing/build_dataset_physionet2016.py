# build_dataset_physionet2016.py


import os
import glob
import pandas as pd
import numpy as np
from extract_features import HeartSignalProcessor

BASE_DIR = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\classification-of-heart-sound-recordings\classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
CARPETAS = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']

MAX_ARCHIVOS_POR_CARPETA = 20  # empieza chico para confirmar que corre bien

def main():
    proc = HeartSignalProcessor()
    todas_las_filas = []

    for carpeta in CARPETAS:
        ruta_carpeta = os.path.join(BASE_DIR, carpeta)
        if not os.path.isdir(ruta_carpeta):
            print(f"[saltada] {carpeta}: no existe esa carpeta")
            continue

        ref = pd.read_csv(os.path.join(ruta_carpeta, 'REFERENCE.csv'), header=None, names=['archivo', 'label'])
        if MAX_ARCHIVOS_POR_CARPETA:
            ref = ref.head(MAX_ARCHIVOS_POR_CARPETA)

        print(f"\n=== {carpeta}: {len(ref)} archivos ===")
        for _, row in ref.iterrows():
            nombre = row['archivo']
            label = 2 if row['label'] == 1 else 0  # 1=abnormal->Soplo(2), -1=normal->Sano(0)
            ruta = os.path.join(ruta_carpeta, nombre + '.wav')

            try:
                x, fs, t = proc.preprocess_audio(ruta)
                Env = proc.compute_shannon_envelope(x, fs)
                iS1_idx = proc.detect_cycles(Env, x, t, fs)
                features = proc.extract_features(x, fs, iS1_idx)
                features = np.asarray(features)
                if features.ndim != 2 or features.shape[1] != 14:
                    raise ValueError(f"forma inesperada: {features.shape}")
            except Exception as e:
                print(f"  [error] {nombre}: {e}")
                continue

            for fila_feat in features:
                fila = {f"MFCC_{i+1}": round(float(fila_feat[i]), 6) for i in range(13)}
                fila["RMS"] = round(float(fila_feat[13]), 6)
                fila["Etiqueta"] = label
                fila["archivo"] = f"{carpeta}_{nombre}"
                fila["paciente_id"] = f"{carpeta}_{nombre}"  # 1 grabacion = 1 sujeto en este dataset
                fila["fuente"] = "PhysioNet2016"
                todas_las_filas.append(fila)
            print(f"  [ok] {nombre}: {len(features)} ciclos")

    df = pd.DataFrame(todas_las_filas)
    df.to_excel("dataset_physionet2016.xlsx", index=False)
    print(f"\nListo: {df.shape[0]} filas de {df['paciente_id'].nunique()} sujetos -> dataset_physionet2016.xlsx")

if __name__ == "__main__":
    main()