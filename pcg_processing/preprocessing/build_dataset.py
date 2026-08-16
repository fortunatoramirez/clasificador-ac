# build_dataset.py
# Corre esto UNA vez (o cada vez que agregues audios nuevos) para regenerar
# dataset.xlsx, ya con la columna "archivo" que faltaba.

import os
import glob
import pandas as pd
from extract_features import HeartSignalProcessor  # reutiliza tu clase existente

AUDIO_DIR = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\backend_carlos-20260331T010941Z-1-001\backend_carlos\backend_carlos\señales"

def inferir_label(nombre_archivo):
    nombre = nombre_archivo.lower()
    if nombre.startswith("normal"):
        return 0  # Sano
    elif nombre.startswith("click"):
        return 1  # Click
    elif nombre.startswith("murmur"):
        return 2  # Soplo
    return None

def main():
    proc = HeartSignalProcessor()
    todas_las_filas = []
    archivos = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.mp3")))
    print(f"Encontre {len(archivos)} archivos .wav")

    for ruta in archivos:
        nombre = os.path.basename(ruta)
        label = inferir_label(nombre)
        if label is None:
            print(f"  [saltado] {nombre}: no pude inferir la etiqueta del nombre")
            continue
        try:
            x, fs, t = proc.preprocess_audio(ruta)
            Env      = proc.compute_shannon_envelope(x, fs)
            iS1      = proc.detect_cycles(Env, x, t, fs)
            features = proc.extract_features(x, fs, iS1)
        except Exception as e:
            print(f"  [error] {nombre}: {e}")
            continue

        for feat in features:
            fila = {f"MFCC_{i+1}": round(float(feat[i]), 6) for i in range(13)}
            fila["RMS"]      = round(float(feat[13]), 6)
            fila["Etiqueta"] = label
            fila["archivo"]  = nombre
            todas_las_filas.append(fila)
        print(f"  [ok] {nombre}: {len(features)} ciclos")

    df = pd.DataFrame(todas_las_filas)
    df.to_excel("dataset_con_archivo.xlsx", index=False)
    print(f"\nListo: {df.shape[0]} filas de {df['archivo'].nunique()} archivos -> dataset_con_archivo.xlsx")

if __name__ == "__main__":
    main()