# build_dataset_circor_con_duracion.py
# Correr desde pcg_processing/preprocessing/ (junto a extract_features.py)

import os
import glob
import numpy as np
import pandas as pd
from python_speech_features import mfcc
from extract_features import HeartSignalProcessor

TRAINING_DIR = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"
CSV_PATH = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data.csv"
MAX_ARCHIVOS = None   # None = todo el dataset (tarda varios minutos)
UMBRAL_CICLOS = 5
SALIDA = "dataset_circor_con_duracion.xlsx"
FEATURES = [f"MFCC_{i+1}" for i in range(13)] + ["RMS"]


def extract_features_con_duracion(x, fs, iS1_idx):
    """Copia exacta de HeartSignalProcessor.extract_features, que ademas regresa
    la duracion de cada ciclo y si es el ultimo intervalo del audio."""
    Ncoef = 13
    winlen, winstep = 0.025, 0.01
    feats, energias, duraciones, ultimos = [], [], [], []
    n_intervalos = len(iS1_idx) - 1
    for k in range(n_intervalos):
        i1, i2 = iS1_idx[k], iS1_idx[k + 1]
        ciclo = x[i1:i2]
        if len(ciclo) < int(0.012 * fs):
            continue
        frame_len = int(winlen * fs)
        nfft = max(512, 1 << (frame_len - 1).bit_length())
        m = mfcc(ciclo, samplerate=fs, numcep=Ncoef, winlen=winlen, winstep=winstep, nfft=nfft)
        if m.size == 0:
            continue
        feats.append(np.mean(m, axis=0))
        energias.append(np.sqrt(np.mean(ciclo ** 2)))
        duraciones.append(len(ciclo) / fs)
        ultimos.append(k == n_intervalos - 1)
    if not feats:
        raise Exception("No se pudieron extraer MFCC")
    energias = np.array(energias).reshape(-1, 1)
    features = np.hstack([feats, energias[:len(feats)]])
    return features, duraciones, ultimos


def cargar_etiquetas(csv_path):
    """Patient ID -> 0 (Absent) o 2 (Present). 'Unknown' se descarta."""
    df_meta = pd.read_csv(csv_path)
    df_meta["Etiqueta"] = df_meta["Murmur"].map({"Absent": 0, "Present": 2})
    df_meta = df_meta.dropna(subset=["Etiqueta"])
    df_meta["Etiqueta"] = df_meta["Etiqueta"].astype(int)
    return dict(zip(df_meta["Patient ID"].astype(str), df_meta["Etiqueta"]))


def verificar_contra_original(df):
    """Compara las features con dataset_circor.xlsx (el de antes): deben ser identicas."""
    if not os.path.exists("dataset_circor.xlsx"):
        print("(no encontre dataset_circor.xlsx en esta carpeta, no pude comparar)")
        return
    orig = pd.read_excel("dataset_circor.xlsx")
    orig["orden"] = orig.groupby("archivo").cumcount()
    nuevo = df.copy()
    nuevo["orden"] = nuevo.groupby("archivo").cumcount()
    comunes = set(orig["archivo"]) & set(nuevo["archivo"])
    o = orig[orig["archivo"].isin(comunes)].set_index(["archivo", "orden"]).sort_index()
    n = nuevo[nuevo["archivo"].isin(comunes)].set_index(["archivo", "orden"]).sort_index()
    if not o.index.equals(n.index):
        print(f"ATENCION: las filas no coinciden con dataset_circor.xlsx ({len(o)} vs {len(n)} filas en {len(comunes)} archivos comunes)")
        return
    ok = np.allclose(o[FEATURES].values, n[FEATURES].values, atol=1e-5)
    print(f"Features identicas a dataset_circor.xlsx ({len(comunes)} archivos comparados): {ok}")


def main():
    proc = HeartSignalProcessor()
    etiquetas = cargar_etiquetas(CSV_PATH)

    archivos = sorted(glob.glob(os.path.join(TRAINING_DIR, "*.wav")))
    if MAX_ARCHIVOS:
        archivos = archivos[:MAX_ARCHIVOS]
    print(f"Encontre {len(archivos)} archivos .wav")

    filas = []
    for ruta in archivos:
        nombre = os.path.basename(ruta)
        paciente_id = nombre.split("_")[0]
        label = etiquetas.get(paciente_id)
        if label is None:
            print(f"  [saltado] {nombre}: paciente {paciente_id} sin etiqueta confiable (Unknown o no esta en el CSV)")
            continue
        try:
            x, fs, t = proc.preprocess_audio(ruta)
            Env = proc.compute_shannon_envelope(x, fs)
            iS1_idx = proc.detect_cycles(Env, x, t, fs)
            features, duraciones, ultimos = extract_features_con_duracion(x, fs, iS1_idx)
        except Exception as e:
            print(f"  [error] {nombre}: {e}")
            continue

        for feat, dur, ult in zip(features, duraciones, ultimos):
            fila = {f"MFCC_{i+1}": round(float(feat[i]), 6) for i in range(13)}
            fila["RMS"] = round(float(feat[13]), 6)
            fila["Etiqueta"] = label
            fila["archivo"] = nombre
            fila["paciente_id"] = paciente_id
            fila["duracion_ciclo_s"] = round(dur, 4)
            fila["es_ultimo"] = bool(ult)
            filas.append(fila)
        print(f"  [ok] {nombre}: {len(features)} ciclos")

    if not filas:
        print("\nNo se proceso ninguna fila. Revisa los errores de arriba.")
        return

    df = pd.DataFrame(filas)

    ciclos_por_archivo = df.groupby("archivo").size()
    ok_archivos = ciclos_por_archivo[ciclos_por_archivo >= UMBRAL_CICLOS].index
    n_antes = df["archivo"].nunique()
    df = df[df["archivo"].isin(ok_archivos)].reset_index(drop=True)
    print(f"\nFiltro de calidad: {df['archivo'].nunique()} de {n_antes} archivos con >= {UMBRAL_CICLOS} ciclos")

    verificar_contra_original(df)

    df.to_excel(SALIDA, index=False)
    print(f"Listo: {df.shape[0]} filas de {df['paciente_id'].nunique()} pacientes -> {SALIDA}")
    print("(Referencia: el dataset de antes tenia 11153 filas de 586 pacientes cuando se corre completo)")


if __name__ == "__main__":
    main()