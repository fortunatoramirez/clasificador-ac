import os
import glob
import numpy as np
import pandas as pd
from python_speech_features import mfcc
from extract_features import HeartSignalProcessor

TRAINING_DIR = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"
CSV_PATH = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data.csv"
MAX_ARCHIVOS = 20  # Cambia a None cuando desees procesar todo el dataset

def extract_features_con_duracion(x, fs, iS1_idx):
    """Extrae características MFCC, RMS y la duración de cada ciclo detectado."""
    Ncoef = 13
    winlen, winstep = 0.025, 0.01
    feats, energias, duraciones = [], [], []
    for k in range(len(iS1_idx) - 1):
        i1, i2 = iS1_idx[k], iS1_idx[k+1]
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
    if not feats:
        raise Exception("No se pudieron extraer MFCC")
    energias = np.array(energias).reshape(-1, 1)
    features = np.hstack([feats, energias[:len(feats)]])
    return features, duraciones

def cargar_mapeo_etiquetas(csv_path):
    """
    Carga el CSV de metadatos y crea un diccionario mapping:
    Patient_ID (str) -> Etiqueta (0: Normal, 1: Absent/Otro, 2: Murmur)
    """
    # 1. Cargar el DataFrame correctamente al inicio de la función
    df_meta = pd.read_csv(csv_path)
    
    # Mapeo de la columna 'Murmur' a enteros
    label_map = {
        'Absent': 0,
        'Present': 2,
        'Unknown': 1
    }
    
    # Si la columna se llama 'Murmur', ajustamos según ella
    if 'Murmur' in df_meta.columns:
        df_meta['Etiqueta'] = df_meta['Murmur'].map(label_map).fillna(1).astype(int)
    else:
        raise KeyError("No se encontró la columna 'Murmur' en el CSV de entrenamiento.")
        
    df_meta = df_meta.dropna(subset=['Etiqueta'])
    df_meta['Etiqueta'] = df_meta['Etiqueta'].astype(int)
    
    # Crear un diccionario para búsqueda rápida por Patient ID (como string)
    return dict(zip(df_meta['Patient ID'].astype(str), df_meta['Etiqueta']))

def main():
    proc = HeartSignalProcessor()
    
    # 1. Cargar metadatos desde el CSV
    print("Cargando archivo de metadatos CSV...")
    dict_labels = cargar_mapeo_etiquetas(CSV_PATH)
    
    # 2. Buscar archivos WAV
    archivos = sorted(glob.glob(os.path.join(TRAINING_DIR, "*.wav")))
    if MAX_ARCHIVOS:
        archivos = archivos[:MAX_ARCHIVOS]
    print(f"Encontrados {len(archivos)} archivos para procesar")

    todas_las_filas = []
    for ruta in archivos:
        nombre = os.path.basename(ruta)
        
        # Extraer Patient ID (ejemplo: '13918_AV.wav' -> '13918')
        patient_id = nombre.split('_')[0]
        label = dict_labels.get(patient_id)
        
        if label is None:
            print(f"  [saltado] {nombre}: no se encontró ID {patient_id} en el CSV")
            continue
            
        try:
            x, fs, t = proc.preprocess_audio(ruta)
            Env = proc.compute_shannon_envelope(x, fs)
            iS1_idx = proc.detect_cycles(Env, x, t, fs)
            features, duraciones = extract_features_con_duracion(x, fs, iS1_idx)
        except Exception as e:
            print(f"  [error] {nombre}: {e}")
            continue

        for feat, dur in zip(features, duraciones):
            # Inicializar correctamente el diccionario fila antes de asignarle valores
            fila = {f"MFCC_{i+1}": round(float(feat[i]), 6) for i in range(13)}
            fila["RMS"] = round(float(feat[13]), 6)
            fila["paciente_id"] = patient_id
            fila["Etiqueta"] = label
            fila["archivo"] = nombre
            fila["duracion_ciclo_s"] = round(dur, 4)
            todas_las_filas.append(fila)
        print(f"  [ok] {nombre}: {len(features)} ciclos")

    # 3. Guardar dataset
    if not todas_las_filas:
        print("\nNo se procesó ninguna fila correctamente. Verifica las rutas y etiquetas.")
        return

    df = pd.DataFrame(todas_las_filas)
    df.to_excel("dataset_circor_con_duracion.xlsx", index=False)
    print(f"\nListo: {df.shape[0]} filas de {df['archivo'].nunique()} archivos -> dataset_circor_con_duracion.xlsx")

if __name__ == "__main__":
    main()