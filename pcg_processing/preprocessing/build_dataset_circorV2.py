# build_dataset_circor_v2.py
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert, find_peaks
from python_speech_features import mfcc
import librosa
import traceback

        
TRAINING_DIR = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"
CSV_PATH = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data.csv"
FS = 4000

MAX_ARCHIVOS = None  # empieza chico, sube a None cuando confirmes que corre bien

# --- preprocesamiento ---
def load_audio(file_path, target_fs=FS):
    x, _ = librosa.load(file_path, sr=target_fs, mono=True)
    return x

def bandpass_filter(x, fs, low=20, high=400, order=4):
    nyquist = fs / 2
    b, a = butter(order, [low/nyquist, high/nyquist], btype='band')
    return filtfilt(b, a, x)

def normalize(x):
    return x / (np.max(np.abs(x)) + 1e-12)

def preprocess(x, fs, low=20, high=400):
    return normalize(bandpass_filter(x, fs, low, high))

# --- segmentacion ---
def obtener_envolvente_cruda2(x, fs):
    senal_analitica = hilbert(x)
    envolvente = np.abs(senal_analitica)
    nyq = 0.5 * fs
    b, a = butter(2, 10 / nyq, btype='low')
    return filtfilt(b, a, envolvente)

def extraer_y_filtrar_ciclos_por_promedio(x, fs, margen_antes=0.05, margen_despues=0.10, umbral_porcentaje=0.80, graficar=False):
    t_rec = np.arange(len(x)) / fs
    x_rec = x
    env_cruda = obtener_envolvente_cruda2(x_rec, fs)
    distancia_minima = int(0.15 * fs)

    picos_todos, _ = find_peaks(env_cruda, distance=distancia_minima, prominence=np.std(env_cruda) * 0.2)

    artefactos = []
    if len(picos_todos) > 0:
        alturas = env_cruda[picos_todos]
        mediana_alturas = np.median(alturas)
        es_artefacto = alturas > mediana_alturas * 2.5
        artefactos = picos_todos[es_artefacto].tolist()
        picos_brutos = picos_todos[~es_artefacto]
    else:
        picos_brutos = picos_todos

    max_real = np.max(env_cruda[picos_brutos]) + 1e-12 if len(picos_brutos) > 0 else np.max(env_cruda) + 1e-12
    env_norm_final = env_cruda / max_real

    s1_picos_temp, s2_picos_temp, picos_descartados = [], [], []
    ventana_max_sistole = int(0.45 * fs)

    i = 0
    while i < len(picos_brutos):
        p1 = picos_brutos[i]
        if i + 1 >= len(picos_brutos):
            picos_descartados.append(p1)
            break
        p2 = picos_brutos[i+1]
        if p2 - p1 <= ventana_max_sistole:
            desplazamiento_aplicado = False
            if i + 2 < len(picos_brutos):
                p3 = picos_brutos[i+2]
                if p3 - p2 <= ventana_max_sistole:
                    if env_norm_final[p1] < 0.5 and env_norm_final[p3] > env_norm_final[p1]:
                        picos_descartados.append(p1)
                        s1_picos_temp.append(p2)
                        s2_picos_temp.append(p3)
                        i += 3
                        desplazamiento_aplicado = True
            if not desplazamiento_aplicado:
                s1_picos_temp.append(p1)
                s2_picos_temp.append(p2)
                i += 2
        else:
            picos_descartados.append(p1)
            i += 1

    s1_picos, s2_picos = [], []
    if len(s1_picos_temp) > 0:
        duraciones = [s2 - s1 for s1, s2 in zip(s1_picos_temp, s2_picos_temp)]
        mediana_duracion = np.median(duraciones)
        umbral_duracion = mediana_duracion * umbral_porcentaje
        for s1, s2, duracion in zip(s1_picos_temp, s2_picos_temp, duraciones):
            if duracion >= umbral_duracion:
                s1_picos.append(s1)
                s2_picos.append(s2)
            else:
                picos_descartados.extend([s1, s2])

    ciclos_senial, ciclos_tiempo, limites_ciclos = [], [], []
    for s1, s2 in zip(s1_picos, s2_picos):
        idx_inicio = max(0, int(s1 - margen_antes * fs))
        idx_fin = min(len(x_rec), int(s2 + margen_despues * fs))
        ciclos_senial.append(x_rec[idx_inicio:idx_fin])
        ciclos_tiempo.append(t_rec[idx_inicio:idx_fin])
        limites_ciclos.append((t_rec[idx_inicio], idx_fin / fs))

    return s1_picos, s2_picos, ciclos_senial, ciclos_tiempo, picos_descartados

# --- features ---
def extraer_mfcc_de_ciclos(ciclos_senial, fs, ncoef=13, winlen=0.025, winstep=0.01):
    filas = []
    ciclos_fallidos = 0
    for ciclo in ciclos_senial:
        if len(ciclo) < int(0.012 * fs):
            continue
        frame_len = int(winlen * fs)
        nfft = max(512, 1 << (frame_len - 1).bit_length())
        try:
            m = mfcc(ciclo, samplerate=fs, numcep=ncoef, winlen=winlen, winstep=winstep, nfft=nfft)
        except IndexError:
            ciclos_fallidos += 1
            continue
        if m.size == 0:
            continue
        mfcc_prom = np.mean(m, axis=0)
        rms = np.sqrt(np.mean(ciclo ** 2))
        filas.append(np.append(mfcc_prom, rms))
    if ciclos_fallidos > 0:
        print(f"    (se saltaron {ciclos_fallidos} ciclos individuales por error de indice)")
    return filas

# --- main ---
def main():
    meta = pd.read_csv(CSV_PATH)
    meta['Patient ID'] = meta['Patient ID'].astype(str)
    mapa_murmur = dict(zip(meta['Patient ID'], meta['Murmur']))

    archivos = sorted(glob.glob(os.path.join(TRAINING_DIR, "*.wav")))
    if MAX_ARCHIVOS:
        archivos = archivos[:MAX_ARCHIVOS]
    print(f"Encontre {len(archivos)} archivos .wav")

    todas_las_filas = []
    for ruta in archivos:
        nombre = os.path.basename(ruta)
        paciente_id = nombre.split('_')[0]

        murmur = mapa_murmur.get(paciente_id)
        if murmur is None:
            print(f"  [saltado] {nombre}: paciente {paciente_id} no esta en el CSV")
            continue
        if murmur == "Unknown":
            print(f"  [saltado] {nombre}: etiqueta 'Unknown', no confiable")
            continue
        label = 0 if murmur == "Absent" else 2

        try:
            x_crudo = load_audio(ruta)
            x_limpio = preprocess(x_crudo, FS)
            _, _, ciclos, _, _ = extraer_y_filtrar_ciclos_por_promedio(x_limpio, FS, graficar=False)
            filas_features = extraer_mfcc_de_ciclos(ciclos, FS)
        except Exception as e:
            print(f"  [error] {nombre}: {e}")
            traceback.print_exc()
            continue

        for feat in filas_features:
            fila = {f"MFCC_{i+1}": round(float(feat[i]), 6) for i in range(13)}
            fila["RMS"] = round(float(feat[13]), 6)
            fila["Etiqueta"] = label
            fila["archivo"] = nombre
            fila["paciente_id"] = paciente_id
            todas_las_filas.append(fila)
        print(f"  [ok] {nombre} (paciente {paciente_id}, {murmur}): {len(filas_features)} ciclos")

    df = pd.DataFrame(todas_las_filas)
    df.to_excel("dataset_circor_v2.xlsx", index=False)
    print(f"\nListo: {df.shape[0]} filas de {df['paciente_id'].nunique()} pacientes -> dataset_circor_v2.xlsx")

if __name__ == "__main__":
    main()