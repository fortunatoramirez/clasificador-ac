

# build_dataset_circor.py


import os
import glob
import pandas as pd
from extract_features import HeartSignalProcessor

TRAINING_DIR = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"
CSV_PATH = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data.csv"

MAX_ARCHIVOS = None  # TODO: pon None para procesar todo, una vez que confirmes que funciona con pocos

def main():
    meta = pd.read_csv(CSV_PATH)
    meta['Patient ID'] = meta['Patient ID'].astype(str)
    mapa_murmur = dict(zip(meta['Patient ID'], meta['Murmur']))

    proc = HeartSignalProcessor()
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

        label = 0 if murmur == "Absent" else 2   # Absent=Sano(0)  Present=Soplo(2)

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
            fila["RMS"]         = round(float(feat[13]), 6)
            fila["Etiqueta"]    = label
            fila["archivo"]     = nombre
            fila["paciente_id"] = paciente_id
            todas_las_filas.append(fila)
        print(f"  [ok] {nombre} (paciente {paciente_id}, {murmur}): {len(features)} ciclos")
    
    df = pd.DataFrame(todas_las_filas)

    UMBRAL_CICLOS = 5
    ciclos_por_archivo = df.groupby('archivo').size()
    archivos_confiables = ciclos_por_archivo[ciclos_por_archivo >= UMBRAL_CICLOS].index
    n_antes = df['archivo'].nunique()
    df = df[df['archivo'].isin(archivos_confiables)]
    print(f"\nFiltro de calidad: {df['archivo'].nunique()} de {n_antes} archivos pasaron el umbral de {UMBRAL_CICLOS} ciclos")

    df.to_excel("dataset_circor.xlsx", index=False)
    print(f"Listo: {df.shape[0]} filas de {df['paciente_id'].nunique()} pacientes -> dataset_circor.xlsx")

if __name__ == "__main__":
    main()