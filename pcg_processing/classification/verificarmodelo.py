# verificar_modelo.py
import sys
import subprocess
import json
import glob
import os
import pandas as pd

CSV_PATH  = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data.csv"
AUDIO_DIR = r"C:\Users\emigo\OneDrive\Documentos\Servicio Social\the-circor-digiscope-phonocardiogram-dataset-1.0.3\the-circor-digiscope-phonocardiogram-dataset-1.0.3\training_data"
N_MUESTRA = 30

meta = pd.read_csv(CSV_PATH)
meta = meta[meta['Murmur'].isin(['Absent', 'Present'])]
muestra = meta.sample(N_MUESTRA, random_state=1)

aciertos = 0
evaluados = 0
for _, row in muestra.iterrows():
    pid = str(row['Patient ID'])
    real = 'Sano' if row['Murmur'] == 'Absent' else 'Soplo'

    candidatos = glob.glob(os.path.join(AUDIO_DIR, f"{pid}_*.wav"))
    if not candidatos:
        print(f"{pid}: sin audio encontrado, saltado")
        continue
    ruta = candidatos[0]

    resultado = subprocess.run([sys.executable, 'arboldeprediccion.py', ruta],
                            capture_output=True, text=True, timeout=60)
    try:
        resp = json.loads(resultado.stdout)
        predicho = resp.get('class', 'ERROR')
    except Exception:
        predicho = f"ERROR ({resultado.stderr[:80]})"

    acierto = (predicho == real)
    aciertos += acierto
    evaluados += 1
    print(f"{pid}: real={real:6s} predicho={predicho:10s} {'OK' if acierto else 'X'}")

print(f"\nAciertos: {aciertos}/{evaluados} ({100*aciertos/evaluados:.0f}%)")