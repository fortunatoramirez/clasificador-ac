# classify.py
import sys
import os
import json
import numpy as np
import warnings
import contextlib
import io
from scipy.signal import butter, filtfilt
from python_speech_features import mfcc
from pycaret.classification import load_model, predict_model
import pandas as pd
import librosa

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def suppress_stdout():
    """Redirige stdout temporalmente a /dev/null para silenciar prints de PyCaret."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ==========================================
# CLASE DE PROCESAMIENTO DE SEÑAL
# (Lógica de arboldeprediccion.py)
# ==========================================
class HeartSignalProcessor:
    def __init__(self, fs=None):
        self.target_fs = fs  # None = mantener fs original del audio

    def preprocess_audio(self, file_path):
        """Carga el audio, lo convierte a mono y lo normaliza."""
        try:
            x, fs = librosa.load(file_path, sr=self.target_fs, mono=True)
        except Exception as e:
            raise Exception(f"Error leyendo {file_path}: {e}")

        x = x / (np.max(np.abs(x)) + 1e-12)
        t = np.arange(len(x)) / fs
        return x, fs, t

    def compute_shannon_envelope(self, x, fs):
        """Calcula la envolvente de Shannon y aplica filtro pasa bajas."""
        p = np.abs(x)
        p = p / (np.max(p) + 1e-12)
        E = -p * np.log10(p + 1e-12)

        E_z = (E - np.mean(E)) / (np.std(E) + 1e-12)
        Env0 = (E_z - np.min(E_z)) / (np.max(E_z) - np.min(E_z) + 1e-12)

        fc = 9
        b, a = butter(4, fc / (fs / 2), 'low')
        Env = filtfilt(b, a, Env0)
        Env = (Env - np.min(Env)) / (np.max(Env) - np.min(Env) + 1e-12)

        return Env

    def detect_cycles(self, Env, x, t, fs):
        """
        Detecta ciclos cardíacos usando geometría de triángulos sobre
        los extremos locales de la envolvente (lógica de arboldeprediccion.py).
        Devuelve lista de índices de inicio de cada ciclo S1.
        """
        # --- Extremos locales ---
        d = np.diff(Env)
        idx_ext, tipo = [], []

        for i in range(len(d) - 1):
            if d[i] < 0 and d[i + 1] > 0:
                idx_ext.append(i + 1)
                tipo.append(1)       # mínimo local
            elif d[i] > 0 and d[i + 1] < 0:
                idx_ext.append(i + 1)
                tipo.append(2)       # máximo local

        # --- Triángulos (mín-máx-mín) ---
        tri_samp, tri_time, tri_amp = [], [], []

        for k in range(len(tipo) - 2):
            if tipo[k] == 1 and tipo[k + 1] == 2 and tipo[k + 2] == 1:
                i1, i2, i3 = idx_ext[k], idx_ext[k + 1], idx_ext[k + 2]
                tri_samp.append([i1, i2, i3])
                tri_time.append([t[i1], t[i2], t[i3]])
                tri_amp.append([Env[i1], Env[i2], Env[i3]])

        if len(tri_samp) == 0:
            raise Exception("No se detectaron triángulos en la envolvente")

        # --- Áreas de triángulos ---
        areas = []
        for i in range(len(tri_time)):
            x1, y1 = tri_time[i][0], tri_amp[i][0]
            x2, y2 = tri_time[i][1], tri_amp[i][1]
            x3, y3 = tri_time[i][2], tri_amp[i][2]
            area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
            areas.append(area)
        areas = np.array(areas)

        Amed = 0.6 * np.max(areas)
        mask_big = areas > Amed

        # --- Selección de ciclos S1 ---
        ciclos_ref = []
        minRR = int(0.1 * fs)
        maxRR = int(2.0 * fs)
        big_idx = np.where(mask_big)[0]

        i = 0
        while i < len(big_idx):
            start = tri_samp[big_idx[i]][0]
            end   = tri_samp[big_idx[i]][2]
            if minRR <= (end - start) <= maxRR:
                if len(ciclos_ref) == 0 or start > ciclos_ref[-1][1]:
                    ciclos_ref.append([start, end])
                i += 2
            else:
                i += 1

        if len(ciclos_ref) == 0:
            raise Exception("No se detectaron ciclos válidos")

        iS1_idx = [c[0] for c in ciclos_ref]
        iS1_idx.append(len(x) - 1)

        return iS1_idx

    def extract_features(self, x, fs, iS1_idx):
        """
        Extrae MFCC (13 coeficientes) + RMS por ciclo,
        igual que en arboldeprediccion.py.
        """
        Ncoef = 13
        winlen = 0.025
        winstep = 0.01
        MFCC_matrix = []
        energias = []

        for k in range(len(iS1_idx) - 1):
            i1, i2 = iS1_idx[k], iS1_idx[k + 1]
            ciclo = x[i1:i2]

            if len(ciclo) < int(0.012 * fs):
                continue

            frame_len = int(winlen * fs)
            nfft = max(512, 1 << (frame_len - 1).bit_length())

            m = mfcc(ciclo, samplerate=fs, numcep=Ncoef,
                     winlen=winlen, winstep=winstep, nfft=nfft)
            if m.size == 0:
                continue

            MFCC_matrix.append(np.mean(m, axis=0))
            energias.append(np.sqrt(np.mean(ciclo ** 2)))

        if len(MFCC_matrix) == 0:
            raise Exception("No se extrajeron MFCC")

        energias = np.array(energias).reshape(-1, 1)
        features = np.hstack([MFCC_matrix, energias[:len(MFCC_matrix)]])
        return features


# ==========================================
# MAIN
# ==========================================
def main():
    response = {"status": "error", "message": "Error desconocido"}

    try:
        if len(sys.argv) < 2:
            raise Exception("No se recibió la ruta del archivo")

        file_path = sys.argv[1]

        # Cargar modelo PyCaret (busca modelo_pcg_final.pkl junto al script)
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_name = os.path.join(model_dir, "modelo_pcg_final")

        with suppress_stdout():
            modelo = load_model(model_name)

        # Procesar señal
        processor = HeartSignalProcessor()
        x, fs, t = processor.preprocess_audio(file_path)
        Env       = processor.compute_shannon_envelope(x, fs)
        iS1_idx   = processor.detect_cycles(Env, x, t, fs)
        features  = processor.extract_features(x, fs, iS1_idx)

        # Construir DataFrame con los nombres de columnas que espera el modelo
        Ncoef = 13
        df_cols = [f"MFCC_{i+1}" for i in range(Ncoef)] + ["RMS"]
        df = pd.DataFrame(features, columns=df_cols)

        # Predicción con voto mayoritario entre ciclos
        with suppress_stdout():
            pred = predict_model(modelo, data=df)
        labels = pred["prediction_label"].values

        unique, counts = np.unique(labels, return_counts=True)
        majority = unique[np.argmax(counts)]
        confidence = round((np.max(counts) / len(labels)) * 100, 2)

        # Mapear etiqueta numérica a texto
        label_map = {0: "Sano", 1: "Click", 2: "Soplo"}
        clase = label_map.get(int(majority), "Desconocido")

        response = {
            "status": "success",
            "class": clase,
            "confidence": confidence,
            "cycles": int(len(labels))
        }

    except Exception as e:
        response["message"] = str(e)

    print(json.dumps(response))


if __name__ == "__main__":
    main()