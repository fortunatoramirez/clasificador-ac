# extract_features.py
# Recibe: ruta_audio label
# Devuelve: JSON con MFCC_1..13 + RMS + label (o error)

import sys
import os
import json
import numpy as np
import warnings
from scipy.signal import butter, filtfilt
from python_speech_features import mfcc
import librosa

warnings.filterwarnings("ignore")


class HeartSignalProcessor:
    def preprocess_audio(self, file_path):
        x, fs = librosa.load(file_path, sr=None, mono=True)
        x = x / (np.max(np.abs(x)) + 1e-12)
        t = np.arange(len(x)) / fs
        return x, fs, t

    def compute_shannon_envelope(self, x, fs):
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
        d = np.diff(Env)
        idx_ext, tipo = [], []
        for i in range(len(d) - 1):
            if d[i] < 0 and d[i + 1] > 0:
                idx_ext.append(i + 1); tipo.append(1)
            elif d[i] > 0 and d[i + 1] < 0:
                idx_ext.append(i + 1); tipo.append(2)

        tri_samp, tri_time, tri_amp = [], [], []
        for k in range(len(tipo) - 2):
            if tipo[k] == 1 and tipo[k+1] == 2 and tipo[k+2] == 1:
                i1, i2, i3 = idx_ext[k], idx_ext[k+1], idx_ext[k+2]
                tri_samp.append([i1, i2, i3])
                tri_time.append([t[i1], t[i2], t[i3]])
                tri_amp.append([Env[i1], Env[i2], Env[i3]])

        if not tri_samp:
            raise Exception("No se detectaron triángulos en la envolvente")

        areas = []
        for i in range(len(tri_time)):
            x1,y1 = tri_time[i][0], tri_amp[i][0]
            x2,y2 = tri_time[i][1], tri_amp[i][1]
            x3,y3 = tri_time[i][2], tri_amp[i][2]
            areas.append(0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))
        areas = np.array(areas)

        Amed = 0.6 * np.max(areas)
        big_idx = np.where(areas > Amed)[0]

        ciclos_ref = []
        minRR, maxRR = int(0.1 * fs), int(2.0 * fs)
        i = 0
        while i < len(big_idx):
            s, e = tri_samp[big_idx[i]][0], tri_samp[big_idx[i]][2]
            if minRR <= (e - s) <= maxRR:
                if not ciclos_ref or s > ciclos_ref[-1][1]:
                    ciclos_ref.append([s, e])
                i += 2
            else:
                i += 1

        if not ciclos_ref:
            raise Exception("No se detectaron ciclos válidos")

        iS1 = [c[0] for c in ciclos_ref]
        iS1.append(len(x) - 1)
        return iS1

    def extract_features(self, x, fs, iS1_idx):
        Ncoef = 13
        winlen, winstep = 0.025, 0.01
        feats, energias = [], []
        for k in range(len(iS1_idx) - 1):
            i1, i2 = iS1_idx[k], iS1_idx[k+1]
            ciclo = x[i1:i2]
            if len(ciclo) < int(0.012 * fs):
                continue
            frame_len = int(winlen * fs)
            nfft = max(512, 1 << (frame_len - 1).bit_length())
            m = mfcc(ciclo, samplerate=fs, numcep=Ncoef,
                     winlen=winlen, winstep=winstep, nfft=nfft)
            if m.size == 0:
                continue
            feats.append(np.mean(m, axis=0))
            energias.append(np.sqrt(np.mean(ciclo ** 2)))

        if not feats:
            raise Exception("No se pudieron extraer MFCC")

        energias = np.array(energias).reshape(-1, 1)
        return np.hstack([feats, energias[:len(feats)]])


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Uso: extract_features.py <ruta_audio> <label>"}))
        sys.exit(1)

    file_path = sys.argv[1]
    label     = int(sys.argv[2])   # 0=Sano 1=Click 2=Soplo

    try:
        proc = HeartSignalProcessor()
        x, fs, t = proc.preprocess_audio(file_path)
        Env      = proc.compute_shannon_envelope(x, fs)
        iS1      = proc.detect_cycles(Env, x, t, fs)
        features = proc.extract_features(x, fs, iS1)

        rows = []
        for feat in features:
            row = {f"MFCC_{i+1}": round(float(feat[i]), 6) for i in range(13)}
            row["RMS"]   = round(float(feat[13]), 6)
            row["label"] = label
            rows.append(row)

        print(json.dumps({"status": "success", "rows": rows, "cycles": len(rows)}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()