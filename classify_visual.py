# classify_visual.py
# Usa EXACTAMENTE el mismo pipeline que classify.py
# y devuelve los campos que espera pcg-dashboard.html

import sys, os, json, warnings
import numpy as np
import joblib
import librosa
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")

MAX_POINTS = 2000
TARGET_FS  = 2000
DURATION   = 5


# ─────────────────────────────────────────────────────────────────────────────
# CLASE IDENTICA A classify.py
# ─────────────────────────────────────────────────────────────────────────────

class HeartSignalProcessor:
    def __init__(self, fs=2000):
        self.target_fs = fs

    def preprocess_audio(self, file_path, duration=5):
        try:
            x, fs = librosa.load(file_path, sr=self.target_fs, duration=duration)
        except Exception as e:
            return None, None
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
        x = x * 2 - 1
        return x, fs

    def compute_shannon_envelope(self, x, fs):
        p    = np.abs(x)
        p    = p / (np.max(p) + 1e-8)
        E    = -p * np.log10(p + 1e-8)
        E_z  = (E - np.mean(E)) / (np.std(E) + 1e-8)
        Env0 = (E_z - np.min(E_z)) / (np.max(E_z) - np.min(E_z) + 1e-8)
        b, a = butter(4, 15 / (0.5 * fs), btype='low')
        Env  = filtfilt(b, a, Env0)
        Env  = (Env - np.min(Env)) / (np.max(Env) - np.min(Env) + 1e-8)
        return Env

    def segment_cycles(self, Env, fs):
        threshold = np.mean(Env) * 1.1
        peaks, _  = find_peaks(Env, height=threshold, distance=int(0.15 * fs))
        if len(peaks) < 2:
            return []
        window_size = int(0.5 * fs)
        min_rr      = int(0.30 * fs)
        regions = []
        for p in peaks:
            s = max(0, p - window_size // 2)
            e = min(len(Env), p + window_size // 2)
            if (e - s) > min_rr:
                regions.append((s, e))
        return regions

    def extract_features(self, x, fs, cycles):
        features_list = []
        for (s, e) in cycles:
            seg = x[s:e]
            if len(seg) < 512:
                continue
            mfccs = librosa.feature.mfcc(y=seg, sr=fs, n_mfcc=13,
                                          n_fft=2048, hop_length=512)
            features_list.append(np.concatenate([np.mean(mfccs, axis=1),
                                                  np.std(mfccs,  axis=1)]))
        return np.array(features_list)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ds(signal, t, n=MAX_POINTS):
    """Downsample manteniendo representatividad."""
    if len(signal) <= n:
        return t.tolist(), signal.tolist()
    step = len(signal) // n
    return t[::step].tolist(), signal[::step].tolist()

def safe(arr):
    return np.nan_to_num(np.asarray(arr, dtype=float)).tolist()

def split_peaks(peaks):
    """Separar picos alternados como S1/S2 (simplificado)."""
    s1 = peaks[0::2]
    s2 = peaks[1::2]
    return s1, s2


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    response = {"status": "error", "message": "Error desconocido"}

    try:
        if len(sys.argv) < 2:
            raise Exception("No se recibio la ruta del archivo")

        file_path  = sys.argv[1]
        model_path = os.path.join(os.path.dirname(__file__), 'heart_sound_model.pkl')

        if not os.path.exists(model_path):
            raise Exception("Modelo no encontrado. Ejecuta train.py primero.")

        clf       = joblib.load(model_path)
        processor = HeartSignalProcessor()

        # ── Etapa 0: carga y normalizacion ───────────────────────────────────
        x, fs = processor.preprocess_audio(file_path, duration=DURATION)
        if x is None:
            raise Exception("Error leyendo audio")
        t = np.arange(len(x)) / fs

        # ── Etapa 4b: envolvente Shannon (la que usa classify.py) ────────────
        Env = processor.compute_shannon_envelope(x, fs)

        # ── Etapas 5-6: segmentacion y features ─────────────────────────────
        cycles = processor.segment_cycles(Env, fs)
        feats  = processor.extract_features(x, fs, cycles)

        # ── Clasificacion identica a classify.py ─────────────────────────────
        if len(feats) == 0:
            diag_class      = "No concluyente"
            diag_confidence = 0
            diag_cycles     = 0
        else:
            preds          = clf.predict(feats)
            unique, counts = np.unique(preds, return_counts=True)
            majority       = unique[np.argmax(counts)]
            conf           = (np.max(counts) / len(preds)) * 100
            diag_class     = str(majority)
            diag_confidence = round(float(conf), 2)
            diag_cycles     = int(len(preds))

        # ── Picos y BPM ───────────────────────────────────────────────────────
        threshold  = np.mean(Env) * 1.1
        all_peaks, _ = find_peaks(Env, height=threshold, distance=int(0.15 * fs))

        bpm_est = 0.0
        if len(all_peaks) > 1:
            intervals = np.diff(all_peaks) / fs
            bpm_est   = round(float(60.0 / np.mean(intervals)), 1)

        s1_raw, s2_raw = split_peaks(all_peaks)

        # ── Downsample para frontend ──────────────────────────────────────────
        step       = max(1, len(x) // MAX_POINTS)
        t_ds, x_ds = ds(x,   t)
        _,  env_ds  = ds(Env, t)

        # indices de picos en el vector downsampleado
        s1_plot = (s1_raw[s1_raw < len(t)] // step).tolist()
        s2_plot = (s2_raw[s2_raw < len(t)] // step).tolist()

        # ciclos para overlay (primeros 4)
        cycles_overlay = []
        for (s, e) in cycles[:4]:
            seg = x[s:e]
            cycles_overlay.append({
                "t": (np.arange(len(seg)) / fs).tolist(),
                "y": safe(seg)
            })

        # MFCC heatmap del primer ciclo
        mfcc_matrix = []
        if cycles:
            s0, e0 = cycles[0]
            seg0   = x[s0:e0]
            if len(seg0) >= 512:
                m = librosa.feature.mfcc(y=seg0, sr=fs, n_mfcc=13,
                                          n_fft=2048, hop_length=512)
                mfcc_matrix = safe(m)

        # stats MFCC globales
        if len(feats) > 0:
            mfcc_mean_g = safe(np.mean(feats[:, :13], axis=0))
            mfcc_std_g  = safe(np.mean(feats[:, 13:], axis=0))
        else:
            mfcc_mean_g = [0] * 13
            mfcc_std_g  = [0] * 13

        # ── Respuesta con los nombres que espera pcg-dashboard.html ───────────
        # Las etapas 1, 2, 3 y 4a no existen en classify.py,
        # asi que devolvemos la señal disponible mas cercana en cada caso.
        response = {
            "status":     "success",
            "class":      diag_class,
            "confidence": diag_confidence,
            "cycles":     diag_cycles,
            "bpm":        bpm_est,
            "fs":         int(fs),
            "duration":   round(float(len(x) / fs), 2),

            "pipeline": {
                "t": t_ds,

                # Etapa 0 — señal normalizada (lo que realmente usa el modelo)
                "stage_0_raw":         x_ds,

                # Etapas 1-3 — classify.py no las calcula
                # Mostramos la misma señal para que el dashboard no quede vacio
                "stage_1_highpass":    x_ds,
                "stage_2_denoised":    x_ds,
                "stage_3_bandpass":    x_ds,

                # Etapa 4 — solo Shannon existe en classify.py
                "stage_4a_env_hilbert": env_ds,   # mismo que Shannon (unico disponible)
                "stage_4b_env_shannon": env_ds,

                # Etapas 5-6
                "stage_5_s1_idxs": s1_plot,
                "stage_5_s2_idxs": s2_plot,
                "stage_6_cycles":  cycles_overlay,

                # Etapa 7
                "stage_7_mfcc_mean":   mfcc_mean_g,
                "stage_7_mfcc_std":    mfcc_std_g,
                "stage_7_mfcc_matrix": mfcc_matrix,
            }
        }

    except Exception as e:
        response["message"] = str(e)

    print(json.dumps(response))


if __name__ == "__main__":
    main()