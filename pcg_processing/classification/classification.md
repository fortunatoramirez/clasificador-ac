# `arboldeprediccion.py` — Clasificación principal del sistema

Script Python que recibe un archivo de audio cardíaco, ejecuta el pipeline completo de procesamiento y devuelve un JSON con el diagnóstico **y todos los datos visuales** del pipeline para mostrar en el frontend. Es el script que llama `server.js` en la ruta `POST /upload`.

---

## ¿Qué hace este archivo?

Es el orquestador principal del sistema. A diferencia de `classify.py`, este script:

- Usa **PyCaret** como framework de clasificación (no `joblib` directamente).
- Devuelve un JSON muy completo que incluye no solo el diagnóstico, sino **todos los datos intermedios del pipeline** para visualizarlos en el dashboard: señal cruda, envolvente, posiciones S1/S2, ciclos superpuestos, heatmap MFCC, etc.
- Suprime toda salida a `stdout` durante la carga del modelo y la predicción para que el único `print()` sea el JSON final (que `server.js` parsea).

---

## Cómo se invoca

```bash
python arboldeprediccion.py <ruta_audio>
```

`server.js` lo lanza así desde `POST /upload`:
```javascript
spawn(pythonExecutable, [PY_CLASIFICAR, filePath])
```

---

## Salida JSON

### Caso exitoso

```json
{
  "status": "success",
  "class": "Sano",
  "confidence": 87.5,
  "cycles": 8,
  "bpm": 72.3,
  "fs": 44100,
  "duration": 10.24,

  "pipeline": {
    "t": [0.0, 0.001, ...],

    "stage_0_raw":          [...],
    "stage_1_highpass":     [...],
    "stage_2_denoised":     [...],
    "stage_3_bandpass":     [...],
    "stage_4a_env_hilbert": [...],
    "stage_4b_env_shannon": [...],

    "stage_5_s1_idxs": [12, 145, 278, ...],
    "stage_5_s2_idxs": [],

    "stage_6_cycles": [
      { "t": [0.0, 0.001, ...], "y": [...] },
      { "t": [0.0, 0.001, ...], "y": [...] }
    ],

    "stage_7_mfcc_mean":   [-312.4, 80.1, ...],
    "stage_7_mfcc_std":    [12.3, 5.6, ...],
    "stage_7_mfcc_matrix": [[...], [...], ...]
  }
}
```

### Caso de error

```json
{
  "status": "error",
  "message": "No se detectaron ciclos válidos"
}
```

---

## Clase `HeartSignalProcessor`

Reutiliza la misma lógica de `extract_features.py` con algunos ajustes menores. Tiene los mismos 3 métodos:

### `preprocess_audio(file_path)`
Carga el audio con `librosa` sin re-muestrear (`sr=None`, respeta el `fs` original). Normaliza la señal dividiendo entre `max(|x|) + 1e-12`.

Diferencia respecto a `extract_features.py`: aquí **no se fuerza una frecuencia de muestreo fija** — se toma la del archivo tal cual.

### `compute_shannon_envelope(x, fs)`
Idéntica a `extract_features.py`. Calcula la envolvente de energía de Shannon y aplica un filtro paso-bajo Butterworth a 9 Hz. Ver documentación de `extract_features.py` para el detalle paso a paso.

### `detect_cycles(Env, x, t, fs)` → devuelve `(iS1_idx, ciclos_ref)`
Idéntica en lógica a `extract_features.py`, pero devuelve **dos valores** en lugar de uno:
- `iS1_idx` — lista de índices de inicio de cada ciclo (para extracción de features).
- `ciclos_ref` — lista de pares `[inicio, fin]` de cada ciclo (para los overlays visuales del frontend).

### `extract_features(x, fs, iS1_idx)` → devuelve `(features, MFCC_matrix)`
Idéntica en lógica, pero también devuelve **`MFCC_matrix`** por separado (lista de vectores MFCC antes de concatenar RMS), necesaria para calcular estadísticas globales y el heatmap.

---

## Funciones auxiliares

### `downsample(signal, t, n=2000)`
Reduce la señal a máximo 2000 puntos para el frontend. Toma una muestra cada `len(signal) // n` muestras. Sin esta reducción, una señal de 44100 Hz × 10 s = 441,000 puntos haría el JSON demasiado grande para transferir por HTTP.

### `suppress_stdout()`
Context manager que redirige `sys.stdout` a `/dev/null` durante su uso. Se aplica a `load_model()` y `predict_model()` porque PyCaret imprime logs de estado que contaminarían el JSON que `server.js` intenta parsear.

### `safe(arr)`
Convierte un array numpy a lista Python, reemplazando `NaN` e `Inf` por `0`. Evita que `json.dumps()` falle con valores no serializables.

---

## Flujo completo de `main()`

```
1. Leer ruta del audio desde sys.argv[1]

2. Cargar el modelo PyCaret
   load_model("models/modelo_pcg_final")
   (stdout suprimido para no contaminar la salida)

3. Procesar la señal:
   preprocess_audio()   → x, fs, t
   compute_shannon_envelope() → Env
   detect_cycles()      → iS1_idx, ciclos_ref
   extract_features()   → features (N×14), MFCC_matrix

4. Clasificar con PyCaret:
   - Construir DataFrame con columnas MFCC_1..13 + RMS
   - predict_model(modelo, data=df)
   - Extraer columna "prediction_label"
   - Votación por mayoría entre todos los ciclos
   - Confianza = (votos_mayoría / total_ciclos) × 100

5. Estimar BPM:
   - Calcular diferencias entre iS1_idx consecutivos (en segundos)
   - BPM = 60 / media(intervalos)

6. Preparar datos visuales del pipeline:
   - Downsample de señal y envolvente a ≤2000 puntos
   - Convertir índices S1 al espacio downsampled (idx // step)
   - Extraer los primeros 4 ciclos como overlays
   - Calcular heatmap MFCC del primer ciclo
   - Calcular media y desviación estándar de MFCC globales

7. Imprimir JSON completo → server.js lo parsea
```

---

## Clasificación con PyCaret

PyCaret es un framework de AutoML que envuelve scikit-learn. El modelo se guarda como `modelo_pcg_final.pkl` (que PyCaret crea con `save_model()`).

La función `predict_model(modelo, data=df)` agrega una columna `prediction_label` al DataFrame con la clase predicha para cada fila (cada ciclo cardíaco).

**Votación por mayoría:** como el modelo predice ciclo a ciclo, el diagnóstico final se determina tomando la clase más frecuente entre todos los ciclos del audio:

```python
unique, counts = np.unique(labels, return_counts=True)
majority   = unique[np.argmax(counts)]           # clase ganadora
confidence = (np.max(counts) / len(labels)) * 100  # % de votos
```

**Mapa de etiquetas:**
```python
label_map = {0: "Sano", 1: "Click", 2: "Soplo"}
```

---

## Datos del pipeline para el frontend

El objeto `pipeline` en el JSON contiene los datos de cada etapa del procesamiento para que el dashboard los grafique:

| Clave | Contenido | Para qué se usa |
|---|---|---|
| `t` | Eje de tiempo (downsampled) | Eje X de todos los gráficos de señal |
| `stage_0_raw` | Señal normalizada | Gráfica "Señal cruda" |
| `stage_1_highpass` | (mismo que raw, simplificado) | Gráfica "Paso alto" |
| `stage_2_denoised` | (mismo que raw, simplificado) | Gráfica "Sin ruido" |
| `stage_3_bandpass` | (mismo que raw, simplificado) | Gráfica "Paso banda" |
| `stage_4a_env_hilbert` | Envolvente calculada | Gráfica de envolvente |
| `stage_4b_env_shannon` | Envolvente calculada | Gráfica de energía de Shannon |
| `stage_5_s1_idxs` | Índices de picos S1 en espacio downsampled | Marcadores sobre la envolvente |
| `stage_5_s2_idxs` | Vacío `[]` (no calculado aquí) | Reservado para S2 |
| `stage_6_cycles` | Primeros 4 ciclos como arrays `{t, y}` | Overlay de latidos superpuestos |
| `stage_7_mfcc_mean` | Vector de 13 medias globales | Gráfico de barras MFCC |
| `stage_7_mfcc_std` | Vector de 13 desviaciones estándar | Barras de error MFCC |
| `stage_7_mfcc_matrix` | Matriz MFCC transpuesta del ciclo 1 | Heatmap MFCC |

> **Nota:** Los stages 1, 2 y 3 actualmente devuelven los mismos datos que la señal cruda. Los slots están reservados para si en el futuro se quiere mostrar cada etapa del filtrado por separado. El filtrado real ocurre internamente pero no se expone en pasos separados.

---

## Comparación con `classify.py`

| Aspecto | `arboldeprediccion.py` | `classify.py` |
|---|---|---|
| **Usado por** | `server.js` (producción) | No está en uso activo |
| **Framework ML** | PyCaret (`load_model`, `predict_model`) | joblib (`joblib.load`) |
| **Features** | 13 MFCC + RMS (14 valores) | 13 MFCC media + 13 MFCC std (26 valores) |
| **Segmentación** | Triángulos en envolvente Shannon | Picos con `find_peaks` + ventanas fijas |
| **Salida JSON** | Diagnóstico + datos completos del pipeline | Solo diagnóstico |
| **Re-muestreo** | `sr=None` (respeta el original) | Fuerza `sr=2000 Hz` |