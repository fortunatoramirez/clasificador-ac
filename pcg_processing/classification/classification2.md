# `classify.py` — Clasificador alternativo (no activo en producción)

Script Python de clasificación con un enfoque de segmentación diferente al de `arboldeprediccion.py`. Actualmente **no está en uso** en la ruta principal del servidor — `server.js` llama a `arboldeprediccion.py`. Este archivo representa una versión anterior o experimental del clasificador.

---

## ¿Qué hace este archivo?

Hace esencialmente lo mismo que `arboldeprediccion.py` pero con diferencias importantes en cómo segmenta los ciclos y qué características extrae:

1. Carga el audio y lo **re-muestrea a 2000 Hz** (frecuencia fija).
2. Calcula la **envolvente de Shannon** con filtro paso-bajo a 15 Hz.
3. Segmenta usando **picos con umbral adaptativo** en lugar del algoritmo de triángulos.
4. Extrae **26 características** por ciclo: media y desviación estándar de 13 MFCC.
5. Carga el modelo con **`joblib`** en lugar de PyCaret.
6. Devuelve JSON con diagnóstico (sin datos visuales del pipeline).

---

## Diferencias clave con `arboldeprediccion.py`

| Aspecto | `classify.py` | `arboldeprediccion.py` |
|---|---|---|
| **Estado** | No activo (experimental) | Activo en producción |
| **Framework ML** | `joblib.load()` | PyCaret `load_model()` |
| **Frecuencia de muestreo** | Forzada a 2000 Hz | Respeta la original |
| **Duración máxima** | 5 segundos | Completa |
| **Segmentación** | Picos + ventanas fijas de 0.5 s | Triángulos en envolvente |
| **Features por ciclo** | 26 (13 media + 13 std MFCC) | 14 (13 MFCC + RMS) |
| **Ruta del modelo** | `classification/modelo_pcg_final.pkl` | `models/modelo_pcg_final.pkl` |
| **Datos visuales** | No | Sí (pipeline completo) |

---

## Cómo se invoca

```bash
python classify.py <ruta_audio>
```

---

## Salida JSON

```json
{
  "status": "success",
  "class": "Normal",
  "confidence": 75.0,
  "cycles": 4
}
```

Si no se extraen features:
```json
{
  "status": "success",
  "class": "No concluyente",
  "confidence": 0,
  "cycles": 0
}
```

---

## Clase `HeartSignalProcessor`

### `preprocess_audio(file_path, duration=5)`

Carga máximo 5 segundos de audio y lo re-muestrea a 2000 Hz. Aplica una normalización diferente a la de `arboldeprediccion.py`:

```python
# classify.py — normalización min-max al rango [-1, 1]
x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
x = x * 2 - 1

# arboldeprediccion.py — normalización por valor máximo absoluto
x = x / (np.max(np.abs(x)) + 1e-12)
```

La normalización min-max de `classify.py` es más agresiva: garantiza que la señal use todo el rango [-1, 1] independientemente de su amplitud original. La de `arboldeprediccion.py` preserva las proporciones relativas de amplitud.

---

### `compute_shannon_envelope(x, fs)`

Misma lógica general que `arboldeprediccion.py`, con una diferencia: el filtro paso-bajo se aplica a **15 Hz** en lugar de 9 Hz. Una frecuencia de corte más alta deja pasar más variaciones rápidas en la envolvente, lo que puede ayudar a detectar ciclos cuando la frecuencia cardíaca es alta.

---

### `segment_cycles(Env, fs)` — Enfoque diferente

Este es el método más distinto del archivo. En lugar del algoritmo de triángulos, usa **`find_peaks` de scipy** con umbral adaptativo y luego extrae **ventanas fijas** alrededor de cada pico:

```
1. Calcular umbral = media(Env) × 1.1
2. Detectar picos con:
   - altura mínima = umbral
   - distancia mínima entre picos = 0.15 s (evita dobles detecciones)
3. Por cada pico detectado:
   - Definir ventana de 0.5 s centrada en el pico
   - Si la ventana es mayor a 0.30 s → agregarla como ciclo
```

**Ventaja:** más simple y directo, no depende de la geometría de la envolvente.  
**Desventaja:** las ventanas de 0.5 s son fijas — no se adaptan al período cardíaco real, lo que puede incluir parte del ciclo anterior o siguiente si la frecuencia cardíaca varía.

---

### `extract_features(x, fs, cycles)`

Usa **`librosa.feature.mfcc()`** en lugar de `python_speech_features.mfcc()`. La diferencia práctica:

| Parámetro | `classify.py` (librosa) | `arboldeprediccion.py` (python_speech_features) |
|---|---|---|
| FFT | `n_fft=2048` | Adaptado al ciclo |
| Salto | `hop_length=512` | `winstep=10ms` |
| Features por ciclo | 26 (media + std de 13) | 14 (media de 13 + RMS) |

Extraer tanto la **media como la desviación estándar** de los MFCC da más información sobre la variabilidad espectral del ciclo, pero requiere que el modelo haya sido entrenado con vectores de 26 dimensiones. Si se usa con `modelo_pcg_final.pkl` (entrenado con 14 features), fallará en la predicción.

---

## Flujo completo de `main()`

```
1. Leer ruta del audio desde sys.argv[1]
2. Verificar que existe modelo_pcg_final.pkl junto al script
3. Cargar modelo con joblib.load()
4. preprocess_audio() → x (5 s, 2000 Hz, normalizado [-1,1])
5. compute_shannon_envelope() → Env
6. segment_cycles() → lista de (inicio, fin) de ventanas
7. extract_features() → array de shape (N, 26)
8. Si N == 0 → devolver "No concluyente"
9. Si N > 0:
   - clf.predict(feats) → array de etiquetas
   - Votación por mayoría
   - confidence = (votos_mayoría / total) × 100
10. Imprimir JSON
```

---

## Por qué no está activo en producción

`server.js` define la ruta del script clasificador en:

```javascript
const PY_CLASIFICAR = path.join(__dirname,
  '../../pcg_processing/classification/arboldeprediccion.py');
```

`classify.py` está en la misma carpeta pero nunca se referencia desde `server.js`. Las razones probables:

1. **Incompatibilidad de dimensiones:** el modelo actual fue entrenado con 14 features (MFCC×13 + RMS). `classify.py` extrae 26 features, por lo que la predicción fallaría con ese modelo.
2. **Falta de datos visuales:** `classify.py` no devuelve los datos del pipeline que el dashboard necesita graficar.
3. **Ruta del modelo incorrecta:** busca el `.pkl` en `classification/modelo_pcg_final.pkl` pero el modelo real está en `models/modelo_pcg_final.pkl`.
4. **Framework diferente:** el modelo actual se guarda con PyCaret (`save_model`), que no es compatible con `joblib.load()` directamente.

---

## Estado recomendado

Este archivo puede mantenerse como referencia o punto de partida para experimentos con modelos alternativos, pero no debe conectarse a `server.js` sin antes:

- Re-entrenar un modelo con vectores de 26 dimensiones.
- Corregir la ruta del modelo.
- Agregar los datos del pipeline a la respuesta JSON.