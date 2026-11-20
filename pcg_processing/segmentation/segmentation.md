# `AUX_segmentation_method.py` — Segmentación de señal PCG

Módulo de procesamiento de señal para fonocardiogramas (PCG). Contiene todo el pipeline de filtrado, cálculo de envolvente, estimación de período cardíaco y segmentación de ciclos individuales. Es utilizado como librería auxiliar por los scripts de clasificación y entrenamiento.

---

## ¿Qué hace este archivo?

A diferencia de `extract_features.py`, este módulo **no se ejecuta directamente desde la línea de comandos**. Es una librería que expone funciones reutilizables. Su función principal pública es `smart_segmentation_recursive()`, que recibe la ruta de un audio y devuelve un diccionario con todos los datos del análisis.

El módulo resuelve el problema central del procesamiento PCG: **separar el audio cardíaco en latidos individuales** de forma robusta, incluso cuando la grabación tiene ruido, frecuencia cardíaca variable, o calidad inconsistente.

---

## Función principal: `smart_segmentation_recursive()`

```python
result = smart_segmentation_recursive(file_path, t0=0, t1=None, plot=False)
```

Es el punto de entrada recomendado. Intenta segmentar el audio usando 4 configuraciones de filtros diferentes, en orden de más estándar a más permisiva. Si una configuración falla, prueba automáticamente la siguiente.

### Configuraciones de filtros (en orden de intento)

| Nivel | Nombre | Paso-banda | Orden | Reducción ruido |
|---|---|---|---|---|
| 0 | Standard | 25–200 Hz | 4 | 15 dB |
| 1 | Wide | 20–500 Hz | 3 | 10 dB |
| 2 | LowFreq | 15–150 Hz | 3 | 15 dB |
| 3 | HighSensitivity | 30–600 Hz | 2 | 5 dB |

Una configuración se considera **exitosa** si cumple las 3 condiciones:
- S2 consistente (coeficiente de variación < 15%)
- Frecuencia cardíaca entre 40 y 200 BPM
- Al menos 3 picos S1 detectados

Si todas fallan y se llega al último nivel, acepta el resultado si hay al menos 1 pico detectado.

### Diccionario de retorno

```python
{
  's1_idxs':        np.array,  # índices de picos S1 en la señal
  's2_idxs':        np.array,  # índices de picos S2 en la señal
  's1_onsets':      np.array,  # inicio de cada ciclo (borde izquierdo del S1)
  's1_offsets':     np.array,  # fin del S1
  'cycles':         list,      # lista de arrays numpy, uno por ciclo
  't':              np.array,  # eje de tiempo en segundos
  'signal_raw':     np.array,  # señal original sin filtrar
  'signal_filtered':np.array,  # señal después del filtrado
  'envelope':       np.array,  # envolvente de Hilbert normalizada
  'bpm':            float,     # frecuencia cardíaca estimada
  'status': {
      's2_consistent': bool,   # True si el patrón S2 es regular
      'phys_check':    str,    # estado fisiológico ('OK', 'INVERTED', etc.)
  },
  'quality_score':  int,       # puntaje 0–45 de calidad del análisis
  'used_config':    dict,      # configuración de filtros que funcionó
}
```

Devuelve `None` si el audio es irrecuperable.

---

## Pipeline interno: `process_heart_sound()`

Esta función realiza el procesamiento completo con una configuración de filtros dada. Es llamada por `smart_segmentation_recursive()`.

### Paso 1 — Carga del audio: `preprocessAudioFile()`

Carga el archivo con `librosa` en mono, respetando la frecuencia de muestreo original. Permite recortar el audio a un intervalo `[t0, t1]` en segundos.

---

### Paso 2 — Cadena de filtros

Se aplican 3 filtros en secuencia sobre la señal cruda:

**`highpass_filter(x, fs, cutoff=20)`**
Filtro Butterworth paso-alto de orden 3. Elimina componentes DC y vibraciones de muy baja frecuencia (movimiento del micrófono, respiración muy lenta).

**`spectral_gate_denoise(x, fs, reduction_db=15)`**
Reducción de ruido espectral mediante STFT:
```
1. Calcular espectrograma STFT (ventana 1024, salto 256)
2. Estimar perfil de ruido = percentil 20 de magnitud por frecuencia
3. Calcular umbral = perfil_ruido × 10^(-dB/20)
4. Aplicar máscara binaria: mantener solo componentes > umbral
5. Reconstruir señal con ISTFT
```
Elimina el ruido de fondo ambiente sin distorsionar los transientes cardíacos.

**`bandpass_filter(x, fs, lowcut, highcut)`**
Filtro Butterworth paso-banda. Retiene solo las frecuencias relevantes del sonido cardíaco (típicamente 25–200 Hz). Elimina componentes de muy alta frecuencia (fricción, interferencias eléctricas).

Después del filtrado se aplica **`emphasize_peaks()`**: eleva la señal al cuadrado para amplificar los picos y atenuar el ruido residual de baja amplitud.

---

### Paso 3 — Envolventes

**`compute_hilbert_envelope(x, fs, lowpass_cutoff=20)`**
Calcula la envolvente analítica usando la transformada de Hilbert. Produce una señal siempre positiva que sigue la amplitud instantánea de la señal filtrada. Se aplica un filtro paso-bajo de 20 Hz para suavizarla.

**`compute_shannon_envelope(x, fs, lowpass_cutoff, ...)`**
Calcula la energía de Shannon: `E = -x² · log(x²)`. Es más sensible a sonidos de amplitud media (como S1 y S2) y suprime mejor los extremos que la envolvente de Hilbert. Se usa internamente en `extract_features.py`.

Se usa la **envolvente de Hilbert** para la segmentación en este módulo porque es más estable para la estimación de período.

---

### Paso 4 — Estimación del período: `get_period_double_pass()`

Estima el período cardíaco (tiempo entre latidos) usando **autocorrelación en dos pasadas** para mayor robustez:

```
Pasada 1:
  - Autocorrelación de la envolvente completa
  - Buscar primer pico en el rango [0.3 s, 2.0 s]
  - → lag1 (período candidato)

Pasada 2:
  - Tomar la señal a partir de 2×lag1
  - Repetir la autocorrelación
  - → lag2

Validación:
  - Si |lag1 - lag2| / promedio < 15% → período = promedio(lag1, lag2)
  - Si no → devuelve None (período no confiable)
```

Esto hace que el estimador sea resistente a artefactos al inicio de la grabación.

---

### Paso 5 — Detección de picos S1: `find_Best_Peak_JIT()`

Función compilada con **Numba JIT** para máximo rendimiento. Localiza los picos S1 (primer sonido cardíaco) en la envolvente.

Algoritmo:
```
1. Buscar el mejor pico en la primera ventana de tamaño = período
2. Avanzar en pasos de ≈ período buscando el siguiente pico en ±10%
3. Verificación de outliers (primeros ciclos):
   - Si el primer pico es > 4σ o > 50% de la media del resto → 
     suprimirlo (rellenar con baseline) y reintentar
   - Esto evita que artefactos al inicio descalibren toda la segmentación
4. Para picos débiles (< 50% de la media actual):
   - Ampliar la ventana de búsqueda a ±20%
   - Si en la ventana ampliada hay un pico mejor, usarlo
```

---

### Paso 6 — Detección de picos S2: `find_S2_JIT()`

También compilada con Numba. Para cada par de S1 consecutivos, busca el S2 (segundo sonido cardíaco) dentro de la sístole:

```
Para cada ciclo [s1[i], s1[i+1]]:
  - Ignorar el 10% inicial (zona del S1)
  - Ignorar el 5% final
  - Buscar el máximo local en la zona restante
  → ese es el S2
```

---

### Paso 7 — Validación de S2: `verify_S2_Consistency_JIT()`

Verifica que los intervalos sístole (S1→S2) sean regulares entre ciclos. Calcula el coeficiente de variación (CV = desviación estándar / media). Si CV < 15%, la segmentación es consistente.

---

### Paso 8 — Detección de inversión S1/S2

Si la sístole promedio es **más larga** que la diástole promedio y la frecuencia cardíaca es < 100 BPM, probablemente S1 y S2 estén intercambiados. En ese caso el algoritmo **intercambia los arrays** automáticamente.

Esto ocurre cuando la energía del S2 es mayor que la del S1 en la grabación (común en algunas condiciones patológicas o posicionamiento del micrófono).

---

### Paso 9 — Fronteras de ciclo

**`find_peak_boundaries(peaks, Env)`**
Para cada pico S1, desciende por la envolvente hacia la izquierda y hacia la derecha hasta encontrar un punto de inflexión (inicio de subida). Esos puntos son los `onsets` y `offsets` del S1.

**`segment_cycles(signal, onsets, fs)`**
Corta la señal original en ciclos usando los `onsets`. Cada ciclo va desde el `onset[i]` hasta el `onset[i+1]`. Devuelve una lista de arrays numpy.

---

### Paso 10 — Puntaje de calidad

```python
quality_score = 0
if s2_consistente:         quality_score += 20
if 40 < bpm < 180:         quality_score += 10
if num_picos > 3:          quality_score += 10
if no_invertido:           quality_score += 5
# Máximo posible: 45
```

---

## Controles de rechazo en `process_heart_sound()`

El pipeline rechaza la señal (devuelve `None`) en varios puntos:

| Condición | Razón |
|---|---|
| `max(|x_raw|) < 1e-4` | Señal silenciosa o vacía |
| `max(|x_filtrada|) < 0.005` | El filtro mató la señal |
| `max(|x_filtrada|) / max(|x_raw|) < 0.01` | Señal extremadamente atenuada |
| `mean(envolvente) > 0.35` | Ruido de fondo demasiado alto |
| `período is None` | No se pudo estimar frecuencia cardíaca |
| `bpm > 199` | Frecuencia cardíaca fisiológicamente imposible |

---

## Optimización con Numba JIT

Las funciones `find_Best_Peak_JIT`, `find_S2_JIT`, `verify_S2_Consistency_JIT`, `find_peak_boundaries` y `get_max_peak_in_window` están decoradas con `@jit(nopython=True)`. Esto las compila a código nativo en la primera ejecución, reduciendo el tiempo de procesamiento en señales largas de varios segundos a milisegundos.

---

## Ejemplo de uso

```python
import AUX_segmentation_method as sp

data = sp.smart_segmentation_recursive("audio_cardiaco.wav", plot=False)

if data:
    print(f"Frecuencia cardíaca: {data['bpm']:.1f} BPM")
    print(f"Ciclos detectados: {len(data['cycles'])}")
    print(f"S2 consistente: {data['status']['s2_consistent']}")
    print(f"Puntaje de calidad: {data['quality_score']}/45")
    
    # Cada elemento de cycles es un array numpy con la señal de un latido
    for i, ciclo in enumerate(data['cycles']):
        print(f"  Ciclo {i+1}: {len(ciclo)} muestras")
else:
    print("No se pudo segmentar el audio.")
```

---

## Relación con otros archivos

| Archivo | Cómo usa este módulo |
|---|---|
| `extract_features.py` | Reimplementa la envolvente de Shannon y detección de ciclos de forma simplificada para uso en línea de comandos |
| `arboldeprediccion.py` | Llama a `smart_segmentation_recursive()` para obtener los ciclos antes de clasificar |
| `train.py` | Llama al pipeline para construir el dataset de entrenamiento |