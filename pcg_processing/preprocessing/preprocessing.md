# `extract_features.py` — Extracción de características

Script Python que recibe un archivo de audio cardíaco y una etiqueta, y devuelve un JSON con las características numéricas extraídas de cada ciclo cardíaco detectado. Es llamado desde `server.js` como subproceso.

---

## ¿Qué hace este archivo?

Dada una grabación de audio, el script:

1. **Carga y normaliza** la señal de audio.
2. **Calcula la envolvente de Shannon** para detectar la energía del sonido cardíaco.
3. **Detecta los ciclos cardíacos** (latidos individuales) usando la envolvente.
4. **Extrae 14 características** por ciclo: 13 coeficientes MFCC + 1 valor RMS.
5. **Imprime un JSON** con todas las filas resultantes listas para guardarse en el dataset.

---

## Cómo se invoca

```bash
python extract_features.py <ruta_audio> <label>
```

| Argumento | Tipo | Valores posibles |
|---|---|---|
| `ruta_audio` | `str` | Ruta absoluta al archivo `.wav` o `.mp3` |
| `label` | `int` | `0` = Sano, `1` = Click, `2` = Soplo |

---

## Salida JSON

Si el procesamiento es exitoso:

```json
{
  "status": "success",
  "cycles": 7,
  "rows": [
    {
      "MFCC_1": -312.45,
      "MFCC_2": 80.12,
      "MFCC_3": -14.33,
      "...": "...",
      "MFCC_13": 3.21,
      "RMS": 0.043210,
      "label": 0
    }
  ]
}
```

Si ocurre un error:

```json
{
  "error": "No se detectaron ciclos válidos"
}
```

---

## Clase `HeartSignalProcessor`

Toda la lógica está encapsulada en esta clase. Tiene 3 métodos principales:

### `preprocess_audio(file_path)`

Carga el audio con `librosa` y lo normaliza entre -1 y 1.

```
Entrada: ruta del archivo
Salida:  x  (señal normalizada)
         fs (frecuencia de muestreo)
         t  (eje de tiempo)
```

La normalización usa `max(|x|) + 1e-12` para evitar división por cero en señales silenciosas.

---

### `compute_shannon_envelope(x, fs)`

Calcula la **envolvente de energía de Shannon**, un método clásico para detectar eventos cardíacos en señales PCG. Es más sensible que la energía cuadrática simple porque penaliza componentes de alta y baja amplitud, resaltando los sonidos cardíacos S1 y S2.

Pasos internos:

```
1. Normalizar la señal: p = |x| / max(|x|)
2. Calcular energía de Shannon: E = -p · log10(p)
3. Estandarizar E (media 0, desviación 1)
4. Normalizar al rango [0, 1]
5. Aplicar filtro paso-bajo Butterworth (fc = 9 Hz, orden 4)
   → suaviza la envolvente para facilitar detección de picos
6. Volver a normalizar al rango [0, 1]
```

---

### `detect_cycles(Env, x, t, fs)`

Detecta los **inicios de cada ciclo cardíaco** (posición del sonido S1) a partir de la envolvente calculada.

El algoritmo funciona así:

**Paso 1 — Detectar mínimos y máximos locales:**
Recorre la derivada de la envolvente buscando cambios de signo. Un cambio de negativo a positivo es un mínimo local (tipo 1), y de positivo a negativo es un máximo local (tipo 2).

**Paso 2 — Detectar triángulos:**
Busca patrones del tipo `mínimo → máximo → mínimo`, que corresponden a un pulso cardíaco en la envolvente. Cada triángulo representa un candidato a ciclo.

**Paso 3 — Filtrar por área:**
Calcula el área de cada triángulo. Descarta los que están por debajo del 60% del área máxima encontrada, eliminando ruidos pequeños y artefactos.

**Paso 4 — Validar duración fisiológica:**
Acepta solo ciclos cuya duración está entre 0.1 s y 2.0 s (equivalente a 30–600 BPM), descartando falsas detecciones.

```
Salida: lista de índices iS1 → posición de inicio de cada ciclo
```

---

### `extract_features(x, fs, iS1_idx)`

Extrae las características de cada ciclo cardíaco detectado.

Para cada ciclo entre `iS1[k]` e `iS1[k+1]`:

```
1. Extraer la señal del ciclo
2. Verificar longitud mínima (≥ 12 ms)
3. Calcular MFCC con 13 coeficientes:
   - winlen = 25 ms
   - winstep = 10 ms
   - nfft = potencia de 2 más cercana al tamaño de ventana
4. Promediar los MFCC a lo largo del tiempo → vector de 13 valores
5. Calcular RMS (energía del ciclo): √(mean(x²))
6. Concatenar: [MFCC_1..13, RMS] → vector de 14 valores
```

---

## Características extraídas

| Característica | Tipo | Descripción |
|---|---|---|
| `MFCC_1` | `float` | Coeficiente cepstral 1 — energía global del espectro |
| `MFCC_2` | `float` | Coeficiente cepstral 2 — inclinación espectral |
| `MFCC_3..13` | `float` | Coeficientes cepstrales 3 al 13 — forma espectral detallada |
| `RMS` | `float` | Energía media del ciclo cardíaco |
| `label` | `int` | Etiqueta de clase: 0=Sano, 1=Click, 2=Soplo |

Los MFCC capturan el **timbre** del sonido cardíaco: un corazón sano suena diferente a uno con soplo o click, y esa diferencia queda codificada en la forma del espectro de mel.

---

## Flujo completo resumido

```
archivo .wav/.mp3
      │
      ▼
preprocess_audio()
  → normalizar señal
      │
      ▼
compute_shannon_envelope()
  → envolvente de energía suavizada
      │
      ▼
detect_cycles()
  → índices de inicio de cada latido
      │
      ▼
extract_features()
  → por cada ciclo: [MFCC_1..13, RMS]
      │
      ▼
JSON → stdout
  → server.js lo lee y lo agrega al dataset
```

---

## Consideraciones importantes

- El script **no modifica ningún archivo**. Solo lee el audio y escribe JSON a `stdout`.
- Si cualquier paso falla (audio silencioso, no se detectan ciclos, MFCC vacíos), imprime `{"error": "..."}` y sale con código `1`.
- El número de filas en la salida es igual al número de ciclos válidos detectados. Un audio de 10 segundos puede generar entre 8 y 15 filas dependiendo de la frecuencia cardíaca.
- Cada fila es independiente: el modelo de clasificación recibe una fila a la vez.