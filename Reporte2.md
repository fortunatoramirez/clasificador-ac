# Reporte: mejora del clasificador de fonocardiograma (PCG)

**Proyecto de Servicio Social — clasificador-ac**

## 1. Objetivo

Evaluar y mejorar el clasificador de sonidos cardiacos existente (`modelo_pcg_final.pkl`), que distingue entre grabaciones **Sano**, **Click** y **Soplo** a partir de audio capturado con estetoscopio.

## 2. Diagnóstico del sistema original

Se revisó el pipeline de producción (`arboldeprediccion.py`, `extract_features.py`) y el dataset de entrenamiento (`dataset.xlsx`, 402 filas). Se encontraron dos problemas antes de intentar cualquier mejora:

- **Sin identificador de origen**: el dataset no tenía ninguna columna que indicara de qué archivo/paciente salía cada fila (cada fila representa un ciclo cardiaco individual, no una grabación completa). Sin esto, es imposible garantizar que la validación del modelo no esté contaminada por ciclos del mismo archivo repartidos entre entrenamiento y prueba.
- **Archivos de audio duplicados**: al agregar el identificador y analizar los 46 audios fuente (biblioteca de la Universidad de Michigan), se descubrió que **44 de los 46 archivos eran copias exactas de solo 9 grabaciones distintas** (confirmado por features idénticas a 6 decimales entre archivos con nombres diferentes). Esto explica por qué evaluaciones previas podían parecer artificialmente perfectas.

## 3. Búsqueda de datos adicionales

Dado que la biblioteca de Michigan es un recurso de enseñanza con muy pocas grabaciones únicas (~23 en total), se identificó y utilizó el **CirCor DigiScope Phonocardiogram Dataset** (PhysioNet, usado en el Challenge 2022 de George B. Moody): 5,282 grabaciones clínicas reales de 1,568 pacientes, con etiqueta de soplo (presente/ausente) verificada por cardiólogos.

- Se procesaron ~3,007 archivos con el pipeline de extracción existente.
- Se aplicó un filtro de calidad (mínimo 5 ciclos detectados por grabación), quedando **1,079 archivos utilizables de 586 pacientes** (459 Sano, 127 Soplo).
- **Nota importante:** ninguna fuente pública consultada (CirCor, PhysioNet 2016, la propia biblioteca de Michigan) tiene la clase "Click" etiquetada en volumen — sigue existiendo solo 1 grabación única de esa clase. Se recomienda tratarla como categoría secundaria hasta conseguir más datos, y discutir el alcance con el asesor de servicio social.

## 4. Metodología de modelado

Se reconstruyó el pipeline paso a paso (preprocesamiento → segmentación → extracción de features → modelado → validación), usando el mismo procesamiento de señal ya existente en el proyecto (envolvente de energía de Shannon + filtro pasa-bajos, MFCCs con `python_speech_features`).

**Punto central de la validación:** todas las evaluaciones se hicieron con `StratifiedGroupKFold`, agrupando por paciente — garantizando que ningún paciente apareciera simultáneamente en entrenamiento y prueba (fuga de datos), confirmado explícitamente en cada corrida (0 pacientes repetidos entre lados).

### Comparación de algoritmos (5-fold, agrupado por paciente)

| Modelo | Exactitud |
|---|---|
| Regresión Logística (modelo actual) | 67.3% |
| SVM | 70.8% |
| **Random Forest** | **79.1%** |
| Gradient Boosting | 79.1% |

Se eligió **Random Forest** por su desempeño y por ser más robusto que Regresión Logística ante relaciones no lineales entre features — relevante dado que la segmentación real es imperfecta (ver sección 6).

### Selección del umbral de decisión

En vez de usar el umbral por default (0.5), se exploró el trade-off completo entre sensibilidad y especificidad, y se seleccionó el punto óptimo con el **índice de Youden** (J = sensibilidad + especificidad − 1), tanto a nivel de ciclo individual como a nivel de grabación completa (votación por mayoría entre los ciclos de un mismo audio, igual que hace el sistema en producción).

### Validación honesta (dev/test)

Para evitar incluso el sesgo sutil de elegir el umbral óptimo mirando los mismos datos que luego se reportan como resultado, se separó un **20% de los pacientes (117) que nunca se usaron ni para entrenar ni para elegir el umbral**, exclusivamente como examen final.

**Resultado del examen final**, con pacientes nunca vistos por el modelo:
- Sensibilidad: **73.5%**
- Especificidad: **69.9%**
- Umbral de decisión final: **0.15**

Este resultado es consistente con la validación cruzada, confirmando que el modelo generaliza como se esperaba y no hay sobreajuste oculto.

## 5. Integración a producción

Se reemplazó el modelo de PyCaret (3 clases) por el Random Forest de sklearn (2 clases: Sano/Soplo) dentro de `arboldeprediccion.py`, manteniendo toda la lógica existente de segmentación, votación por mayoría y formato de respuesta para el frontend. El cambio se probó ejecutando el script real (no solo en notebook) contra 30 pacientes de CirCor con etiqueta conocida, confirmando resultados consistentes con la validación formal.

Archivos generados:
- `modelo_pcg_soplo_rf.joblib` — modelo entrenado con el 100% de los 586 pacientes.
- `modelo_pcg_soplo_rf_metadata.json` — columnas de features, umbral de decisión, y métricas del examen final documentadas.

## 6. Limitaciones conocidas

- **Clase "Click" sin datos suficientes** para entrenar o validar (1 grabación única disponible). No se intentó predecirla en el modelo nuevo.
- **La segmentación (`detect_cycles`) pierde muchos ciclos en audio real y ruidoso.** En grabaciones limpias de enseñanza detecta 70-100 ciclos; en grabaciones clínicas reales de CirCor, la mediana fue de solo 3 ciclos por archivo, y se observaron casos con 1 solo ciclo detectado en grabaciones de 18-26 segundos (donde deberían detectarse 15-25). Esto probablemente le pone un techo al desempeño del modelo, y hace que el BPM y el "porcentaje de confianza" reportados no sean confiables cuando se detectan muy pocos ciclos.
- El modelo se entrenó y validó exclusivamente con CirCor (grabaciones clínicas reales con ruido de fondo real) — no se mezcló con la biblioteca de Michigan (grabaciones de enseñanza, limpias) para evitar que el modelo aprenda a distinguir la *fuente* de la grabación en vez de la condición médica.

## 7. Investigación de segmentación mejorada (Opción 2)

Siguiendo la recomendación #1 del análisis inicial, se investigó a fondo si una segmentación más robusta mejoraba el desempeño real del clasificador.

### 7.1 Diagnóstico del problema original

Se inspeccionó visualmente la envolvente de grabaciones reales con pocos ciclos detectados (ej. un archivo de 18s que solo detectaba 1 ciclo, cuando lo esperable eran 15-25). Se encontraron tres causas concretas en `detect_cycles`:

- Un error de conteo que saltaba índices incorrectamente al procesar grupos de picos, descartando ciclos válidos.
- Una regla de clasificación S1/S2 basada en amplitud, que asignaba mal las etiquetas cuando un soplo fuerte alteraba el orden de amplitudes (la solución correcta es clasificar por **orden cronológico**: el primer pico de un grupo siempre es S1 y el último siempre es S2, sin importar cuál suene más fuerte).
- Una normalización basada en percentil global, que en grabaciones con silencios largos confundía silencio con señal real (el "techo" de referencia salía calculado casi enteramente sobre ruido de fondo, no sobre los picos verdaderos).

También se comprobó, con la transformada de Fourier (método de Welch), que el pipeline original nunca aplicaba ningún filtro de ruido antes de calcular la envolvente — se agregó un filtro pasa-banda (20-400 Hz) como paso de preprocesamiento nuevo.

### 7.2 Segmentación rediseñada

Se reconstruyó la segmentación con: filtro pasa-banda previo, envolvente vía transformada de Hilbert, detección de picos por **prominencia relativa a la desviación estándar** de la envolvente (en vez de un umbral fijo o percentil global), e identificación de artefactos comparando cada pico contra la **mediana** de todos los picos detectados (más robusto que un percentil global). Tras varias iteraciones de prueba y corrección de bugs (incluyendo un `IndexError` por un error de índice al calcular límites de ciclo), el nuevo pipeline se validó primero en casos de prueba controlados antes de aplicarse al dataset completo.

**Resultado en volumen de datos:** el nuevo dataset (`dataset_circor_v2.xlsx`) generó 89,238 filas de 873 pacientes (mediana de 28 ciclos por grabación), contra 11,153 filas de 586 pacientes con la segmentación original (mediana de ~3 ciclos por grabación) — una mejora sustancial en cantidad de señal aprovechada.

### 7.3 Comparación en validación cruzada (dev)

| Modelo | Segmentación original | Segmentación mejorada |
|---|---|---|
| Regresión Logística | 67.3% | 69.5% |
| SVM | 70.8% | 73.8% |
| **Random Forest** | **79.1%** | **84.1%** |
| Gradient Boosting | 79.1% | 84.3% |

Los 4 modelos mejoraron de forma consistente, lo que en un primer momento sugirió una mejora real y generalizable.

### 7.4 Resultado del examen final honesto (hallazgo clave)

Usando exactamente los mismos 117 pacientes apartados como prueba final (nunca vistos durante entrenamiento ni selección de umbral), el resultado fue distinto al que la validación cruzada predecía:

| Métrica | Segmentación original | Segmentación mejorada |
|---|---|---|
| Sensibilidad | **73.5%** | 58.8% |
| Especificidad | 69.9% | **89.2%** |
| ROC-AUC | 0.706 | 0.698 |
| Índice de Youden (J) | 0.434 | 0.480 |

El AUC —que mide la capacidad de separación del modelo independientemente del umbral elegido— resultó **prácticamente idéntico** entre ambas versiones (0.706 vs 0.698). Comparando ambos modelos en el mismo punto de especificidad (0.699), el modelo con la segmentación **original** tiene mejor sensibilidad (0.735 vs 0.676). Esto indica que la mejora observada en la validación cruzada probablemente reflejaba, en parte, un ajuste fino de los parámetros de segmentación (el umbral de prominencia, el factor de detección de artefactos) sobre las mismas grabaciones de prueba usadas para calibrarlos durante el desarrollo — una forma sutil de sobreajuste al proceso de diseño, no a los datos de entrenamiento en sí.

### 7.5 Decisión

**Se optó por mantener el modelo original (`modelo_pcg_soplo_rf.joblib`, segmentación original) en producción**, por tener mejor sensibilidad — la métrica más importante en un contexto de tamizaje, donde es preferible una falsa alarma a un caso real no detectado. La segmentación mejorada no se descarta como aprendizaje: quedó documentado el código corregido y las lecciones sobre normalización robusta (mediana vs. percentil, prominencia vs. umbral fijo), útiles para trabajo futuro. Se consideró combinar ambos modelos (ensemble) pero no se llevó a cabo, dado que un AUC casi idéntico entre ambos sugiere que es poco probable que la combinación aporte una mejora sustancial.

## 8. Recomendaciones actualizadas para siguientes pasos

1. **Agregar una alerta de "confianza baja"** cuando se detecten muy pocos ciclos, en vez de reportar 100% de confianza de forma engañosa (sigue pendiente, y es de implementación rápida).
2. **Buscar o generar más ejemplos de la clase Click** antes de intentar reincorporarla — ninguna fuente pública consultada la tiene etiquetada en volumen.
3. **Usar la biblioteca de Michigan como conjunto de prueba adicional** ("¿el modelo sigue funcionando bien en audio limpio?"), no como datos de entrenamiento.
4. Si se retoma el trabajo de segmentación en el futuro, **calibrar los parámetros de detección (prominencia, umbral de artefactos) usando el conjunto `dev`, nunca mirando directamente las grabaciones de prueba final** — evitar repetir el sesgo sutil identificado en la sección 7.4, por ejemplo separando un tercer subconjunto exclusivo para ajuste visual de parámetros, distinto del usado para el examen final.
5. Explorar si un **umbral de decisión distinto por fold** (en vez de uno solo elegido sobre todo `dev`) generaliza mejor a pacientes nuevos — el examen final mostró que el umbral óptimo puede variar entre grupos de pacientes.
