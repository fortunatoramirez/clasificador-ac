# CardioAC — Clasificador de Sonidos Cardíacos

Sistema de análisis y clasificación de fonocardiogramas (PCG) mediante inteligencia artificial. Permite subir grabaciones de audio cardíaco y obtener un diagnóstico automático (Sano, Click o Soplo) junto con visualización del pipeline de procesamiento de señal.

## Tecnologías

- **Backend:** Node.js + Express
- **Frontend:** HTML, CSS, JavaScript vanilla
- **ML:** Python 3.11 + PyCaret (Regresión Logística)
- **Base de datos:** MySQL

---

## Requisitos previos

Antes de clonar el proyecto, asegúrate de tener instalado:

- [Node.js](https://nodejs.org/) v18 o superior
- [Python 3.11.9](https://www.python.org/downloads/release/python-3119/) — **importante: no marcar "Add to PATH" si ya tienes otra versión instalada**
- [XAMPP](https://www.apachefriends.org/) o cualquier servidor MySQL

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd clasificador-ac
```

### 2. Crear el entorno virtual de Python

Desde la **raíz del proyecto**:

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias de Python

Con el `.venv` activado:

```bash
pip install -r requirements.txt
pip install python_speech_features
```

### 4. Instalar dependencias de Node.js

```bash
cd web/backend
npm install
```

### 5. Configurar variables de entorno

Copia el archivo de ejemplo y edítalo con tus datos:

```bash
cp web/backend/.env.example web/backend/.env
```

Abre `.env` y completa:

```env
PORT=5001
DB_HOST=localhost
DB_USER=tu_usuario_mysql
DB_PASSWORD=tu_contraseña
DB_NAME=hospital_db
SESSION_SECRET=cambia_esto_por_algo_seguro
```

### 6. Crear la base de datos

Abre phpMyAdmin o tu cliente MySQL e importa el archivo:

```
hospital_db.sql
```

---

## Ejecutar el proyecto

Cada vez que vayas a trabajar, activa el `.venv` y luego levanta el servidor:

```bash
# Desde la raíz del proyecto — activar venv
# Windows:
.venv\Scripts\activate

# Luego ir a backend y correr el servidor
cd web/backend
node server.js
```

Abre el navegador en: **http://localhost:5001**

---

## Estructura del proyecto

```
clasificador-ac/
├── .venv/                          # Entorno virtual Python (no se sube a git)
├── models/
│   └── modelo_pcg_final.pkl        # Modelo entrenado (Regresión Logística)
├── pcg_processing/
│   ├── classification/
│   │   └── arboldeprediccion.py    # Pipeline principal: procesa audio y clasifica
│   ├── preprocessing/
│   │   └── extract_features.py     # Extrae features para reentrenamiento
│   └── training/
│       ├── train.py                # Entrena un nuevo modelo desde cero
│       └── retrain_eval.py         # Evalúa modelo candidato con cross-validation
├── web/
│   ├── backend/
│   │   ├── server.js               # Servidor Express
│   │   ├── .env                    # Variables de entorno (no se sube a git)
│   │   ├── .env.example            # Plantilla de variables de entorno
│   │   └── uploads/                # Audios subidos (no se sube a git)
│   └── frontend/
│       ├── index.html              # Página principal — subir audio
│       ├── pacientes.html          # Historial de diagnósticos
│       ├── detalle.html            # Detalle de un diagnóstico
│       ├── pcg-dashboard.html      # Visualización del pipeline de señal
│       ├── login.html              # Inicio de sesión
│       ├── registro.html           # Registro de usuarios
│       └── style.css               # Estilos compartidos
├── dataset.xlsx                    # Dataset de entrenamiento
├── hospital_db.sql                 # Esquema de la base de datos
├── requirements.txt                # Dependencias de Python
└── package.json                    # Dependencias de Node.js
```

---

## Tipos de usuario

El sistema tiene tres roles:

| Rol | Acceso |
|-----|--------|
| `paciente` | Solo puede ver sus propios diagnósticos |
| `medico` | Puede ver el historial de todos los pacientes |
| `admin` | Acceso completo + panel de reentrenamiento del modelo |

Los códigos de acceso para registro se configuran en la tabla `codigos_acceso` de la base de datos.

---
