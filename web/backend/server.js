const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const mysql = require('mysql2');
const session = require('express-session');
const bcrypt = require('bcrypt');
const bodyParser = require('body-parser');
const XLSX   = require('xlsx');
const app = express();
const PORT = 5001;



app.use(express.urlencoded({ extended: true }));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

// Configuración de la sesión
app.use(session({
  secret: 'secretKey',
  resave: false,
  saveUninitialized: false,
  cookie: { secure: false } // en producción con https poner true
}));


// --- 1. CONFIGURACIÓN BASE DE DATOS ---
const db = mysql.createPool({
    host: 'localhost',
    user: 'admin',      // Asegúrate que este usuario existe en MySQL
    password: 'Galgos2026!',
    database: 'hospital_db',
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
});


db.getConnection((err, connection) => {
    if (err) {
        console.error('❌ Error fatal: No se pudo conectar a MySQL.');
        console.error('   Causa:', err.code);
        console.error('   Verifica que XAMPP/MySQL esté encendido.');
    } else {
        console.log('✅ Conectado a MySQL (hospital_db)');
        connection.release();
    }
});

//Middleware
function requireLogin(req, res, next) {
  if (!req.session.userId) {
    return res.redirect('/login.html');
  }
  next();
}

function requireRole(roles) {
  return (req, res, next) => {
      if (req.session.userId && roles.includes(req.session.userId.tipo_usuario)) {
          next();
      } else {
          res.status(403).send('Acceso denegado');
      }
  };
}

// Ruta para obtener el tipo de usuario actual
app.get('/tipo-usuario', requireLogin, (req, res) => {
  res.json({ tipo_usuario: req.session.userId.tipo_usuario });
});

// Ruta protegida (Página principal después de iniciar sesión)
app.get('/', requireLogin, (_req, res) => {
  res.sendFile(__dirname + '../frontend/index.html');
});

// Servir archivos estáticos (HTML)
app.use(express.static(path.join(__dirname, '../frontend')));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

app.post('/registrar', (req, res) => {
    const { nombre_usuario, password, codigo_acceso, nombre, apellido, email, fecha_nacimiento } = req.body;

    // 1. Validar datos básicos
    if (!nombre_usuario || !password || !codigo_acceso) {
        return res.status(400).send('Faltan datos obligatorios (Usuario, Contraseña o Código).');
    }

    // 2. Verificar Código de Acceso
    // USO DE 'db' EN LUGAR DE 'connection'
    const sqlCodigo = 'SELECT tipo_usuario FROM codigos_acceso WHERE codigo = ?';
    db.query(sqlCodigo, [codigo_acceso], (err, results) => {
        if (err) { console.error(err); return res.status(500).send('Error verificando código.'); }
        if (results.length === 0) return res.status(400).send('Código de acceso inválido.');

        const tipo_usuario = results[0].tipo_usuario;

        // 3. Verificar si el usuario ya existe
        const sqlExisteUser = 'SELECT 1 FROM usuarios WHERE nombre_usuario = ?';
        db.query(sqlExisteUser, [nombre_usuario], (err, existingUser) => {
            if (err) return res.status(500).send('Error verificando usuario.');
            if (existingUser.length > 0) return res.status(409).send('El nombre de usuario ya existe.');

            // 4. Encriptar contraseña
            bcrypt.hash(password, 10, (err, hashedPassword) => {
                if (err) return res.status(500).send('Error encriptando contraseña.');

                // 5. Insertar Usuario
                const sqlInsertUser = 'INSERT INTO usuarios (nombre_usuario, password_hash, tipo_usuario) VALUES (?, ?, ?)';
                db.query(sqlInsertUser, [nombre_usuario, hashedPassword, tipo_usuario], (err, resultUser) => {
                    if (err) { console.error(err); return res.status(500).send('Error creando usuario.'); }

                    const id_usuario = resultUser.insertId; // ID del usuario recién creado

                    // 6. Si es PACIENTE, guardar datos personales
                    if (tipo_usuario === 'paciente') {
                        if (!nombre || !apellido || !email || !fecha_nacimiento) {
                            return res.status(400).send('Faltan datos personales del paciente.');
                        }

                        const sqlInsertPaciente = 'INSERT INTO pacientes (id_usuario, nombre, apellido, email, fecha_nacimiento) VALUES (?, ?, ?, ?, ?)';
                        db.query(sqlInsertPaciente, [id_usuario, nombre, apellido, email, fecha_nacimiento], (err, resultPaciente) => {
                            if (err) { console.error(err); return res.status(500).send('Error guardando datos del paciente.'); }
                            
                            // Redirigir al login tras éxito
                            res.redirect('/login.html');
                        });

                    } else {
                        // Si es MEDICO o ADMIN, terminamos aquí
                        res.redirect('/login.html');
                    }
                });
            });
        });
    });
});





// Iniciar sesión
app.post('/login.html', (req, res) => {
    const { nombre_usuario, password } = req.body;

    // USO DE 'db' EN LUGAR DE 'connection'
    const sqlLogin = `
        SELECT u.*, p.id as id_paciente 
        FROM usuarios u 
        LEFT JOIN pacientes p ON u.id = p.id_usuario 
        WHERE u.nombre_usuario = ?`;

    db.query(sqlLogin, [nombre_usuario], async (err, results) => {
        if (err) { console.error(err); return res.status(500).send('Error en el servidor'); }
        
        if (results.length === 0) {
            return res.send('<h3 style="color:red">Usuario no encontrado</h3><a href="/login.html">Volver</a>');
        }

        const user = results[0];

        // Comparar contraseñas
        const match = await bcrypt.compare(password, user.password_hash);
        if (!match) {
            return res.send('<h3 style="color:red">Contraseña incorrecta</h3><a href="/login.html">Volver</a>');
        }

        // Guardar sesión
        req.session.userId = {
            id: user.id, // ID de la tabla usuarios
            nombre_usuario: user.nombre_usuario,
            tipo_usuario: user.tipo_usuario,
            paciente_id: user.id_paciente || null // ID de la tabla pacientes (si existe)
        };

        // Guardar la sesión explícitamente antes de redirigir (buena práctica)
        req.session.save(() => {
            res.redirect('/');
        });
    });
});

// Cerrar sesión
app.get('/logout', (req, res) => {
  req.session.destroy();
  res.redirect('/login.html');
});

app.get('/session-data', (req, res) => {
  if (!req.session || !req.session.userId) {
    return res.status(401).json({ error: 'No autenticado' });
  }

  res.json({
    nombre_usuario: req.session.userId.nombre_usuario,
    pacienteId: req.session.userId.paciente_id || null
  });
});

// Ruta para que solo admin pueda ver todos los usuarios
app.get('/ver-usuarios', requireLogin, requireRole('admin'), (_req, res) => {
  db.query('SELECT * FROM usuarios', (err, results) => {
    if (err) {
      return res.send('Error al obtener los datos.');
    }

  let html = `
    <html>
    <head>
      <link rel="stylesheet" href="/styles.css">
      <title>Usuarios</title>
    </head>
    <body>
      <h1>Usuarios Registrados</h1>
      <table>
        <thead>
          <tr>
            <th>id</th>
            <th>Nombre</th>
            <th>Contraseña Encriptada</th>
            <th>Tipo de usuario</th>
          </tr>
        </thead>
        <tbody>
  `;

  results.forEach(usuario => {
    html += `
      <tr>
        <td>${usuario.id}</td>
        <td>${usuario.nombre_usuario}</td>
        <td>${usuario.password_hash}</td>
        <td>${usuario.tipo_usuario}</td>
      </tr>
    `;
  });

  html += `
        </tbody>
      </table>
      <button onclick="window.location.href='/'">Volver</button>
    </body>
    </html>
  `;

  res.send(html);
});
});



// --- 2. MULTER ---
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const dir = path.join(__dirname, 'uploads'); // Ruta absoluta
        if (!fs.existsSync(dir)) fs.mkdirSync(dir);
        cb(null, dir);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + '-' + file.originalname);
    }
});

const upload = multer({ 
    storage: storage,
    fileFilter : (req, file, cb) => {
    // Tipos MIME permitidos
    const allowedMimes = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/wave'];
    const ext = path.extname(file.originalname).toLowerCase();

    if (allowedMimes.includes(file.mimetype) || ext === '.mp3' || ext === '.wav') {
        cb(null, true);
    } else {
        cb(new Error('Formato no válido. Solo se permiten archivos MP3 y WAV.'), false);
    }
    },
    limits: { fileSize: 10 * 1024 * 1024 } 
});



// --- RUTAS ---
app.get('/', (req, res) => res.sendFile(path.join(__dirname, '../frontend', 'index.html')));
app.get('/ver-pacientes', requireRole(['medico']), (req, res) => res.sendFile(path.join(__dirname, '../frontend', 'pacientes.html')));

app.get('/api/pacientes', (req, res) => {
    const query = `
        SELECT 
            d.id,              /* <--- ESTO ES LO IMPORTANTE: El ID del diagnóstico */
            p.nombre, 
            p.edad, 
            d.resultado_ia, 
            d.confianza, 
            d.fecha_analisis 
        FROM diagnosticos d
        INNER JOIN pacientes p ON d.paciente_id = p.id
        ORDER BY d.fecha_analisis DESC
    `;
    
    db.query(query, (err, results) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json(results);
    });
});

// --- 4. RUTA CRÍTICA: UPLOAD & CLASSIFY ---
app.post('/upload', upload.single('cancion'), (req, res) => {
    if (!req.file) return res.status(400).send('Falta archivo MP3.');

    const nombre = req.body.nombre || "Anónimo";
    const edad = req.body.edad || 0;
    const filePath = req.file.path; // Multer ya da la ruta completa
    const scriptPath = path.join(__dirname, 'arboldeprediccion.py');

    // --- CORRECCIÓN DE LA RUTA DE PYTHON ---
    // Intentamos buscar el venv, si no existe, usamos el python global del sistema
    let pythonExecutable;
    const venvPathWin = path.join(__dirname, 'venv', 'Scripts', 'python.exe');
    const venvPathLinux = path.join(__dirname, 'venv', 'bin', 'python');

    if (process.platform === 'win32' && fs.existsSync(venvPathWin)) {
        pythonExecutable = venvPathWin;
    } else if (process.platform !== 'win32' && fs.existsSync(venvPathLinux)) {
        pythonExecutable = venvPathLinux;
    } else {
        // FALLBACK: Si no encuentra la carpeta venv, usa el comando global
        console.warn("⚠️ Advertencia: No se encontró carpeta 'venv'. Usando Python global.");
        pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    }

    console.log(`--- PROCESANDO ---`);
    console.log(`Script: ${scriptPath}`);
    console.log(`Python: ${pythonExecutable}`);

    const pythonProcess = spawn(pythonExecutable, [scriptPath, filePath]);

    let outputData = '';
    let errorData = '';

    // Error al INICIAR el proceso (ej. ruta de python mal)
    pythonProcess.on('error', (err) => {
        console.error("❌ ERROR AL SPAWN PYTHON:", err);
        errorData = `No se pudo iniciar Python. Verifica la ruta o instalación.\n${err.message}`;
    });

    pythonProcess.stdout.on('data', (data) => outputData += data.toString());
    pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString();
        console.error(`PyLog: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python terminó con código: ${code}`);

        // CASO 1: Error crítico de Python (Crash o Spawn fallido)
        if (code !== 0 || errorData.includes("Traceback") || !outputData) {
            return res.status(500).send(`
                <div style="font-family: monospace; background: #ffe6e6; padding: 20px;">
                    <h2 style="color: red;">Error Técnico</h2>
                    <p>El análisis falló.</p>
                    <strong>Detalles:</strong>
                    <pre>${errorData || "El script no devolvió datos (posible error de librería faltante)."}</pre>
                    <a href="/">Volver</a>
                </div>
            `);
        }

        // CASO 2: Intentar leer JSON
        let result = null;
        try {
            result = JSON.parse(outputData.trim());
        } catch (e) {
            console.error("JSON Parse Error:", outputData);
            return res.status(500).send("Error interno: Respuesta de IA inválida.");
        }

        if (result.status === 'error') {
            return res.status(500).send(`<h3>Error IA:</h3> <p>${result.message}</p><a href="/">Volver</a>`);
        }

        // CASO 3: Éxito -> Guardar en BD
        const sqlPaciente = "INSERT INTO pacientes (nombre, edad) VALUES (?, ?)";
        db.query(sqlPaciente, [nombre, edad], (err, resP) => {
            if (err) {
                console.error("DB Error Paciente:", err);
                return res.status(500).send("Error guardando en base de datos.");
            }

            const pid = resP.insertId;
            const sqlDiag = `INSERT INTO diagnosticos (paciente_id, ruta_audio, resultado_ia, confianza, ciclos_detectados) VALUES (?, ?, ?, ?, ?)`;
            
            db.query(sqlDiag, [pid, req.file.path, result.class, result.confidence, result.cycles], (err, resD) => {
                if (err) console.error("DB Error Diagnóstico:", err);

                // HTML RESPUESTA
                const isNormal = result.class.toLowerCase() === 'normal';
                const color = isNormal ? '#28a745' : '#dc3545';
                const icon = isNormal ? '💚' : '⚠️';

                res.send(`
                    <!DOCTYPE html>
                    <html lang="es">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Resultados</title>
                        <link rel="stylesheet" href="styles.css">
                        <style>
                            body { text-align: center; padding: 40px; }
                            .stats { background: #f8f9fa; padding: 15px; border-left: 5px solid ${color}; text-align: left; margin: 20px auto; max-width: 400px; }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                        <div class="card" style="max-width: 600px; margin: 0 auto;">
                            <h1>${icon} ${isNormal ? 'NORMAL' : 'ANOMALÍA'}</h1>
                            <p><strong>Paciente:</strong> ${nombre} (${edad} años)</p>
                            <div class="stats">
                                <p>🔍 Diagnóstico: <strong>${result.class}</strong></p>
                                <p>📊 Confianza: ${result.confidence}%</p>
                                <p>💓 Latidos: ${result.cycles}</p>
                            </div>
                            <small style="color:green">💾 Guardado ID #${pid}</small>
                            <br><br>
                            <a href="/" class="btn-upload" style="text-decoration:none">Nuevo Análisis</a>
                            <br><br>
                            <a href="/ver-pacientes" style="color:#666">Ver Historial</a>
                        </div>
                        </div>
                    </body>
                    </html>
                `);
            });
        });
    });
});

// 2. RUTA PARA SERVIR LA VISTA DE DETALLE
app.get('/ver-detalle', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend', 'detalle.html'));
});

// 3. API PARA OBTENER UN DIAGNÓSTICO ESPECÍFICO POR ID
app.get('/api/diagnostico/:id', (req, res) => {
    const id = req.params.id;
    
    const query = `
        SELECT d.*, p.nombre, p.edad 
        FROM diagnosticos d
        JOIN pacientes p ON d.paciente_id = p.id
        WHERE d.id = ?
    `;

    db.query(query, [id], (err, results) => {
        if (err) return res.status(500).json({ error: err.message });
        if (results.length === 0) return res.status(404).json({ error: 'No encontrado' });

        const diag = results[0];

        // --- MEJORA DE ROBUSTEZ ---
        // 1. Extraemos el nombre del archivo de forma segura (funciona en Windows y Linux)
        // Esto toma lo que está después de la última barra '/' o '\'
        const filename = diag.ruta_audio.split(/[\\/]/).pop();
        
        // 2. Construimos la URL pública
        diag.url_web = `/uploads/${filename}`;

        // 3. (Opcional) Le decimos al frontend qué tipo de archivo es
        diag.tipo_archivo = filename.endsWith('.wav') ? 'WAV' : 'MP3';

        res.json(diag);
    });
});

// ... (Tus códigos anteriores) ...

// ====================================================
// ZONA ADMINISTRADOR: ENTRENAMIENTO
// ====================================================

// 1. Configuración Multer especial para la carpeta DATA
const storageData = multer.diskStorage({
    destination: (req, file, cb) => {
        // Asegurar que la carpeta data existe
        const dir = path.join(__dirname, 'data');
        if (!fs.existsSync(dir)) fs.mkdirSync(dir);
        cb(null, dir);
    },
    filename: (req, file, cb) => {
        // TRUCO: Renombramos el archivo con la etiqueta al principio
        // Ejemplo: "normal_17623123_audio.mp3"
        const etiqueta = req.body.etiqueta || "unknown"; 
        
        // Usamos un número aleatorio extra para evitar colisiones si se suben varios al mismo tiempo
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        const ext = path.extname(file.originalname).toLowerCase();
        
        const nuevoNombre = `${etiqueta}_${uniqueSuffix}${ext}`;
        cb(null, nuevoNombre);
    }
});

// Filtro para aceptar MP3 y WAV
const uploadData = multer({ 
    storage: storageData,
    fileFilter: (req, file, cb) => {
        const allowed = ['.mp3', '.wav'];
        const ext = path.extname(file.originalname).toLowerCase();
        if (allowed.includes(ext)) {
            cb(null, true);
        } else {
            cb(new Error('Solo archivos .mp3 y .wav permitidos'), false);
        }
    }
});

// 2. Ruta para subir MÚLTIPLES archivos de entrenamiento (Solo Admin)
// CAMBIO CLAVE: .array('audio_entrenamiento') en lugar de .single()
app.post('/api/admin/upload-data', requireLogin, requireRole(['admin']), uploadData.array('audio_entrenamiento'), (req, res) => {
    
    if (!req.files || req.files.length === 0) {
        return res.status(400).json({ error: 'No se subieron archivos.' });
    }

    const count = req.files.length;
    console.log(`📂 Admin subió ${count} archivos con etiqueta: ${req.body.etiqueta}`);

    res.json({ 
        success: true, 
        message: `Se guardaron ${count} archivos correctamente para entrenamiento.` 
    });
});


// 3. Ruta para EJECUTAR EL REENTRENAMIENTO (Solo Admin)
app.post('/api/admin/retrain', requireLogin, requireRole(['admin']), (req, res) => {
    
    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Transfer-Encoding', 'chunked');

    const scriptPath = path.join(__dirname, 'train.py');
    
    let pythonExecutable;
    const venvPathWin = path.join(__dirname, 'venv', 'Scripts', 'python.exe');
    const venvPathLinux = path.join(__dirname, 'venv', 'bin', 'python');

    if (process.platform === 'win32' && fs.existsSync(venvPathWin)) {
        pythonExecutable = venvPathWin;
    } else if (process.platform !== 'win32' && fs.existsSync(venvPathLinux)) {
        pythonExecutable = venvPathLinux;
    } else {
        pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    }

    res.write(`🚀 Iniciando motor de IA...\n`);
    res.write(`📂 Ejecutando: ${scriptPath}\n`);

    const pythonProcess = spawn(pythonExecutable, [scriptPath]);

    pythonProcess.stdout.on('data', (data) => {
        res.write(data.toString());
    });

    pythonProcess.stderr.on('data', (data) => {
        res.write(`[LOG]: ${data.toString()}`);
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            res.write(`\n✅ ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!\n`);
            res.write(`El nuevo modelo 'heart_sound_model.pkl' ya está activo.`);
        } else {
            res.write(`\n❌ ERROR FATAL: El proceso terminó con código ${code}`);
        }
        res.end(); 
    });
});

// ============================================================
//  RUTAS DE REENTRENAMIENTO — agregar a server.js
//  Dependencias npm adicionales: multer, xlsx
//  npm install multer xlsx
// ============================================================

// ── Rutas de archivos clave ──────────────────────────────────
const DATASET_PATH       = path.join(__dirname, 'dataset.xlsx');
const DATASET_TEMP_PATH  = path.join(__dirname, 'dataset_candidato.xlsx');
const MODEL_PROD_PATH    = path.join(__dirname, 'modelo_pcg_final');        // sin .pkl
const MODEL_CAND_PATH    = path.join(__dirname, 'modelo_pcg_candidato');    // sin .pkl

// ── Python executable (reutiliza tu lógica existente) ────────
function getPythonExe() {
  const win   = path.join(__dirname, 'venv', 'Scripts', 'python.exe');
  const linux = path.join(__dirname, 'venv', 'bin', 'python');
  if (process.platform === 'win32' && fs.existsSync(win))   return win;
  if (process.platform !== 'win32' && fs.existsSync(linux)) return linux;
  return process.platform === 'win32' ? 'python' : 'python3';
}

// ── Multer: audios temporales ─────────────────────────────────
const audioStorage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, path.join(__dirname, 'uploads_tmp')),
  filename:    (req, file, cb) => cb(null, `tmp_${Date.now()}_${file.originalname}`)
});
const uploadAudio = multer({
  storage: audioStorage,
  fileFilter: (req, file, cb) => {
    const ok = /\.(wav|mp3|ogg|flac)$/i.test(file.originalname);
    cb(null, ok);
  }
});

// Crear carpeta temporal si no existe
const tmpDir = path.join(__dirname, 'uploads_tmp');
if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir);


// ============================================================
//  1. GET /api/admin/dataset/info
//     Devuelve estadísticas del dataset actual
// ============================================================
app.get('/api/admin/dataset/info', requireLogin, requireRole(['admin']), (req, res) => {
  try {
    if (!fs.existsSync(DATASET_PATH)) {
      return res.json({ exists: false, rows: 0, class_counts: {} });
    }
    const wb  = XLSX.readFile(DATASET_PATH);
    const ws  = wb.Sheets[wb.SheetNames[0]];
    const data = XLSX.utils.sheet_to_json(ws);

    const counts = { sano: 0, click: 0, soplo: 0 };
    data.forEach(row => {
    const l = parseInt(row.Etiqueta);
    if (l === 0)      counts.sano++;
    else if (l === 1) counts.click++;
    else if (l === 2) counts.soplo++;
    });

    res.json({ exists: true, rows: data.length, class_counts: counts });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});


// ============================================================
//  2. POST /api/admin/dataset/add
//     Sube uno o más audios, extrae features y los agrega
//     al dataset TEMPORAL (no toca el de producción todavía)
//     Body: multipart — files[]: audios, label: 0|1|2
// ============================================================
app.post('/api/admin/dataset/add',
  requireLogin, requireRole(['admin']),
  uploadAudio.array('files'),
  async (req, res) => {

  const label = parseInt(req.body.label);
  if (![0, 1, 2].includes(label)) {
    return res.status(400).json({ error: 'Label inválido. Debe ser 0, 1 o 2.' });
  }
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ error: 'No se recibieron archivos.' });
  }

  const python      = getPythonExe();
  const extractScript = path.join(__dirname, 'extract_features.py');
  const results     = [];

  // Cargar dataset base (producción o candidato si ya existe)
  const baseDataset = fs.existsSync(DATASET_TEMP_PATH) ? DATASET_TEMP_PATH : DATASET_PATH;
  let existingRows  = [];

  if (fs.existsSync(baseDataset)) {
    const wb   = XLSX.readFile(baseDataset);
    const ws   = wb.Sheets[wb.SheetNames[0]];
    existingRows = XLSX.utils.sheet_to_json(ws);
  }

  let totalAdded = 0;

for (const file of req.files) {
    try {
    const output = await new Promise((resolve, reject) => {
        let stdout = '';
        const proc = spawn(python, [extractScript, file.path, String(label)]);
        proc.stdout.on('data', d => stdout += d.toString());
        proc.on('close', code => {
        if (code !== 0) reject(new Error(`Python salió con código ${code}`));
        else resolve(stdout.trim());
        });
    });

    const parsed = JSON.parse(output);
    if (parsed.error) throw new Error(parsed.error);

    // Renombrar 'label' → 'Etiqueta' para que coincida con el dataset
    const rowsNormalized = parsed.rows.map(r => {
        const { label: lbl, ...rest } = r;
        return { ...rest, Etiqueta: lbl };
    });

    existingRows.push(...rowsNormalized);
    totalAdded += parsed.cycles;
    results.push({ file: file.originalname, cycles: parsed.cycles, ok: true });

    } catch (e) {
    results.push({ file: file.originalname, ok: false, error: e.message });
    } finally {
    fs.unlink(file.path, () => {});
    }
}

  // Guardar dataset candidato actualizado
  const wb  = XLSX.utils.book_new();
  const ws  = XLSX.utils.json_to_sheet(existingRows);
  XLSX.utils.book_append_sheet(wb, ws, 'dataset');
  XLSX.writeFile(wb, DATASET_TEMP_PATH);

  res.json({
    status:      'success',
    added_cycles: totalAdded,
    total_rows:  existingRows.length,
    files:       results
  });
});


// ============================================================
//  3. POST /api/admin/retrain/eval
//     Entrena con dataset candidato y devuelve métricas
//     NO toca nada de producción
// ============================================================
app.post('/api/admin/retrain/eval', requireLogin, requireRole(['admin']), (req, res) => {
  if (!fs.existsSync(DATASET_TEMP_PATH)) {
    return res.status(400).json({ error: 'No hay dataset candidato. Agrega audios primero.' });
  }

  const python     = getPythonExe();
  const evalScript = path.join(__dirname, 'retrain_eval.py');

  let stdout = '';
  const proc = spawn(python, [evalScript, DATASET_TEMP_PATH, MODEL_CAND_PATH]);
  proc.stdout.on('data', d => stdout += d.toString());
  proc.stderr.on('data', () => {}); // silenciar stderr de pycaret

  proc.on('close', code => {
    try {
      const result = JSON.parse(stdout.trim());
      res.json(result);
    } catch {
      res.status(500).json({ error: 'Respuesta inválida del script Python', raw: stdout });
    }
  });
});


// ============================================================
//  4. POST /api/admin/retrain/promote
//     Reemplaza modelo y dataset de producción con candidatos
// ============================================================
app.post('/api/admin/retrain/promote', requireLogin, requireRole(['admin']), (req, res) => {
  try {
    const candModel = MODEL_CAND_PATH + '.pkl';
    if (!fs.existsSync(candModel)) {
      return res.status(400).json({ error: 'No existe modelo candidato. Ejecuta la evaluación primero.' });
    }
    if (!fs.existsSync(DATASET_TEMP_PATH)) {
      return res.status(400).json({ error: 'No existe dataset candidato.' });
    }

    // Backup del modelo anterior
    const prodModel = MODEL_PROD_PATH + '.pkl';
    if (fs.existsSync(prodModel)) {
      fs.copyFileSync(prodModel, MODEL_PROD_PATH + '_backup.pkl');
    }
    // Backup del dataset anterior
    if (fs.existsSync(DATASET_PATH)) {
      fs.copyFileSync(DATASET_PATH, DATASET_PATH.replace('.xlsx', '_backup.xlsx'));
    }

    // Promover candidatos a producción
    fs.copyFileSync(candModel,        prodModel);
    fs.copyFileSync(DATASET_TEMP_PATH, DATASET_PATH);

    // Limpiar candidatos
    fs.unlinkSync(candModel);
    fs.unlinkSync(DATASET_TEMP_PATH);

    res.json({ status: 'success', message: 'Modelo y dataset promovidos a producción.' });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});


// ============================================================
//  5. POST /api/admin/retrain/discard
//     Descarta el dataset y modelo candidatos sin tocar producción
// ============================================================
app.post('/api/admin/retrain/discard', requireLogin, requireRole(['admin']), (req, res) => {
  try {
    if (fs.existsSync(DATASET_TEMP_PATH))          fs.unlinkSync(DATASET_TEMP_PATH);
    if (fs.existsSync(MODEL_CAND_PATH + '.pkl'))   fs.unlinkSync(MODEL_CAND_PATH + '.pkl');
    res.json({ status: 'success', message: 'Candidatos descartados. Producción intacta.' });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Ruta para acceder al panel (HTML)
app.get('/admin', requireLogin, requireRole(['admin']), (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend', 'admin.html'));
});

app.use((err, req, res, next) => {
    if (err) return res.status(500).send(`Error Servidor: ${err.message}`);
    next();
});



const uploadVisual = multer({
    storage: multer.memoryStorage(),   // No guarda en disco permanente
    fileFilter: (req, file, cb) => {
        const allowed = ['.mp3', '.wav'];
        const ext = path.extname(file.originalname).toLowerCase();
        if (allowed.includes(ext)) cb(null, true);
        else cb(new Error('Solo .mp3 y .wav'), false);
    },
    limits: { fileSize: 10 * 1024 * 1024 }
});
 
app.post('/api/analyze-visual', uploadVisual.single('audio'), (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'Falta archivo de audio.' });
 
    // 1. Guardar temporalmente para que Python pueda leerlo
    const tmpDir  = path.join(__dirname, 'uploads', 'tmp');
    if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });
 
    const tmpName = `visual_${Date.now()}_${req.file.originalname}`;
    const tmpPath = path.join(tmpDir, tmpName);
 
    fs.writeFileSync(tmpPath, req.file.buffer);
 
    // 2. Resolver ejecutable Python (igual que en /upload)
    let pythonExecutable;
    const venvPathWin   = path.join(__dirname, 'venv', 'Scripts', 'python.exe');
    const venvPathLinux = path.join(__dirname, 'venv', 'bin', 'python');
 
    if (process.platform === 'win32' && fs.existsSync(venvPathWin)) {
        pythonExecutable = venvPathWin;
    } else if (process.platform !== 'win32' && fs.existsSync(venvPathLinux)) {
        pythonExecutable = venvPathLinux;
    } else {
        pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    }
 
    const scriptPath = path.join(__dirname, 'classify_visual.py');
    console.log(`[Visual] Python: ${pythonExecutable} | Script: ${scriptPath}`);
 
    const pythonProcess = spawn(pythonExecutable, [scriptPath, tmpPath]);
 
    let outputData = '';
    let errorData  = '';
 
    pythonProcess.stdout.on('data', d => outputData += d.toString());
    pythonProcess.stderr.on('data', d => {
        errorData += d.toString();
        console.error(`[PyVisual LOG]: ${d.toString()}`);
    });
 
    pythonProcess.on('error', err => {
        errorData = `No se pudo iniciar Python: ${err.message}`;
    });
 
    pythonProcess.on('close', code => {
        // Limpiar archivo temporal
        try { fs.unlinkSync(tmpPath); } catch (_) {}
 
        if (code !== 0 || !outputData) {
            return res.status(500).json({
                error: 'Error en el análisis Python.',
                details: errorData || 'Sin datos de salida.'
            });
        }
 
        let result;
        try {
            result = JSON.parse(outputData.trim());
        } catch (e) {
            console.error('[Visual] JSON parse error:', outputData.slice(0, 300));
            return res.status(500).json({ error: 'Respuesta JSON inválida del script.' });
        }
 
        if (result.status === 'error') {
            return res.status(500).json({ error: result.message });
        }
 
        // Devolver JSON completo al dashboard
        res.json(result);
    });
});
 
// ============================================================
// RUTA HTML: /dashboard
// Sirve el archivo pcg-dashboard.html como página del sistema
// Agregar pcg-dashboard.html a la carpeta public/
// ============================================================
app.get('/dashboard', requireLogin, (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend', 'pcg-dashboard.html'));
});


app.listen(PORT, () => {
    console.log(`🚀 Servidor en ${PORT}`);
});