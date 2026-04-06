-- ============================================================
--  hospital_db  ·  Script de creación completo
--  Compatible con MySQL 5.7+ / MariaDB 10.3+
--  Ejecutar en XAMPP / phpMyAdmin o línea de comandos
-- ============================================================

CREATE DATABASE IF NOT EXISTS hospital_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE hospital_db;

-- ------------------------------------------------------------
-- 1. codigos_acceso
--    Controla quién puede registrarse y con qué rol.
--    El servidor valida este código antes de crear el usuario.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS codigos_acceso (
    id           INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    codigo       VARCHAR(64)     NOT NULL UNIQUE,
    tipo_usuario ENUM('admin','medico','paciente') NOT NULL,
    activo       TINYINT(1)      NOT NULL DEFAULT 1,
    created_at   TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Códigos de ejemplo (cámbialos en producción)
INSERT INTO codigos_acceso (codigo, tipo_usuario) VALUES
    ('ADMIN-2026',   'admin'),
    ('MEDICO-2026',  'medico'),
    ('PACIENTE-2026','paciente');

-- ------------------------------------------------------------
-- 2. usuarios
--    Tabla central de autenticación.
--    tipo_usuario determina los permisos en el servidor.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS usuarios (
    id             INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    nombre_usuario VARCHAR(80)     NOT NULL UNIQUE,
    password_hash  VARCHAR(255)    NOT NULL,           -- bcrypt hash
    tipo_usuario   ENUM('admin','medico','paciente') NOT NULL,
    created_at     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_tipo (tipo_usuario)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- 3. pacientes
--    Datos personales ligados a un usuario de tipo 'paciente'.
--    La columna `edad` existe para compatibilidad con /upload
--    (el formulario envía nombre + edad directamente).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pacientes (
    id              INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    id_usuario      INT UNSIGNED    NULL,               -- NULL cuando se crea desde /upload sin cuenta
    nombre          VARCHAR(100)    NOT NULL,
    apellido        VARCHAR(100)    NOT NULL DEFAULT '',
    edad            TINYINT UNSIGNED NOT NULL DEFAULT 0,
    email           VARCHAR(150)    NULL,
    fecha_nacimiento DATE           NULL,
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_usuario (id_usuario),
    CONSTRAINT fk_paciente_usuario
        FOREIGN KEY (id_usuario) REFERENCES usuarios(id)
        ON DELETE SET NULL
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- 4. diagnosticos
--    Un paciente puede tener múltiples diagnósticos.
--    ruta_audio guarda la ruta absoluta en disco;
--    el servidor construye url_web = /uploads/<filename>.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS diagnosticos (
    id                 INT UNSIGNED    NOT NULL AUTO_INCREMENT,
    paciente_id        INT UNSIGNED    NOT NULL,
    ruta_audio         VARCHAR(500)    NOT NULL,
    resultado_ia       VARCHAR(100)    NULL,            -- ej. 'normal', 'murmur'
    confianza          DECIMAL(5,2)    NULL,            -- 0.00 – 100.00
    ciclos_detectados  SMALLINT UNSIGNED NULL,
    fecha_analisis     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_paciente (paciente_id),
    INDEX idx_fecha    (fecha_analisis),
    CONSTRAINT fk_diagnostico_paciente
        FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
--  Vista de conveniencia: últimos diagnósticos con nombre
--  Usada implícitamente por GET /api/pacientes
-- ============================================================
CREATE OR REPLACE VIEW v_historial AS
    SELECT
        d.id,
        p.nombre,
        p.edad,
        d.resultado_ia,
        d.confianza,
        d.fecha_analisis
    FROM diagnosticos d
    INNER JOIN pacientes p ON d.paciente_id = p.id
    ORDER BY d.fecha_analisis DESC;

-- ============================================================
--  Usuario MySQL para la aplicación
--  (ejecutar como root si aún no existe)
-- ============================================================
-- CREATE USER IF NOT EXISTS 'admin'@'localhost' IDENTIFIED BY 'Galgos2026!';
-- GRANT ALL PRIVILEGES ON hospital_db.* TO 'admin'@'localhost';
-- FLUSH PRIVILEGES;
