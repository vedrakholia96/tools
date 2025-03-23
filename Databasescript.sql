CREATE DATABASE IF NOT EXISTS lab_management;
USE lab_management;

-- step 2 select and run
-- In case you need to drop and re-create tables:
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS qna;
DROP TABLE IF EXISTS notifications;
DROP TABLE IF EXISTS eln_reports;
DROP TABLE IF EXISTS data_visualizations;
DROP TABLE IF EXISTS experiment_instruments;
DROP TABLE IF EXISTS experiment_data_entries;
DROP TABLE IF EXISTS experiment_collaborators;
DROP TABLE IF EXISTS experiment_conditions;
DROP TABLE IF EXISTS experiment_parameters;
DROP TABLE IF EXISTS experiments;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS instruments;
DROP TABLE IF EXISTS experiment_types;
DROP TABLE IF EXISTS departments;
DROP TABLE IF EXISTS organizations;

SET FOREIGN_KEY_CHECKS = 1;

-- 1) organizations
CREATE TABLE organizations (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2) departments
CREATE TABLE departments (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  organization_id INT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- 3) experiment_types
CREATE TABLE experiment_types (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4) instruments
CREATE TABLE instruments (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  organization_id INT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- 5) users
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(255) NOT NULL,
  email VARCHAR(255) NOT NULL,
  password_bcrypted TEXT NOT NULL,
  phone VARCHAR(15),
  role ENUM('super_admin','org_admin','staff') NOT NULL,
  organization_id INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (organization_id) REFERENCES organizations(id)
);

-- 6) experiments
CREATE TABLE experiments (
  id INT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  objective TEXT,
  type_id INT,
  start_date DATE,
  end_date DATE,
  department_id INT,
  created_by INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (type_id) REFERENCES experiment_types(id),
  FOREIGN KEY (department_id) REFERENCES departments(id),
  FOREIGN KEY (created_by) REFERENCES users(id)
);

-- 7) experiment_parameters
CREATE TABLE experiment_parameters (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  type_id INT,
  is_predefined BOOLEAN,
  FOREIGN KEY (type_id) REFERENCES experiment_types(id)
);

-- 8) experiment_conditions
CREATE TABLE experiment_conditions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  experiment_id INT NOT NULL,
  parameter_id INT NOT NULL,
  value TEXT,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id),
  FOREIGN KEY (parameter_id) REFERENCES experiment_parameters(id)
);

-- 9) experiment_collaborators
CREATE TABLE experiment_collaborators (
  id INT AUTO_INCREMENT PRIMARY KEY,
  experiment_id INT NOT NULL,
  user_id INT NOT NULL,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id),
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 10) experiment_data_entries
CREATE TABLE experiment_data_entries (
  id INT AUTO_INCREMENT PRIMARY KEY,
  experiment_id INT NOT NULL,
  recorded_by INT NOT NULL,
  raw_data JSON,
  uploaded_file VARCHAR(255),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id),
  FOREIGN KEY (recorded_by) REFERENCES users(id)
);

-- 11) experiment_instruments
CREATE TABLE experiment_instruments (
  id INT AUTO_INCREMENT PRIMARY KEY,
  experiment_id INT NOT NULL,
  instrument_id INT NOT NULL,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id),
  FOREIGN KEY (instrument_id) REFERENCES instruments(id)
);

-- 12) data_visualizations
CREATE TABLE data_visualizations (
  id INT AUTO_INCREMENT PRIMARY KEY,
  experiment_id INT NOT NULL,
  plot_type ENUM('bar','line','scatter'),
  settings JSON,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- 13) eln_reports
CREATE TABLE eln_reports (
  id INT AUTO_INCREMENT PRIMARY KEY,
  experiment_id INT NOT NULL,
  observations TEXT,
  raw_data JSON,
  visualizations JSON,
  conclusions TEXT,
  pdf_path VARCHAR(255),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- 14) notifications
CREATE TABLE notifications (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  message TEXT NOT NULL,
  is_read BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 15) qna
CREATE TABLE qna (
  id INT AUTO_INCREMENT PRIMARY KEY,
  organization_id INT NOT NULL,
  submitted_by INT NOT NULL,
  message TEXT NOT NULL,
  status ENUM('open','in_progress','resolved') DEFAULT 'open',
  resolved_by INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (organization_id) REFERENCES organizations(id),
  FOREIGN KEY (submitted_by) REFERENCES users(id),
  FOREIGN KEY (resolved_by) REFERENCES users(id)
);

-- step3 select and run
-- Organizations
INSERT INTO organizations (name) VALUES
('Acme Labs'),
('Invitria'),
('Future Innovations');

-- Departments
INSERT INTO departments (name, organization_id) VALUES
('R&D', 1),
('Quality Assurance', 1),
('Research', 2);

-- Experiment Types
INSERT INTO experiment_types (name) VALUES
('Chemical Analysis'),
('Biological Assay'),
('Physical Stress Test');

-- Instruments
INSERT INTO instruments (name, description, organization_id) VALUES
('HPLC Machine', 'High-performance liquid chromatography system', 1),
('Microscope', 'Optical microscope for cell observation', 2),
('Spectrometer', 'Measures electromagnetic spectra', 1);

-- Users
INSERT INTO users (username, email, password_bcrypted, phone, role, organization_id) VALUES
('alice', 'alice@example.com', '$2y$12$alicehash', '1234567890', 'super_admin', 1),
('bob',   'bob@example.com',   '$2y$12$bobhash',   '2345678901', 'org_admin',   1),
('carol', 'carol@example.com', '$2y$12$carolhash', '3456789012', 'staff',       2);

-- Experiments
INSERT INTO experiments (title, objective, type_id, start_date, end_date, department_id, created_by)
VALUES
('Protein Analysis', 'Analyze protein folding behavior', 2, '2025-01-15', '2025-02-15', 1, 1),
('Surface Stress Test', 'Determine stress fractures on materials', 3, '2025-02-01', NULL, 2, 2),
('Antibody Reaction', 'Test new antibody reactivity', 2, '2025-02-10', '2025-03-10', 3, 3);

-- Experiment Parameters
INSERT INTO experiment_parameters (name, type_id, is_predefined)
VALUES
('Temperature', 2, TRUE),
('pH', 1, TRUE),
('Concentration', 1, FALSE),
('Stress Level', 3, TRUE);

-- Experiment Conditions
INSERT INTO experiment_conditions (experiment_id, parameter_id, value)
VALUES
(1, 1, '37C'),
(1, 2, '7.4'),
(2, 4, 'High'),
(3, 1, '30C');

-- Experiment Collaborators
INSERT INTO experiment_collaborators (experiment_id, user_id)
VALUES
(1, 2),
(1, 3),
(2, 3),
(3, 1);

-- Experiment Data Entries
INSERT INTO experiment_data_entries (experiment_id, recorded_by, raw_data, uploaded_file)
VALUES
(1, 2, JSON_OBJECT('result','Protein concentration measured'), 'protein_report.pdf'),
(2, 3, JSON_OBJECT('stress','Peak stress at 1200 psi'), NULL),
(3, 1, JSON_OBJECT('notes','Antibody had unexpected reaction'), 'antibody_analysis.xlsx');

-- Experiment Instruments
INSERT INTO experiment_instruments (experiment_id, instrument_id)
VALUES
(1, 2),
(2, 3),
(3, 1);

-- Data Visualizations
INSERT INTO data_visualizations (experiment_id, plot_type, settings)
VALUES
(1, 'line', JSON_OBJECT('x_axis','time','y_axis','concentration')),
(2, 'bar', JSON_OBJECT('categories','stress_levels')),
(3, 'scatter', JSON_OBJECT('x','time','y','reaction_intensity'));

-- ELN Reports
INSERT INTO eln_reports (experiment_id, observations, raw_data, visualizations, conclusions, pdf_path)
VALUES
(1, 'Observation details here...', JSON_OBJECT('extra','raw data'), JSON_OBJECT('plot_type','line'), 'Concluded stable folding', 'report_1.pdf'),
(2, 'High stress tolerance', JSON_OBJECT('extra','materials'), JSON_OBJECT('graph','bar'), 'Material meets standard', 'report_2.pdf');

-- Notifications
INSERT INTO notifications (user_id, message, is_read)
VALUES
(2, 'Experiment 1 requires attention', FALSE),
(3, 'Experiment 2 data upload pending', FALSE);

-- QnA
INSERT INTO qna (organization_id, submitted_by, message, status, resolved_by)
VALUES
(1, 2, 'Need clarifications on data entry protocol', 'open', NULL),
(2, 3, 'Who is responsible for final sign-off?', 'resolved', 1);
