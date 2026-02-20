# 🧠🏥 DermaRAG - Sistema Multiagente de Razonamiento Clínico para el Diagnóstico Dermatológico Asistido

DermaRAG es una prueba de concepto (PoC) académica que integra Visión Computacional (EfficientNet-B4 + Grad-CAM) y Agentes de IA (CrewAI) impulsados por RAG (Retrieval-Augmented Generation) para el análisis de lesiones cutáneas.

---

> **⚠️ AVISO MÉDICO-LEGAL IMPORTANTE:**
> Este repositorio contiene código con fines estrictamente educativos y de investigación en Inteligencia Artificial. **NO es un dispositivo médico certificado**. Las predicciones y textos generados por este software no reemplazan bajo ninguna circunstancia el juicio clínico de un profesional de la salud.

## 📁 Estructura del proyecto

```
DermaRAG/
├── .gitignore              # Reglas de lo que NO se debe subir
├── LICENSE                 # Nuestra protección legal
├── README.md               # La presentación del proyecto
├── app.py                  # Código principal de Streamlit
├── RAG_tool.py             # Código de la herramienta
└── requirements.txt        # Lista de librerías
```
---

## 🚀 Características

- Análisis Visual Explicable: Uso de Grad-CAM para visualizar las áreas de atención de la red neuronal sobre la lesión.

- Sistema Multiagente: Un agente "Auditor" valida la visión artificial, mientras un agente "Oncólogo" genera un reporte basado en evidencia.

- RAG Estricto: Recuperación de información de guías clínicas mediante bases de datos vectoriales (ChromaDB).

---

## 🛠️ Instalación y Uso (Local)
1. Clona este repositorio:
```bash
git clone [https://github.com/tu-usuario/DermaRAG.git](https://github.com/tu-usuario/DermaRAG.git)
```
2. Instala las dependencias:
```bash
pip install -r requirements.txt
```
3. Ejecuta la app:
```bash
streamlit run app.py
```
---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - mira el archivo LICENSE para detalles.

---

## ✨ Créditos

Creado por Miguel Granados. Este repositorio está abierto para mejoras por parte de la comunidad.
