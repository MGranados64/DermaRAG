# ==============================================================================
# 0. PARCHE DE SISTEMA (Requerido para HF Spaces / ChromaDB)
# ==============================================================================
import sys

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

# ==============================================================================
# 1. LIBRERÍAS
# ==============================================================================
import streamlit as st
import os
import time
import csv
import math
import datetime
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F  
from torchvision import transforms
from torchvision.models import efficientnet_b4
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from crewai import Agent, Task, Crew, Process, LLM
from RAG_tool import BuscadorGuiasClinicas
from fpdf import FPDF

# LIBRERÍAS RAGAS (Usando Clases Base)
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI 

# ==============================================================================
# 2. CONFIGURACIÓN VISUAL Y DE PRIVACIDAD
# ==============================================================================
st.set_page_config(
    page_title="DermaRAG - Diagnóstico", 
    page_icon="🏥", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Inicialización de variables de estado para privacidad
if "privacy_ack" not in st.session_state: 
    st.session_state["privacy_ack"] = False
if "show_privacy_dialog" not in st.session_state: 
    st.session_state["show_privacy_dialog"] = True
if "consent_data_health" not in st.session_state: 
    st.session_state["consent_data_health"] = False
if "consent_ai_support" not in st.session_state: 
    st.session_state["consent_ai_support"] = False
if "consent_images" not in st.session_state: 
    st.session_state["consent_images"] = False

# ==============================================================================
# 3. INYECCIÓN DE CSS
# ==============================================================================
st.markdown("""
    <style>
    .block-container { padding-top: 3rem; padding-bottom: 5rem; padding-left: 5rem; padding-right: 5rem; max-width: 80% !important; }
    .stApp { background-color: #f4f6f9; color: #333333; }
    .header-container { background: linear-gradient(135deg, #003366 0%, #004080 100%); padding: 30px; border-radius: 12px; color: white; text-align: center; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    .stApp .header-container h1, .stApp .header-container p, .stMarkdown .header-container p, .stMarkdown .header-container h1 { color: white !important; border-bottom: none !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background-color: #ffffff !important; border-radius: 12px !important; padding: 20px !important; border: 1px solid #e0e0e0 !important; box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important; }
    h1, h2, h3, h4, h5 { color: #003366 !important; }
    h2 { border-bottom: 2px solid #667eea; padding-bottom: 8px; margin-bottom: 20px !important; }
    .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"], .stNumberInput div[data-baseweb="input"] { background-color: #ffffff !important; color: #333333 !important; border: 1px solid #cccccc !important; }
    .stNumberInput input, [data-testid="stFileUploaderDropzone"] section, [data-testid="stFileUploaderDropzone"] div, [data-testid="stFileUploaderDropzone"] span { color: #333333 !important; }
    .stNumberInput button { background-color: #f0f2f6 !important; color: #333333 !important; }
    [data-testid="stFileUploaderDropzone"] { background-color: #f8f9fa !important; border: 2px dashed #667eea !important; }
    [data-testid="stFileUploaderDropzone"] button { background-color: #ffffff !important; color: #003366 !important; border: 1px solid #003366 !important; }
    .stCheckbox label p, .stCheckbox label span, label p, label span, .stMarkdown p:not(.header-container p) { color: #333333 !important; font-weight: 500 !important; }
    div.stButton > button { background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important; color: white !important; border: none !important; padding: 15px 30px !important; font-size: 18px !important; font-weight: bold !important; border-radius: 8px !important; width: 100% !important; box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3) !important; }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4) !important; }
    div.stButton > button p { color: white !important; }
    .privacy-banner { background: linear-gradient(135deg, #fff7e6 0%, #fff3cd 100%); border: 1px solid #f0c36d; border-left: 8px solid #d97706; border-radius: 12px; padding: 18px 22px; margin-bottom: 20px; color: #5b3b00; }
    .privacy-banner h3, .privacy-banner p, .privacy-banner li { color: #5b3b00 !important; }
    .privacy-note-box { background: #f8fbff; border: 1px solid #cfe2ff; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }
    .privacy-note-box strong, .privacy-note-box p, .privacy-note-box li { color: #0b3a66 !important; }
    .medical-warning { background: #fff1f2; border: 1px solid #fecdd3; border-left: 6px solid #e11d48; border-radius: 10px; padding: 14px 16px; margin-top: 10px; color: #881337; font-size: 14px; }
    .medical-warning strong, .medical-warning p { color: #881337 !important; }
    /* RESPONSIVE MÓVIL */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        .header-container h1 {
            font-size: 1.3rem !important;
            line-height: 1.4 !important;
        }
        .header-container p {
            font-size: 0.85rem !important;
        }
    }

    /* FORZADO ANTI-FLICKERING (ESTABILIZADOR DE IMAGEN) */
    [data-testid="stImage"], [data-testid="stImage"] > img {
        zoom: 1 !important; 
        transform: translateZ(0) !important; 
        backface-visibility: hidden !important;
        -webkit-transform: translate3d(0,0,0) !important;
        perspective: 1000px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3.1 FUNCIONES DE PRIVACIDAD
# ==============================================================================
def render_main_privacy_messages():
    st.markdown("""
        <div class="privacy-banner">
            <h3>🔐 Aviso importante sobre privacidad y uso responsable</h3>
            <p>Esta plataforma tiene <strong>fines académicos</strong> y utiliza <strong>inteligencia artificial como apoyo</strong> para el análisis dermatológico.
            <strong>No sustituye</strong> el criterio médico, el diagnóstico profesional ni la atención clínica presencial.</p>
            <ul>
                <li>Ingrese únicamente la <strong>información mínima necesaria</strong> para el análisis.</li>
                <li>Evite nombres completos, números de identidad, direcciones u otros <strong>identificadores directos</strong>.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
            <div class="privacy-note-box">
                <strong>📌 Tratamiento de datos y minimización</strong>
                <p>Los datos personales y de salud son sensibles. Por ello, solo deben cargarse los datos estrictamente necesarios para fines académicos.</p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="privacy-note-box">
                <strong>🖼️ Uso de imágenes clínicas</strong>
                <p>No cargue imágenes con elementos innecesarios que permitan identificar directamente al paciente, salvo autorización.</p>
            </div>
        """, unsafe_allow_html=True)

def open_consent_dialog(force=False):
    dialog_callable = getattr(st, "dialog", None)
    
    if dialog_callable is None:
        st.warning("Este entorno no soporta ventanas modales. Consentimiento en línea:")
        with st.container(border=True):
            consent_data = st.checkbox("Comprendo que trato datos sensibles.", key="inline_data")
            consent_ai = st.checkbox("Comprendo que es IA de apoyo.", key="inline_ai")
            consent_img = st.checkbox("Confirmo anonimización.", key="inline_img")
            
            if st.button("Aceptar y continuar", key="inline_accept"):
                if consent_data and consent_ai and consent_img: 
                    st.session_state.update({"privacy_ack": True, "show_privacy_dialog": False})
                    st.rerun()
                else: 
                    st.error("Debe aceptar todos los puntos.")
        return

    @dialog_callable("Consentimiento informado y privacidad")
    def _dialog():
        st.markdown("""
        Antes de utilizar la plataforma, confirme lo siguiente:
        - Esta herramienta tiene **fines académicos**.
        - Usa **IA como apoyo** y **no sustituye** evaluación médica profesional.
        - Solo ingresará datos **autorizados, anonimizados o seudonimizados**.
        """)
        consent_data = st.checkbox("Comprendo el tratamiento de datos.", key="modal_data")
        consent_ai = st.checkbox("Comprendo que la IA es de apoyo.", key="modal_ai")
        consent_img = st.checkbox("Cuento con autorización para imágenes.", key="modal_img")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Aceptar y continuar", use_container_width=True):
                if consent_data and consent_ai and consent_img: 
                    st.session_state.update({"privacy_ack": True, "show_privacy_dialog": False})
                    st.rerun()
                else: 
                    st.error("Debe aceptar todos los puntos.")
        with c2:
            if st.button("Cerrar", use_container_width=True):
                st.session_state["show_privacy_dialog"] = False
                if force: 
                    st.warning("Debe aceptar para continuar.")
                st.rerun()
                
    _dialog()

# ==============================================================================
# 4. CLASES DE VISIÓN (GRAD-CAM)
# ==============================================================================
class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.activations = {}
        for name, layer in target_layers.items(): 
            layer.register_forward_hook(self.get_hook(name)) 
            
    def get_hook(self, name):
        def hook(model, input, output): 
            self.activations[name] = output.detach() 
        return hook

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient) 
        
    def save_activation(self, module, input, output): 
        self.activations = output 
        
    def save_gradient(self, module, grad_input, grad_output): 
        self.gradients = grad_output[0] 
        
    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        idx = torch.argmax(output, dim=1)
        output[0, idx].backward() 
        
        grads = self.gradients.cpu().data.numpy()[0]
        fmaps = self.activations.cpu().data.numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(fmaps.shape[1:], dtype=np.float32) 
        
        for i, w in enumerate(weights): 
            cam += w * fmaps[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (380, 380))
        cam = (cam - np.min(cam)) / (np.max(cam) + 1e-8) 
        
        return cam, output, idx 

def plot_feature_maps(activations, layer_name, title, output_file):
    act = activations[layer_name].squeeze().cpu().numpy()
    mean_act = np.mean(act, axis=(1, 2))
    top_indices = np.argsort(mean_act)[::-1][:16] 
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(top_indices):
            fmap_idx = top_indices[idx]
            fmap = act[fmap_idx]
            fmap = (fmap - np.min(fmap)) / (np.max(fmap) + 1e-8)
            ax.imshow(fmap, cmap='viridis')
            ax.set_title(f"Filtro {fmap_idx}", fontsize=8) 
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return output_file

@st.cache_resource
def cargar_tu_modelo_especifico(ruta_pth):
    model = efficientnet_b4(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.45), 
        nn.Linear(num_ftrs, 3)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try: 
        state_dict = torch.load(ruta_pth, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e: 
        st.error(f"❌ Error cargando pesos: {e}")
        return None
        
    model.to(device)
    model.eval()
    
    return model

transformacion_validacion = transforms.Compose([
    transforms.Resize((380, 380)), 
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def ejecutar_pipeline_gradcam(modelo, ruta_img, temp_dir):
    feature_extractor = FeatureExtractor(
        modelo, 
        {'capa_inicial': modelo.features[0], 'capa_final': modelo.features[-1]}
    )
    grad_cam = GradCAM(modelo, modelo.features[-1]) 
    
    pil_img = Image.open(ruta_img).convert('RGB')
    device = next(modelo.parameters()).device
    img_tensor = transformacion_validacion(pil_img).unsqueeze(0).to(device) 
    
    cam_map, logits, pred_idx = grad_cam(img_tensor)
    probs = F.softmax(logits, dim=1).cpu().data.numpy()[0]
    CLASES_NOMBRES = ['Benigno', 'Melanoma', 'Carcinoma'] 
    
    img_cv = cv2.imread(ruta_img)
    img_cv = cv2.resize(img_cv, (380, 380))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0) 
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.title(f"Atención IA\n({CLASES_NOMBRES[pred_idx]})")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    bars = plt.bar(CLASES_NOMBRES, probs, color=['green', 'red', 'orange'])
    plt.title("Probabilidades")
    plt.ylim(0, 1.15) 
    
    for bar in bars: 
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, 
            bar.get_height() + 0.02, 
            f'{bar.get_height()*100:.1f}%', 
            ha='center', 
            va='bottom', 
            fontsize=10, 
            fontweight='bold'
        )
        
    path_diag = os.path.join(temp_dir, "1_diagnostico_clinico.png")
    plt.savefig(path_diag)
    plt.close()
    
    path_bordes = os.path.join(temp_dir, "2_analisis_bordes.png")
    plot_feature_maps(
        feature_extractor.activations, 
        'capa_inicial', 
        "BORDES Y FORMAS", 
        path_bordes
    ) 
    
    path_patrones = os.path.join(temp_dir, "3_analisis_patrones.png")
    plot_feature_maps(
        feature_extractor.activations, 
        'capa_final', 
        "TEXTURA", 
        path_patrones
    ) 
    
    return path_diag, path_bordes, path_patrones, CLASES_NOMBRES[pred_idx], probs

def analizar_imagen_medica(ruta_imagen, modelo):
    if modelo is None: 
        return "Error: Modelo no cargado."
        
    CLASES = ['Benigno', 'Melanoma', 'Carcinoma'] 
    
    try:
        image = transformacion_validacion(Image.open(ruta_imagen).convert('RGB')).unsqueeze(0).to(next(modelo.parameters()).device)
        with torch.no_grad(): 
            probs = torch.nn.functional.softmax(modelo(image), dim=1)
            clase_idx = torch.argmax(probs, 1).item()
            
        return f"ANÁLISIS DE IA:\n- Predicción: {CLASES[clase_idx].upper()}\n- Confianza: {probs[0][clase_idx].item()*100:.2f}%\n  * Benigno: {probs[0][0].item()*100:.2f}%\n  * Melanoma: {probs[0][1].item()*100:.2f}%\n  * Carcinoma: {probs[0][2].item()*100:.2f}%"
    except Exception as e: 
        return f"Error: {str(e)}"

# ==============================================================================
# 5. GENERADOR PDF
# ==============================================================================
class PDFReport(FPDF):
    def __init__(self, paciente_info): 
        super().__init__()
        self.paciente_info = paciente_info
        
    def header(self): 
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'DermaRAG - Informe Diagnóstico', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(5)
        
    def footer(self): 
        self.set_y(-20)
        # Disclaimer medico-legal (en cada pagina)
        self.set_font('Arial', 'I', 7)
        self.set_text_color(150, 30, 30)
        disclaimer = ("AVISO MEDICO-LEGAL: Esta herramienta tiene fines academicos, "
                      "usa IA como apoyo y no sustituye evaluacion medica profesional.")
        self.multi_cell(0, 3, disclaimer, 0, 'C')
        # Datos del paciente y numero de pagina
        self.set_text_color(0, 0, 0)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 5, f"ID Paciente: {self.paciente_info['id']} | Pag {self.page_no()}", 0, 0, 'C')
        
    def chapter_title(self, label): 
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, label, 0, 1, 'L', 1)
        self.ln(4)
        
    def chapter_body(self, text): 
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, text)
        self.ln()

# ==============================================================================
# 6. INTERFAZ DE USUARIO STREAMLIT
# ==============================================================================
st.markdown("""
    <div class="header-container">
        <h1 style="color: white !important; font-size: clamp(1.1rem, 5vw, 2rem); line-height: 1.3; word-wrap: break-word; overflow-wrap: break-word;">🏥 DermaRAG - Sistema Multiagente de Diagnóstico Dermatológico</h1>
        <p style="color: white !important;">Prototipo Funcional| IA Explicable con Retrieval-Augmented Generation | Guías AAD/BAD/NCCN</p>
    </div>
""", unsafe_allow_html=True)

# Validación de Token GROQ
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY: 
    st.error("⚠️ Falta el token de Groq. Añade `GROQ_API_KEY` a tus Secrets.")
    st.stop()

# Funciones de privacidad en la vista principal
render_main_privacy_messages()

if not st.session_state.get("privacy_ack", False) and st.session_state.get("show_privacy_dialog", True): 
    open_consent_dialog(force=False)

RUTA_MODELO = 'mejor_modelo_v5.pth'

if os.path.exists(RUTA_MODELO):
    modelo_cnn = cargar_tu_modelo_especifico(RUTA_MODELO)
else:
    st.error(f"⚠️ Falta '{RUTA_MODELO}'")
    modelo_cnn = None

col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    with st.container(border=True):
        st.markdown("## 📋 Datos del Paciente")
        c1, c2 = st.columns(2)
        nombre = c1.text_input("Nombre del paciente *", placeholder="Ej. Gerardo García")
        edad = c1.number_input("Edad *", value=0, min_value=0, max_value=120, step=1)
        fototipo = c1.selectbox(
            "Fototipo Fitzpatrick *", 
            ["Tipo I - Piel muy clara", "Tipo II - Piel clara", "Tipo III - Piel intermedia", 
                 "Tipo IV - Piel morena clara", "Tipo V - Piel morena", "Tipo VI - Piel negra"], 
            index=None,
            placeholder="Selecciona una opción..."
        )
        id_paciente = c2.text_input("ID Paciente *", placeholder="Ej. PAC-2025-001")
        sexo = c2.selectbox("Sexo *", ["Masculino", "Femenino", "Otro"], index=None, placeholder="Seleccionar...")
    
    with st.container(border=True):
        st.markdown("## 🔬 Datos Clínicos de la Lesión")
        localizacion = st.selectbox(
            "Localización Anatómica *", 
            ["Tronco (pecho/espalda)", "Cabeza y Cuello", "Extremidades Superiores", 
             "Extremidades Inferiores", "Manos/Pies (Acral)", "Mucosas"], 
            index=None,
            placeholder="Seleccionar ubicación..."
        )
        cc1, cc2 = st.columns(2)
        tamano = cc1.number_input("Tamaño (mm) *", value=0, min_value=0, step=1)
        evolucion = cc2.number_input("Evolución (meses)", value=0, min_value=0, step=1)
        sintomas = st.text_area("Síntomas Asociados", placeholder="Ej. Prurito, sangrado, asimetría...", height=80)
        historia = st.text_area("Antecedentes Relevantes", placeholder="Ej. Historia familiar de melanoma...", height=80)
    
    with st.container(border=True):
        st.markdown("## 🔎 Criterios ABCDE (Dermoscopia Visual)")
        col_checks = st.columns(5)
        check_a = col_checks[0].checkbox("A", value=False, help="Asimetría")
        check_b = col_checks[1].checkbox("B", value=False, help="Bordes Irregulares")
        check_c = col_checks[2].checkbox("C", value=False, help="Color (Policromía)")
        check_d = col_checks[3].checkbox("D", value=False, help="Diámetro > 6mm")
        check_e = col_checks[4].checkbox("E", value=False, help="Evolución")

with col_der:
    with st.container(border=True):
        st.markdown("## 📸 Imagen de la Lesión Cutánea")
        uploaded_file = st.file_uploader("Sube imagen (JPG/PNG)", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            img_temp_pil = Image.open(uploaded_file)
            w, h = img_temp_pil.size
            size_kb = uploaded_file.size / 1024
            
            st.success(f"✅ Archivo: {uploaded_file.name} | {size_kb:.2f} KB | {w}x{h} px")
            st.image(img_temp_pil, caption="Vista Previa", width=375)

    st.markdown("""
        <div class="medical-warning">
            <strong>Antes de analizar:</strong> Confirme consentimiento de privacidad.
        </div>
    """, unsafe_allow_html=True)
    
    analyze_btn = st.button("🔍 Analizar con IA Multiagente + GradCAM", use_container_width=True)

# ==============================================================================
# 7. EJECUCIÓN DEL SISTEMA
# ==============================================================================
if analyze_btn:
    if not st.session_state.get("privacy_ack", False): 
        st.session_state["show_privacy_dialog"] = True
        open_consent_dialog(force=True)
        st.stop()

    if uploaded_file and modelo_cnn:
        if not nombre or localizacion is None: 
            st.error("⚠️ Completa al menos: Nombre y Localización.")
        else:
            with st.status("🔄 Ejecutando Sistema...", expanded=True) as status:
                temp_dir = tempfile.mkdtemp()
                ruta_input = os.path.join(temp_dir, "input.jpg")
                
                with open(ruta_input, "wb") as f: 
                    f.write(uploaded_file.getvalue())

                st.write("🧠 Percepción Visual...")
                t0 = time.time()
                
                path_diag, path_bordes, path_patrones, pred_clase, probs = ejecutar_pipeline_gradcam(
                    modelo_cnn, 
                    ruta_input, 
                    temp_dir
                )
                
                resultado_vision = analizar_imagen_medica(ruta_input, modelo_cnn)
                latencia_vision = time.time() - t0

                st.write("⚕️ Razonamiento Clínico Groq...")
                
                llm_agentes = LLM(
                    model="groq/llama-3.3-70b-versatile",
                    api_key=GROQ_API_KEY,
                    temperature=0.5
                )

                llm_especialista = LLM(
                    model="groq/openai/gpt-oss-120b",
                    api_key=GROQ_API_KEY,
                    temperature=0.1,
                    max_tokens=8000
                )

                hallazgos_lista = []
                if check_a: hallazgos_lista.append("Asimetría")
                if check_b: hallazgos_lista.append("Bordes")
                if check_c: hallazgos_lista.append("Policromía")
                if check_d: hallazgos_lista.append(f"Diámetro > 6mm ({tamano}mm)")
                if check_e: hallazgos_lista.append("Evolución")
                
                hallazgos_txt = ", ".join(hallazgos_lista) if hallazgos_lista else "Ninguno"
                
                task_med = (
                    f"DATOS: {edad} años, {sexo}, Fototipo: {fototipo}\n"
                    f"CLÍNICA: {localizacion}, {tamano}mm, {evolucion} meses\n"
                    f"SÍNTOMAS: {sintomas}\n"
                    f"ANTECEDENTES: {historia}\n"
                    f"ABCDE: {hallazgos_txt}\n"
                    f"VISION IA: [{resultado_vision}]"
                )

                medico_atencion_primaria = Agent(
                    role='Auditor Clínico', 
                    goal=f'Validar coherencia Grad-CAM vs clínica. Contexto: {task_med}', 
                    backstory='Especialista en Triaje. Tu filosofía: "La IA es herramienta". IDIOMA OBLIGATORIO: EXCLUSIVAMENTE ESPAÑOL.', 
                    verbose=True, 
                    allow_delegation=False, 
                    llm=llm_agentes
                )
                
                herramienta_rag = BuscadorGuiasClinicas()
                
                especialista_piel = Agent(
                    role='Oncólogo Dermatólogo Basado en Evidencia',
                    goal=(
                        'Generar un plan oncológico respaldado EXCLUSIVAMENTE por las guías '
                        'clínicas indexadas (NCCN, AAD, BAD, oncosur). NUNCA respondes de '
                        'memoria. Tu primer acto SIEMPRE es consultar la herramienta de búsqueda.'
                    ),
                    backstory=(
                        'Eres un oncólogo dermatólogo certificado que SOLO confía en evidencia '
                        'documentada. Tu protocolo personal es: "Sin guía, no hay respuesta". '
                        'Antes de emitir cualquier opinión, SIEMPRE consultas las guías clínicas '
                        'mediante la herramienta disponible. OBLIGACIÓN ABSOLUTA: REDACTAR EN '
                        'ESPAÑOL PERFECTO. Tus respuestas siempre incluyen citas textuales con '
                        'la fuente exacta.'
                    ),
                    verbose=True,
                    allow_delegation=False,
                    tools=[herramienta_rag],
                    max_iter=12,
                    llm=llm_especialista
                )

                task_atencion_primaria = Task(
                    description=f"Analiza: {task_med}. REGLA: 100% ESPAÑOL. Fidelidad a IA. Traducción Semiológica.", 
                    agent=medico_atencion_primaria, 
                    expected_output="1. Validación Visión\n2. Resumen Semiológico\n3. Solicitud Interconsulta"
                )
                
                task_especialista = Task(
                    description=(
                        f"Eres Oncólogo Dermatólogo. El paciente presenta:\n"
                        f"- Predicción IA (CNN): {pred_clase}\n"
                        f"- Localización: {localizacion}\n"
                        f"- Tamaño: {tamano} mm\n"
                        f"- Evolución: {evolucion} meses\n"
                        f"- Fototipo: {fototipo}\n"
                        f"- Hallazgos ABCDE: {hallazgos_txt}\n"
                        f"- Síntomas: {sintomas}\n"
                        f"- Antecedentes: {historia}\n\n"
                        "═══════════════════════════════════════════════\n"
                        "PASO 1 OBLIGATORIO — ANTES de redactar UNA sola palabra del informe, "
                        "DEBES llamar la herramienta 'buscador_guias_clinicas' AL MENOS 5 VECES "
                        "con estas queries EXACTAS, una por una:\n\n"
                        f"  Query 1: 'protocolo tratamiento {pred_clase.lower()}'\n"
                        f"  Query 2: 'márgenes quirúrgicos {pred_clase.lower()}'\n"
                        f"  Query 3: 'cirugía Mohs {pred_clase.lower()}'\n"
                        f"  Query 4: 'estadificación {pred_clase.lower()} factores riesgo'\n"
                        f"  Query 5: 'seguimiento {pred_clase.lower()} recurrencia'\n\n"
                        "Si no llamas la herramienta 5 veces, tu respuesta será RECHAZADA.\n"
                        "═══════════════════════════════════════════════\n\n"
                        "PASO 2 — REGLA DE FIDELIDAD ABSOLUTA (CRÍTICO):\n"
                        "1. CADA AFIRMACIÓN clínica del informe debe estar respaldada por una cita "
                        "TEXTUAL Y LITERAL (copy-paste exacto, palabra por palabra) de un fragmento "
                        "recuperado por la herramienta. PROHIBIDO parafrasear, resumir o reformular.\n"
                        "2. Antes de escribir cada oración, identifica primero el fragmento que la "
                        "respalda. Si no encuentras un fragmento que diga LITERALMENTE eso, NO LO "
                        "ESCRIBAS.\n"
                        "3. Las citas en la sección Referencias deben ser COPIA EXACTA de los "
                        "fragmentos del RAG. No invento, no embellezco, no acorto.\n\n"
                        "═══════════════════════════════════════════════\n"
                        "PASO 3 — REGLA CRÍTICA DE CANTIDAD vs CALIDAD:\n"
                        "Mínimo 3 referencias, máximo 6. PROHIBIDO rellenar con citas inventadas "
                        "para llegar a un número objetivo. Es 1000 veces preferible 3 referencias "
                        "100% reales que 8 referencias mezcladas con invenciones.\n\n"
                        "ANTES de escribir cada referencia, pregúntate: ¿esta cita aparece "
                        "TEXTUALMENTE en alguno de los fragmentos que me devolvió la herramienta? "
                        "Si la respuesta es 'no estoy seguro', NO LA INCLUYAS.\n\n"
                        "Las citas que mencionan al paciente concreto (su edad, tamaño de lesión, "
                        "síntomas específicos) son SIEMPRE inventadas — las guías clínicas hablan "
                        "de poblaciones, no de pacientes individuales. Si una de tus 'citas' "
                        "menciona '50mm' o 'cabeza y cuello del paciente', es INVENTADA. "
                        "Bórrala.\n\n"
                        "PASO 4 — FUENTES VÁLIDAS (LISTA BLANCA):\n"
                        "Las únicas fuentes válidas son los archivos .pdf que aparezcan en los "
                        "fragmentos recuperados por la herramienta (ej: 'COL_D1_GUIA COMPLETA "
                        "carcinoma basocelular.pdf', 'jnccn-article-p1181.pdf', 'cutaneous_melanoma.pdf', "
                        "'guia-oncosur-de-melanoma.pdf', 'basoespino.pdf', etc.).\n\n"
                        "PROHIBIDO ABSOLUTAMENTE citar como fuente:\n"
                        "  ❌ 'Validación Visión'\n"
                        "  ❌ 'Resumen Semiológico'\n"
                        "  ❌ 'Análisis Clínico'\n"
                        "  ❌ Cualquier nombre que NO termine en .pdf\n"
                        "  ❌ Cualquier output del agente anterior (auditor clínico)\n\n"
                        "Si no tienes 3 fragmentos del RAG con archivos .pdf reales, usa solo los "
                        "que sí tengas (mínimo 3) y NO inventes los demás.\n"
                        "═══════════════════════════════════════════════\n\n"
                        "PASO 5 — Redacta el informe en ESPAÑOL siguiendo el expected_output. "
                        "Cada sección debe tener AL MENOS 4 oraciones sustantivas, todas con (ver Ref. N).\n\n"
                        "Si una query devuelve 'No se encontró información relevante', intenta "
                        f"con queries más cortas (ej: '{pred_clase.lower()}', 'biopsia piel')."
                    ),
                    agent=especialista_piel,
                    context=[task_atencion_primaria],
                    expected_output=(
                        "### 1. Diagnóstico Presuntivo\n"
                        "[4+ oraciones integrando IA, ABCDE, contexto. Cada afirmación con (ver Ref. N).]\n\n"
                        "### 2. Protocolo de Tratamiento\n"
                        "[4+ oraciones: técnica, márgenes, alternativas. Cada afirmación con (ver Ref. N).]\n\n"
                        "### 3. Seguimiento\n"
                        "[4+ oraciones: frecuencia, signos de alarma, autoexamen. Cada afirmación con (ver Ref. N).]\n\n"
                        "### Referencias\n"
                        "(SOLO citas LITERALES copy-paste de fragmentos del RAG. Solo fuentes .pdf reales. "
                        "Mínimo 3, máximo 6. NUNCA mencionar al paciente individual en una cita.)\n\n"
                        "**Ref. 1:** \"[copy-paste LITERAL del fragmento, sin modificar nada]\"\n"
                        "_Fuente: nombre_archivo.pdf, página X_\n\n"
                        "**Ref. 2:** \"[copy-paste LITERAL del fragmento]\"\n"
                        "_Fuente: nombre_archivo.pdf, página Y_\n\n"
                        "**Ref. 3:** \"[copy-paste LITERAL del fragmento]\"\n"
                        "_Fuente: nombre_archivo.pdf, página Z_\n\n"
                        "(Agrega Ref. 4-6 SOLO si tienes fragmentos reales adicionales del RAG.)"
                    )
                )

                # Aquí limpiamos el archivo de memoria RAG antes de cada corrida
                # para no mezclar contextos de pacientes anteriores. La herramienta
                # BuscadorGuiasClinicas escribe ahí cada fragmento que recupera de ChromaDB.
                if os.path.exists("memoria_rag.txt"):
                    os.remove("memoria_rag.txt")

                crew = Crew(
                    agents=[medico_atencion_primaria, especialista_piel], 
                    tasks=[task_atencion_primaria, task_especialista], 
                    verbose=True, 
                    process=Process.sequential, 
                    language='es'
                )
                
                st.session_state['resultado_final'] = crew.kickoff()

                # Guardamos en session_state para mostrar FUERA del st.status .
                if os.path.exists("memoria_rag.txt"):
                    with open("memoria_rag.txt", "r", encoding="utf-8") as f:
                        n_frags = len([x for x in f.read().split("\n\n") if x.strip()])
                    st.session_state['rag_n_frags'] = n_frags
                else:
                    st.session_state['rag_n_frags'] = 0

                # DETECTOR DE REFERENCIAS FALSAS
                resultado_str = str(st.session_state.get('resultado_final', ''))
                fuentes_falsas = [
                    "Validación Visión", "Resumen Semiológico", "Análisis Clínico",
                    "Auditor Clínico", "Solicitud Interconsulta", "Validación de Visión",
                    "Resumen Semiologico", "Validacion Vision"
                ]
                st.session_state['refs_falsas'] = [f for f in fuentes_falsas if f in resultado_str]

                # VERIFICADOR DE CITAS: compara cada Ref. del informe contra los
                # fragmentos reales del RAG. Si una cita no aparece literalmente (o casi),
                # la marca como sospechosa de invención.
                import re

                def normalizar(texto):
                    """Quita puntuación, espacios extras, pasa a minúsculas para comparar."""
                    texto = re.sub(r'[^\w\s]', ' ', texto.lower())
                    texto = re.sub(r'\s+', ' ', texto).strip()
                    return texto

                def verificar_cita(cita, fragmentos_normalizados):
                    """
                    Devuelve True si la cita aparece (al menos parcialmente) en algún fragmento.
                    Usa matching de ventanas de 8 palabras consecutivas — basta que UNA ventana
                    coincida para considerar la cita como respaldada.
                    """
                    cita_norm = normalizar(cita)
                    palabras = cita_norm.split()
                    if len(palabras) < 5:
                        return False
                    # Generar ventanas deslizantes de 8 palabras
                    ventana_size = min(8, len(palabras))
                    for i in range(len(palabras) - ventana_size + 1):
                        ventana = ' '.join(palabras[i:i + ventana_size])
                        for frag_norm in fragmentos_normalizados:
                            if ventana in frag_norm:
                                return True
                    return False

                # Cargar fragmentos del RAG y normalizarlos
                citas_verificadas = []
                if os.path.exists("memoria_rag.txt"):
                    with open("memoria_rag.txt", "r", encoding="utf-8") as f:
                        fragmentos = [x.strip() for x in f.read().split("\n\n") if x.strip()]
                    fragmentos_norm = [normalizar(frag) for frag in fragmentos]

                    # Extraer todas las citas del formato: Ref. N: "..." o **Ref. N:** "..."
                    patron_cita = r'\*?\*?Ref\.?\s*(\d+):?\*?\*?\s*[""]([^""]+)[""]'
                    matches = re.findall(patron_cita, resultado_str)

                    for num_ref, texto_cita in matches:
                        es_real = verificar_cita(texto_cita, fragmentos_norm)
                        citas_verificadas.append({
                            'num': num_ref,
                            'texto': texto_cita[:100] + ('...' if len(texto_cita) > 100 else ''),
                            'real': es_real
                        })

                st.session_state['citas_verificadas'] = citas_verificadas
                
                st.session_state.update({
                    'diagnostico_generado': True, 
                    'pred_clase': pred_clase, 
                    'probs': probs, 
                    'path_diag': path_diag, 
                    'path_bordes': path_bordes, 
                    'path_patrones': path_patrones, 
                    'temp_dir': temp_dir, 
                    'ragas_scores': None,
                    'pdf_bytes': None,  
                    'pdf_para_id': None
                })
                
                latencia_total = time.time() - t0
                status.update(label=f"✅ Diagnóstico en {latencia_total:.2f}s", state="complete")
                
                archivo_logs = "logs_latencia.csv"
                if not os.path.exists(archivo_logs):
                    with open(archivo_logs, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["Fecha", "ID_Paciente", "Latencia_Vision_seg", "Latencia_Total_seg"])
                
                with open(archivo_logs, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([fecha_actual, id_paciente, round(latencia_vision, 2), round(latencia_total, 2)])
    else: 
        st.warning("⚠️ Por favor sube una imagen para proceder.") 

# ==============================================================================
# 8. RENDERIZADO FUERA DEL BOTÓN Y RAGAS MANUAL
# ==============================================================================
if st.session_state.get('diagnostico_generado', False):
    st.markdown("---")

    # BANNERS DE VERIFICACIÓN RAG (visibles fuera del st.status colapsado)
    n_frags = st.session_state.get('rag_n_frags', 0)
    if n_frags > 0:
        st.info(f"🔍 RAG recuperó **{n_frags} fragmentos** de las guías clínicas durante el análisis.")
    else:
        st.warning("⚠️ El agente NO invocó la herramienta RAG en esta corrida. Las métricas RAGas serán 0.")

    refs_falsas = st.session_state.get('refs_falsas', [])
    if refs_falsas:
        st.warning(
            f"⚠️ **Referencias inventadas detectadas:** {', '.join(refs_falsas)}. "
            f"El agente citó el output del agente anterior en vez de fragmentos reales del RAG. "
            f"Esto bajará la Fidelidad RAGas. Considera regenerar el diagnóstico."
        )

    # VERIFICADOR DE CITAS: muestra el desglose de cuántas refs son reales vs inventadas
    citas_verif = st.session_state.get('citas_verificadas', [])
    if citas_verif:
        n_total = len(citas_verif)
        n_reales = sum(1 for c in citas_verif if c['real'])
        n_inventadas = n_total - n_reales
        ratio = n_reales / n_total if n_total > 0 else 0

        if ratio >= 0.8:
            st.success(
                f"✅ **Verificación de citas:** {n_reales}/{n_total} referencias respaldadas "
                f"por fragmentos reales del RAG ({ratio*100:.0f}%)."
            )
        elif ratio >= 0.5:
            st.warning(
                f"⚠️ **Verificación de citas:** solo {n_reales}/{n_total} referencias están "
                f"respaldadas por el RAG ({ratio*100:.0f}%). Las demás parecen inventadas."
            )
        else:
            st.error(
                f"❌ **Verificación de citas:** solo {n_reales}/{n_total} referencias son reales "
                f"({ratio*100:.0f}%). El modelo está alucinando la mayoría de las citas."
            )

        # Desglose detallado en expander
        with st.expander(f"🔎 Ver desglose de las {n_total} citas"):
            for c in citas_verif:
                icono = "✅" if c['real'] else "❌"
                st.markdown(f"{icono} **Ref. {c['num']}:** _{c['texto']}_")

    st.subheader("👁️ Análisis Explicable y Auditoría")
    
    t1, t2, t3, t4 = st.tabs(["Diagnóstico IA", "Bordes (Capa Baja)", "Patrones (Capa Alta)", "📊 Auditoría RAGas"])
    
    with t1: 
        st.image(st.session_state['path_diag'], use_container_width=True)
    with t2: 
        st.image(st.session_state['path_bordes'], use_container_width=True)
    with t3: 
        st.image(st.session_state['path_patrones'], use_container_width=True)
        
    with t4:
        st.markdown("### Auditoría Clínica RAGas (Ejecución Manual)")
        
        if st.button("🚀 Ejecutar Auditoría", use_container_width=True):
            with st.spinner("Auditando con IA Juez Groq..."):
                try:
                    # Juez RAGas = GPT-OSS 120B
                    # ChatOpenAI usa el endpoint OpenAI-compatible de Groq directamente.
                    llm_juez = ChatOpenAI(
                        api_key=GROQ_API_KEY, 
                        base_url="https://api.groq.com/openai/v1", 
                        model="openai/gpt-oss-120b", 
                        temperature=0,
                        max_tokens=16000
                    )
                    embeddings_juez = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )
                    
                    # Leer los fragmentos reales que la herramienta
                    # BuscadorGuiasClinicas escribió en memoria_rag.txt durante el kickoff.
                    ctx = []
                    if os.path.exists("memoria_rag.txt"):
                        with open("memoria_rag.txt", "r", encoding="utf-8") as f:
                            contenido = f.read().strip()
                        ctx = [frag.strip() for frag in contenido.split("\n\n") if frag.strip()]

                    if not ctx:
                        st.error("⚠️ No hay contextos RAG para auditar. El agente especialista no usó la herramienta de guías clínicas en esta corrida. Vuelve a generar el diagnóstico.")
                        st.stop()

                    res_txt = str(st.session_state['resultado_final'])
                    # El informe cubre 3 secciones (diagnóstico, protocolo,
                    # seguimiento). Si la pregunta solo menciona "protocolo", la Relevancia baja
                    # porque las preguntas hipotéticas que RAGas genera no coinciden.
                    pregunta = (
                        f"¿Cuál es el diagnóstico presuntivo, el protocolo de tratamiento "
                        f"basado en guías clínicas, y el plan de seguimiento recomendado para "
                        f"un paciente con sospecha de {st.session_state['pred_clase']} "
                        f"de {tamano}mm localizado en {localizacion}, considerando los hallazgos "
                        f"ABCDE y la evidencia de las guías oncológicas?"
                    )

                    dataset = Dataset.from_dict({
                        "question": [pregunta],
                        "contexts": [ctx],
                        "answer": [res_txt]
                    })
                    
                    res = evaluate(
                        dataset=dataset, 
                        metrics=[Faithfulness(), AnswerRelevancy(strictness=1)], 
                        llm=llm_juez, 
                        embeddings=embeddings_juez, 
                        raise_exceptions=True
                    )
                    
                    def s_score(c): 
                        for col in res.to_pandas().columns:
                            if c.lower() in col.lower(): 
                                return 0.0 if math.isnan(res.to_pandas()[col][0]) else res.to_pandas()[col][0]
                        return 0.0
                    
                    st.session_state['ragas_scores'] = {
                        'f': s_score('faithfulness'), 
                        'r': s_score('relevancy')
                    }
                except Exception as e: 
                    st.error(f"Error RAGas: {e}")

        if st.session_state.get('ragas_scores'):
            c_r1, c_r2 = st.columns(2)
            
            def fmt(s):
                color = 'green' if s > 0.8 else 'orange' if s > 0.6 else 'red'
                return f"<span style='color: {color}; font-size:24px; font-weight:bold;'>{s:.2f}</span>"
            
            with c_r1: 
                st.markdown(f"**Fidelidad:**<br>{fmt(st.session_state['ragas_scores']['f'])}", unsafe_allow_html=True)
            with c_r2: 
                st.markdown(f"**Relevancia:**<br>{fmt(st.session_state['ragas_scores']['r'])}", unsafe_allow_html=True)

    st.markdown("### 📊 Informe Final")
    with st.container(border=True): 
        st.markdown(st.session_state['resultado_final'])
    
    st.markdown("""
        <div style="background: #fff1f2; border: 1px solid #fecdd3; border-left: 6px solid #e11d48; 
                    border-radius: 8px; padding: 12px 16px; margin-top: 12px; margin-bottom: 12px; 
                    font-size: 13px; color: #881337; text-align: center;">
            ⚠️ <strong>AVISO MÉDICO-LEGAL:</strong> Esta herramienta tiene fines académicos, 
            usa IA como apoyo y no sustituye evaluación médica profesional.
        </div>
    """, unsafe_allow_html=True)
    
    # El PDF se genera UNA sola vez y se cachea en session_state.
    # Antes se regeneraba en cada rerun causando reescritura del archivo + I/O continuo
    # + remontaje del download_button → flickering visible.
    if not st.session_state.get('pdf_bytes') or st.session_state.get('pdf_para_id') != id_paciente:
        pdf = PDFReport({'id': id_paciente, 'edad': edad})
        pdf.add_page()
        pdf.chapter_title("1. Análisis")
        pdf.image(st.session_state['path_diag'], w=190)
        pdf.ln(5)
        pdf.chapter_title("2. Informe")
        # Limpieza AGRESIVA de caracteres Unicode con unicodedata,
        # Porque GPT-OSS genera muchos caracteres no-latin1 (especialmente \u00a0 non-breaking space)
        import unicodedata
        texto_informe = str(st.session_state['resultado_final']).replace('**', '')
        reemplazos = {
            '\u00a0': ' ', '\u2013': '-', '\u2014': '-',
            '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
            '\u2026': '...', '\u2265': '>=', '\u2264': '<=', '\u00b1': '+/-',
            '\u2192': '->', '\u2190': '<-', '\u00b7': '*', '\u2022': '*',
            '\u00bf': '?', '\u00a1': '!', '\u2212': '-', '\u00d7': 'x',
            '\t': ' ',
        }
        for orig, repl in reemplazos.items():
            texto_informe = texto_informe.replace(orig, repl)
        texto_normalizado = ""
        for char in texto_informe:
            try:
                char.encode('latin-1')
                texto_normalizado += char
            except UnicodeEncodeError:
                decomp = unicodedata.normalize('NFKD', char)
                for c in decomp:
                    try:
                        c.encode('latin-1')
                        texto_normalizado += c
                    except UnicodeEncodeError:
                        pass
        pdf.chapter_body(texto_normalizado)
        
        # Generar bytes del PDF en memoria (sin escribir a disco repetidamente)
        out_pdf = os.path.join(st.session_state['temp_dir'], "reporte.pdf")
        pdf.output(out_pdf)
        with open(out_pdf, "rb") as f:
            st.session_state['pdf_bytes'] = f.read()
        st.session_state['pdf_para_id'] = id_paciente
    
    # Botón de descarga usa los bytes cacheados (sin I/O en cada rerun)
    st.download_button(
        "📄 Descargar PDF",
        data=st.session_state['pdf_bytes'],
        file_name=f"Reporte_{id_paciente}.pdf",
        mime="application/pdf",
        key="download_pdf_btn"
    )

# --- SIDEBAR ("Estado Consentimiento") ---
with st.sidebar:
    st.markdown("### 🔐 Estado Privacidad")
    if st.session_state.get("privacy_ack", False): 
        st.success("Consentimiento aceptado.")
    else: 
        st.warning("Pendiente.")
        if st.button("Ver consentimiento"): 
            st.session_state["show_privacy_dialog"]=True
            open_consent_dialog()
            
    st.markdown("---")
    st.markdown("### 📊 Panel de Administración")
    archivo_logs = "logs_latencia.csv"
    if os.path.exists(archivo_logs):
        st.write("Descarga los registros de tiempo para calcular el Percentil 95.")
        with open(archivo_logs, "rb") as f:
            st.download_button(
                label="📥 Descargar Logs (CSV)",
                data=f,
                file_name="historial_latencia_dermarag.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("Aún no hay logs generados. Analiza una imagen primero.")

    # ==========================================================================
    # 🔬 DIAGNÓSTICO CHROMADB
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 🔬 Diagnóstico ChromaDB")
    if st.button("Verificar base RAG", use_container_width=True):
        try:
            # ¿Existe la carpeta?
            if not os.path.exists("./chroma_db"):
                st.error("❌ La carpeta ./chroma_db NO existe en el Space.")
                st.info("Verifica que la carpeta esté subida al repo del Space (puede requerir git lfs).")
            else:
                archivos = os.listdir("./chroma_db")
                st.write(f"📁 Archivos en chroma_db: **{len(archivos)}**")
                with st.expander("Ver archivos"):
                    st.code("\n".join(archivos[:20]))

                # ¿Carga y tiene documentos?
                from langchain_huggingface import HuggingFaceEmbeddings as _HFE
                from langchain_community.vectorstores import Chroma as _Chroma

                emb = _HFE(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                db = _Chroma(persist_directory="./chroma_db", embedding_function=emb)
                total = db._collection.count()
                st.metric("Total de chunks indexados", total)

                if total == 0:
                    st.error("❌ La base existe pero está VACÍA. Hay que reindexar los PDFs.")
                else:
                    # Prueba de búsqueda real
                    resultados = db.similarity_search("margen melanoma", k=3)
                    st.success(f"✅ Búsqueda funcional. {len(resultados)} resultados para 'margen melanoma':")
                    for i, r in enumerate(resultados, 1):
                        fuente = r.metadata.get('source', '?')
                        pagina = r.metadata.get('page', '?')
                        with st.expander(f"📄 Resultado {i}: {os.path.basename(fuente)} (pág {pagina})"):
                            st.write(r.page_content[:500] + "...")
        except Exception as e:
            st.error(f"Error al verificar: {e}")
            import traceback
            with st.expander("Traceback completo"):
                st.code(traceback.format_exc())

st.markdown("""
    <div style='text-align: center; color: #666666; padding: 20px;'>
        DermaRAG MVP v2.0 | EfficientNet-B4 · Llama 3.3 70B + GPT-OSS 120B (Groq)
    </div>
    """, unsafe_allow_html=True)