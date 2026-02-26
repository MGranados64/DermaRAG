# ==============================================================================
# 1. LIBRERIAS
# ==============================================================================
import streamlit as st
import os
import torch
import torch.nn as nn
import torch.nn.functional as F  
from torchvision import transforms, models
from torchvision.models import efficientnet_b4
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
from crewai import Agent, Task, Crew, Process
from RAG_tool import BuscadorGuiasClinicas
import tempfile
from fpdf import FPDF
import datetime

# ==============================================================================
# 2. CONFIGURACIÓN VISUAL Y MÁRGENES
# ==============================================================================
st.set_page_config(
    page_title="DermaRAG - Diagnóstico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# ==============================================================================
# 3. Inyección de CSS 
# ==============================================================================
st.markdown("""
    <style>
    /* AJUSTE DE MÁRGENES: Reducir padding del contenedor principal */
    .block-container {
        padding-top: 5rem;
        padding-bottom: 5rem;
        padding-left: 5rem;
        padding-right: 5rem;
        max-width: 75% !important;
    }

    /* 1. Fondo General Oscuro */
    .stApp {
        background-color: #0e1117; 
        color: #fafafa;
    }
    
    /* 2. Header Azul */
    .header-container {
        background: linear-gradient(135deg, #003366 0%, #001a33 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    
    /* 3. Paneles / Cards */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
        background-color: #262730;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #414245;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }

    /* 4. Botón CTA Verde */
    div.stButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        padding: 18px 40px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        width: 100%;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
        color: white;
    }

    /* 5. Títulos */
    h1, h2, h3 {
        color: #e6f3ff !important;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    
    /* 6. Inputs y TextAreas */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background-color: #1f2937;
        color: white;
        border-color: #4b5563;
    }
    
    /* 7. Disclaimer Box */
    .disclaimer {
        background-color: #3e3308;
        border-left: 5px solid #ffc107;
        padding: 20px;
        margin-top: 30px;
        border-radius: 5px;
        color: #ffda6a;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. CLASES PARA GRAD-CAM
# ==============================================================================

class FeatureExtractor:
    """Clase para 'espiar' lo que ve la red en capas intermedias [cite: 36, 37]"""
    def __init__(self, model, target_layers):
        self.activations = {}
        for name, layer in target_layers.items():
            layer.register_forward_hook(self.get_hook(name)) 

    def get_hook(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach() 
        return hook

class GradCAM:
    """Clase para mapa de calor de decisión final [cite: 47, 48]"""
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
    """Dibuja las 16 características más activas de una capa"""
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

# ==============================================================================
# 5. LÓGICA DE VISIÓN (Backend + Integración GradCAM)
# ==============================================================================

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
    """Función wrapper para ejecutar el flujo del PDF [cite: 115]"""
    # 1. DEFINIT CAPAS (Features extraction hooks)
    feature_extractor = FeatureExtractor(modelo, {
        'capa_inicial': modelo.features[0], 
        'capa_final': modelo.features[-1]   
    })
    
    # 2. GradCAM instance
    grad_cam = GradCAM(modelo, modelo.features[-1]) 
    
    # 3. PROCESAR IMAGEN
    pil_img = Image.open(ruta_img).convert('RGB')
    device = next(modelo.parameters()).device
    img_tensor = transformacion_validacion(pil_img).unsqueeze(0).to(device) 
    
    # 4. INFERENCIA GradCAM
    cam_map, logits, pred_idx = grad_cam(img_tensor) 
    probs = F.softmax(logits, dim=1).cpu().data.numpy()[0] 
    
    CLASES_NOMBRES = ['Benigno', 'Melanoma', 'Carcinoma'] 
    
    # 5. VISUALIZACION 1: DIAGNÓSTICO CLÍNICO (Grad-CAM)
    img_cv = cv2.imread(ruta_img) 
    img_cv = cv2.resize(img_cv, (380, 380)) 
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET) 
    superimposed = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0) 
    
    # 6. GENERAR FIGURA COMPUESTA
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
        altura = bar.get_height() 
        porcentaje = altura * 100
        plt.text(bar.get_x() + bar.get_width() / 2.0, altura + 0.02, 
                 f'{porcentaje:.1f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    path_diag = os.path.join(temp_dir, "1_diagnostico_clinico.png")
    plt.savefig(path_diag) 
    plt.close()
    
    # 7. Visualización 2: QUÉ BORDES VIO?
    path_bordes = os.path.join(temp_dir, "2_analisis_bordes.png")
    plot_feature_maps(
        feature_extractor.activations, 
        'capa_inicial', 
        "CARACTERÍSTICAS VISUALES: BORDES Y FORMAS", 
        path_bordes
    ) 
    
    # 8. Visualización 3: QUÉ PATRONES VIO?
    path_patrones = os.path.join(temp_dir, "3_analisis_patrones.png")
    plot_feature_maps(
        feature_extractor.activations, 
        'capa_final', 
        "CARACTERÍSTICAS ABSTRACTAS: TEXTURA Y ANOMALÍAS", 
        path_patrones
    ) 
    
    return path_diag, path_bordes, path_patrones, CLASES_NOMBRES[pred_idx], probs

def analizar_imagen_medica(ruta_imagen, modelo):
    # Función existente mantenida para reporte texto simple
    if modelo is None: return "Error: Modelo no cargado."
    CLASES = ['Benigno', 'Melanoma', 'Carcinoma'] 
    try:
        image = Image.open(ruta_imagen).convert('RGB')
        image = transformacion_validacion(image).unsqueeze(0) 
        device = next(modelo.parameters()).device
        image = image.to(device)
        with torch.no_grad():
            outputs = modelo(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            p_ben = probs[0][0].item() * 100
            p_mel = probs[0][1].item() * 100
            p_car = probs[0][2].item() * 100
            clase_idx = torch.argmax(probs, 1).item()
            clase_predicha = CLASES[clase_idx]
            confianza_max = probs[0][clase_idx].item() * 100
        reporte = (
            f"ANÁLISIS DE IA (EfficientNet-B4):\n"
            f"- Diagnóstico Computacional: {clase_predicha.upper()}\n"
            f"- Confianza Principal: {confianza_max:.2f}%\n"
            f"- Desglose de Probabilidades:\n"
            f"  * Benigno: {p_ben:.2f}%\n"
            f"  * Melanoma: {p_mel:.2f}%\n"
            f"  * Carcinoma: {p_car:.2f}%"
        )
        return reporte
    except Exception as e:
        return f"Error: {str(e)}"

# ==============================================================================
# 6. GENERADOR DE PDF CON FORMATO APA PARA MEJOR PRESENTACION
# ==============================================================================
class PDFReport(FPDF):
    def __init__(self, paciente_info):
        super().__init__()
        self.paciente_info = paciente_info

    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'DermaRAG - Informe Diagnóstico Asistido por IA', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Fecha de Emisión: {datetime.date.today()}', 0, 1, 'R')
        self.line(10, 30, 200, 30)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        # 1. PIE DE PAGINA CON DATOS DEL PACIENTE
        texto_pie = f"Paciente: {self.paciente_info['nombre']} | ID: {self.paciente_info['id']} | Edad: {self.paciente_info['edad']} | Página " + str(self.page_no())
        self.cell(0, 10, texto_pie, 0, 0, 'C')

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
# 7. INTERFAZ DE USUARIO STREAMLIT
# ==============================================================================

st.markdown("""
    <div class="header-container">
        <h1>🏥 DermaRAG - Sistema Multiagente de Diagnóstico Dermatológico</h1>
        <p>IA Explicable con Retrieval-Augmented Generation | Guías AAD/BAD/NCCN</p>
        <div style="background: #28a745; color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; font-size: 12px; margin-top: 10px; font-weight: bold;">MVP v1.0 - Prototipo Funcional</div>
    </div>
""", unsafe_allow_html=True)

# 1. CARGA MODELO 
RUTA_MODELO = 'mejor_modelo_v5.pth'
modelo_cnn = None
if os.path.exists(RUTA_MODELO):
    modelo_cnn = cargar_tu_modelo_especifico(RUTA_MODELO)
else:
    st.error(f"⚠️ Falta el archivo '{RUTA_MODELO}'")

# 2. GRID PRINCIPAL 
col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.markdown("## 📋 Datos del Paciente")
    
    c1, c2 = st.columns(2)
    with c1:
        nombre = st.text_input("Nombre Completo *", value="", placeholder="Ej. Gerardo García Pérez")
        edad = st.number_input("Edad *", value=0, min_value=0, max_value=120, step=1)
        fototipo = st.selectbox("Fototipo Fitzpatrick *", 
            ["Tipo I - Piel muy clara, siempre se quema", 
             "Tipo II - Piel clara, usualmente se quema", 
             "Tipo III - Piel intermedia, a veces se quema", 
             "Tipo IV - Piel morena clara, rara vez se quema", 
             "Tipo V - Piel morena, muy rara vez se quema", 
             "Tipo VI - Piel negra, nunca se quema"], 
            index=None, placeholder="Seleccionar opción...")
            
    with c2:
        id_paciente = st.text_input("ID Paciente *", value="", placeholder="Ej. PAC-2025-001")
        sexo = st.selectbox("Sexo *", ["Masculino", "Femenino", "Otro"], 
            index=None, placeholder="Seleccionar...")
        
    st.markdown("#### 🔬 Datos Clínicos de la Lesión")
    localizacion = st.selectbox("Localización Anatómica *", 
        ["Tronco (pecho/espalda)", "Cabeza y Cuello", "Extremidades Superiores", 
         "Extremidades Inferiores", "Manos/Pies (Acral)", "Mucosas"], 
        index=None, placeholder="Seleccionar ubicación...")
    
    cc1, cc2 = st.columns(2)
    tamano = cc1.number_input("Tamaño (mm) *", value=0, min_value=0, step=1)
    evolucion = cc2.number_input("Evolución (meses)", value=0, min_value=0, step=1)
    
    sintomas = st.text_area("Síntomas Asociados", 
        value="", placeholder="Ej. Prurito, sangrado, cambio de color, asimetría...", height=80)
        
    historia = st.text_area("Antecedentes Relevantes", 
        value="", placeholder="Ej. Historia familiar de melanoma, exposición solar crónica...", height=80)
    
    st.markdown("#### Criterios ABCDE (Dermoscopia Visual)")
    col_checks = st.columns(5)
    check_a = col_checks[0].checkbox("A", value=False, help="Asimetría")
    check_b = col_checks[1].checkbox("B", value=False, help="Bordes")
    check_c = col_checks[2].checkbox("C", value=False, help="Color")
    check_d = col_checks[3].checkbox("D", value=False, help="Diámetro")
    check_e = col_checks[4].checkbox("E", value=False, help="Evolución")

with col_der:
    st.markdown("## 📸 Imagen de la Lesión Cutánea")
    
    if 'uploaded_file' not in st.session_state:
        st.markdown("""
        <div style="border: 3px dashed #667eea; border-radius: 15px; padding: 30px; text-align: center; margin-bottom: 20px; background-color: #1f2937;">
            <div style="font-size: 48px; margin-bottom: 10px;">📷</div>
            <p style="color: #e6f3ff; font-weight: bold; margin: 0;">Haz clic para buscar o arrastra tu imagen</p>
            <p style="color: #9ca3af; font-size: 13px; margin-top: 5px;">Formatos: JPG, PNG | Máx. 10 MB</p>
        </div>
        """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        img_temp_pil = Image.open(uploaded_file)
        w, h = img_temp_pil.size
        size_kb = uploaded_file.size / 1024
        
        st.markdown(f"""
        <div style="background-color: #1a4023; border: 1px solid #166534; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; color: #4ade80; font-weight: bold; font-size: 16px; margin-bottom: 8px;">
                <span style="font-size: 20px; margin-right: 8px;">✓</span> Imagen cargada exitosamente
            </div>
            <div style="color: #dcfce7; font-size: 14px; line-height: 1.6;">
                <div>Archivo: <span style="color: #86efac; font-weight: 500;">{uploaded_file.name}</span></div>
                <div>Tamaño: <span style="color: #86efac; font-weight: 500;">{size_kb:.2f} KB</span></div>
                <div>Dimensiones: <span style="color: #86efac; font-weight: 500;">{w} x {h} px</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.image(img_temp_pil, caption="Vista Previa", use_container_width=True)

    analyze_btn = st.button("🔍 Analizar con IA Multiagente + GradCAM")

    st.markdown("""
    <div style="margin-top: 15px; padding: 15px; background-color: #0f2b45; border: 1px solid #1e40af; border-radius: 8px; color: #dbeafe; font-size: 13px;">
        <div style="font-weight: bold; margin-bottom: 8px; font-size: 14px; display: flex; align-items: center; color: #93c5fd;">
            <span style="margin-right: 5px; font-size: 16px;">ℹ️</span> Flujo del Sistema:
        </div>
        <div style="margin-bottom: 4px;">
            <span style="background-color: #1d4ed8; color: white; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px; margin-right: 5px;">1</span> 
            <strong>Agente Percepción:</strong> CNN (EfficientNet-B4) + GradCAM
        </div>
        <div style="margin-bottom: 4px;">
            <span style="background-color: #1d4ed8; color: white; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px; margin-right: 5px;">2</span> 
            <strong>Agente Investigación:</strong> RAG busca en guías AAD/BAD/NCCN
        </div>
        <div>
            <span style="background-color: #1d4ed8; color: white; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px; margin-right: 5px;">3</span> 
            <strong>Agente Síntesis:</strong> Mistral genera explicación pedagógica
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 8. LÓGICA DE AGENTES Y EJECUCIÓN
# ==============================================================================

if analyze_btn:
    if uploaded_file and modelo_cnn:
        
        if not nombre or localizacion is None:
            st.error("⚠️ Por favor completa al menos: Nombre y Localización.")
        else:
            with st.status("🔄 Ejecutando Sistema Multiagente...", expanded=True) as status:
                
                # 1. CREACION DIRECTORIO TEMPORAL PARA IMÁGENES
                temp_dir = tempfile.mkdtemp()
                
                # 2. GUARDAR IMAGEN UPLOADED
                with open(os.path.join(temp_dir, "input.jpg"), "wb") as f:
                    f.write(uploaded_file.getvalue())
                ruta_input = os.path.join(temp_dir, "input.jpg")

                # 3. VISION + GradCAM
                st.write("🧠 **Percepción Visual:** Ejecutando EfficientNet-B4 + Grad-CAM...")
                
                # 4. LLAMA A LA FUNCION DE GradCAM
                path_diag, path_bordes, path_patrones, pred_clase, probs = ejecutar_pipeline_gradcam(
                    modelo_cnn, ruta_input, temp_dir
                )
                
                # 5. REPORTE DE TEXTO PARA EL AGENTE
                resultado_vision = analizar_imagen_medica(ruta_input, modelo_cnn)
                
                st.write("✅ Análisis visual y mapas de calor generados.")
                
                # 6. CONFIGURACION DE AGENTES
                st.write("⚕️ **Razonamiento Clínico:** Iniciando agentes CrewAI...")
                os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
                os.environ["OPENAI_MODEL_NAME"] = "dolphin-mistral"
                os.environ["OPENAI_API_KEY"] = "ollama"

                # 7. PROMPT CONSTRUCTION
                abcde_sel = []
                if check_a: abcde_sel.append("Asimetría")
                if check_b: abcde_sel.append("Bordes Irregulares")
                if check_c: abcde_sel.append("Policromía")
                if check_d: abcde_sel.append(f"Diámetro > 6mm ({tamano}mm)")
                if check_e: abcde_sel.append("Evolución")
                hallazgos_txt = ", ".join(abcde_sel) if abcde_sel else "Sin hallazgos marcados"

                edad_val = edad
                tamano_val = tamano
                evolucion_val = evolucion
                sintomas_val = sintomas if sintomas else "No reportados"
                historia_val = historia if historia else "No reportados"
                # 8. HIPAA + GDPR , EN PDF SEGUIRA IMPRIMIENDO DATOS 
                task_med = (
                    f"DATOS DEL PACIENTE: Paciente Anónimo, {edad_val} años, {sexo if sexo else 'No esp.'}.\n"
                    f"FOTOTIPO: {fototipo if fototipo else 'No esp.'}.\n"
                    f"CLÍNICA: Lesión en {localizacion}, tamaño {tamano_val}mm, {evolucion_val} meses evolución.\n"
                    f"SÍNTOMAS: {sintomas_val}.\n"
                    f"ANTECEDENTES: {historia_val}.\n"
                    f"HALLAZGOS ABCDE MANUALES: {hallazgos_txt}.\n"
                    f"REPORTE DE LABORATORIO IA (EfficientNet-B4): [{resultado_vision}]."
                )

# ==============================================================================
# 9. DEFINICIÓN DE AGENTES Y TAREAS (OPTIMIZADO PARA GROQ LLM, RAG Y GRAD-CAM)
# ==============================================================================

                # 1. AGENTE INTERNISTA 
                medico_atencion_primaria = Agent(
                    role='Auditor Clínico de Inteligencia Artificial',
                    goal=(
                        f'Validar la coherencia entre el mapa de calor (Grad-CAM) y la clínica del paciente. '
                        f'Detectar alucinaciones de la visión artificial y preparar el caso con terminología médica precisa. '
                        f'Contexto del caso: {task_med}'
                    ),
                    backstory=(
                        'Eres un especialista en Diagnóstico por Imagen y Triaje. Tu trabajo NO es tratar, sino '
                        'auditar a la máquina. Tienes frente a ti un reporte de "EfficientNet-B4" y mapas de calor "Grad-CAM". '
                        'Tu filosofía es: "La IA es una herramienta, no un doctor". '
                        'Si la IA dice "Melanoma (90%)" pero el mapa de calor (Grad-CAM) está señalando piel sana o pelo '
                        'en lugar de la lesión, DEBES reportarlo como una "Posible Falsa Predicción". '
                        'Eres obsesivo con la semiología: transformas "le pica y sangra" en "prurito y ulceración activa".'
                    ),
                    verbose=True, 
                    allow_delegation=False
                )

                herramienta_rag = BuscadorGuiasClinicas()

                # 2. AGENTE ESPECIALISTA 
                especialista_piel = Agent(
                    role='Oncólogo Investigador Senior (Basado en Evidencia)',
                    goal=(
                        'Generar un plan de tratamiento donde CADA afirmación esté respaldada por una cita bibliográfica '
                        'extraída de la base de datos interna. NO usar conocimiento general externo.'
                    ),
                    backstory=(
                        'Eres el Jefe de Oncología Cutánea. Tienes prohibido dar opiniones personales. '
                        'Solo validas tratamientos que estén escritos en las Guías Clínicas (NCCN, SEOM, etc.) que tienes en tu base de datos. '
                        'Si recomiendas un margen de 1cm, debes decir: "Según NCCN 2024 (Pág 45)...". '
                        'Integras el riesgo visual calculado por tu colega auditor con la literatura médica estricta. '
                        'Tu tono es académico, frío y extremadamente preciso.'
                    ),
                    verbose=True, 
                    allow_delegation=False, 
                    tools=[herramienta_rag] 
                )

# ==============================================================================
# 10. TAREAS PARA AGENTES CREW IA & CREACION DE PDF EN APA
# ==============================================================================
                # 1. TAREA AGENTE #1
                task_atencion_primaria = Task(
                    description=f"""
                    Analiza los datos crudos del paciente y el reporte de la visión artificial: {task_med}.

                    REGLA DE ORO DE IDIOMA: TODA TU RESPUESTA DEBE SER 100% EN ESPAÑOL. ESTÁ ESTRICTAMENTE PROHIBIDO USAR INGLÉS.
                    
                    TU OBJETIVO ES REALIZAR UN RESUMEN CLÍNICO CON PRECISIÓN MATEMÁTICA Y SEMIOLÓGICA:
                    
                    1. **Fidelidad Absoluta a la IA:** - Tu diagnóstico de trabajo DEBE basarse exactamente en la predicción principal de la IA (EfficientNet-B4). 
                       - Si la IA marca una probabilidad abrumadora hacia "Carcinoma", tu redacción debe enfocarse en Carcinoma, sin sugerir Melanoma a menos que los datos clínicos sean contradictorios.
                    
                    2. **Traducción Semiológica y Lenguaje:**
                       - Convierte los síntomas del usuario a lenguaje médico formal y académico en español.
                       - Interpreta de forma lógica el mapa de calor (Grad-CAM).
                    
                    3. **Síntesis para el Especialista:**
                       - Redacta el estado actual del paciente cruzando el ABCDE manual con el resultado de la red neuronal.
                    """,
                    agent=medico_atencion_primaria,
                    expected_output="""
                    Reporte de Auditoría de IA (Markdown Español):
                    
                    ### 1. Validación de Visión Artificial
                    - **Diagnóstico Modelo:** [Clase principal dictada por la IA] (Confianza: [X]%).
                    - **Análisis de Atención (Grad-CAM):** [Interpretación clara de la zona de calor en Español].
                    - **Veredicto de Coherencia:** [Alta/Baja] [Explicación médica en Español].

                    ### 2. Resumen Semiológico
                    - **Perfil de Riesgo:** [Evaluación basada en fototipo, edad e historia].
                    - **Hallazgos Clínicos:** [Traducción técnica de los síntomas presentados].

                    ### 3. Solicitud de Interconsulta
                    - **Pregunta Clave:** [Pregunta específica para el oncólogo sobre el manejo a seguir].
                    """
                )
                # 2. TAREA AGENTE #2
                task_especialista = Task(
                    description="""
                    Recibes el análisis del auditor. AHORA DEBES BUSCAR LA EVIDENCIA CIENTÍFICA.
                    
                    PASOS OBLIGATORIOS:
                    1. **BÚSQUEDA AGRESIVA (RAG):**
                       - Usa la herramienta `Buscador de Guías Clínicas` MÚLTIPLES VECES.
                       - REGLA CRÍTICA PARA LA HERRAMIENTA: El término de búsqueda que envíes a la herramienta DEBE SER UNA SOLA CADENA DE TEXTO SIMPLE (simple string). ESTÁ ESTRICTAMENTE PROHIBIDO enviar formato JSON, diccionarios o listas. 
                         -> Correcto: "márgenes quirúrgicos carcinoma"
                         -> Incorrecto: {"query": "márgenes quirúrgicos carcinoma"}
                       - Busca explícitamente términos clínicos específicos.
                    
                    2. **CRUCE DE INFORMACIÓN Y CONTRASTE:**
                       - Busca en los documentos qué se hace EXACTAMENTE con ese diagnóstico.
                       - Contrasta al menos 2 documentos reales diferentes (ej. NCCN, AAD, SEOM).
                    
                    3. **REDACCIÓN BIBLIOGRÁFICA Y FORMATO (¡CRÍTICO!):**
                       - Redacta el plan de acción de forma nutrida y profesional EN ESPAÑOL.
                       - **REGLA DE CITACIÓN:** Cada frase médica debe terminar con una cita usando el nombre real del documento. Formato obligatorio: `[Fuente: Nombre Real de la Guía, Pág: X]`. 
                       - **REGLA DE ESTRUCTURA:** Debes respetar EXACTAMENTE el orden y los títulos de la plantilla solicitada. LA SECCIÓN DE REFERENCIAS BIBLIOGRÁFICAS ES OBLIGATORIA AL FINAL.
                       - **PROHIBICIONES:** ESTÁ PROHIBIDO usar bloques de código (```). ESTÁ PROHIBIDO generar tablas o cuadros. NO inventes nombres.
                       
                    ACLARACIÓN DE IDIOMA (¡MUY IMPORTANTE!):
                    Puedes usar INGLÉS para tu proceso lógico interno (Thought, Action, Action Input). SIN EMBARGO, tu RESPUESTA FINAL (Final Answer) debe ser 100% traducida y redactada en ESPAÑOL MÉDICO formal, rellenando la plantilla.
                    """,
                    agent=especialista_piel,
                    context=[task_atencion_primaria],
                    expected_output="""### 1. Diagnóstico e Interpretación
- **Diagnóstico de Trabajo:** [Enfermedad validada por el auditor en ESPAÑOL].
- **Justificación:** [Breve justificación clínica apoyada en la probabilidad IA en ESPAÑOL].

### 2. Protocolo de Manejo (Evidencia Contrastada)
- **Manejo Recomendado:** [Instrucción médica detallada en español]. [Fuente: Nombre Real del Documento, Pág: X]
- **Márgenes Quirúrgicos:** [Medida]. [Fuente: Nombre Real del Documento, Pág: X]. En contraste, la guía [Otro Nombre Real] sugiere [Medida alterna]. [Fuente: Otro Documento, Pág: Y]
- **Estudios Complementarios:** [Instrucción en español]. [Fuente: Nombre Real del Documento, Pág: X]

### 3. Pronóstico y Seguimiento
- **Pauta de Revisión:** [Frecuencia en español]. [Fuente: Nombre Real del Documento, Pág: X]

---
### 📚 Referencias Bibliográficas
1. **[Nombre completo de la Institución/Sociedad Médica]**. ([Año]). *[Título oficial y completo del documento o guía clínica]*. Base de conocimientos DermaRAG. Páginas consultadas: [X-Y].
2. **[Nombre completo de la 2da Institución/Sociedad Médica]**. ([Año]). *[Título oficial y completo del documento o guía clínica]*. Base de conocimientos DermaRAG. Páginas consultadas: [X-Y].

---
*Nota: Las citas provienen exclusivamente de la base de conocimiento cargada.*
"""
                )

                crew = Crew(
                    agents=[medico_atencion_primaria, especialista_piel],
                    tasks=[task_atencion_primaria, task_especialista],
                    verbose=True, process=Process.sequential, language='es'
                )
                
                resultado_final = crew.kickoff()
                status.update(label="✅ Diagnóstico Finalizado", state="complete", expanded=False)

            # 3. MOSTRAR RESULTADOS
            st.markdown("---")
            
            # 4. SECCIÓN VISUAL DE IA 
            st.subheader("👁️ Análisis de Explicabilidad (Grand-CAM)")
            tab1, tab2, tab3 = st.tabs(["Diagnóstico Clínico", "Análisis de Bordes (Capa Baja)", "Análisis de Patrones (Capa Alta)"])
            
            with tab1:
                st.image(path_diag, caption="Mapa de Atención y Probabilidades", use_container_width=True)
            with tab2:
                st.image(path_bordes, caption="Filtros de Capa Inicial (Detección de Bordes)", use_container_width=True)
            with tab3:
                st.image(path_patrones, caption="Filtros de Capa Final (Texturas Complejas)", use_container_width=True)

            st.markdown("### 📊 Informe Clínico Multiagente")
            
            with st.container(border=True):
                st.markdown(resultado_final)
                st.markdown("""
                <div class="disclaimer">
                    <strong>⚠️ AVISO MÉDICO-LEGAL:</strong> Este sistema es una herramienta de apoyo diagnóstico y 
                    <strong>NO reemplaza el juicio clínico profesional</strong>. 
                </div>
                """, unsafe_allow_html=True)

            # 5. GENERACIÓN Y DESCARGA DE PDF
            paciente_data = {
                'nombre': nombre,
                'id': id_paciente,
                'edad': f"{edad} años"
            }
            
            # 6. CREAR PDF
            pdf = PDFReport(paciente_data)
            pdf.add_page()
            
            # 7. TITULO
            pdf.chapter_title("1. Resumen Clínico e Imágenes")
            pdf.chapter_body(f"Paciente: {nombre}\nDiagnóstico IA: {pred_clase} (Confianza: {np.max(probs)*100:.2f}%)")
            
            # 8. INSERTAR IMAGENES, AJUSTAR COORDENADAS SEGUN NECESIDAD
            pdf.image(path_diag, x=10, y=None, w=190)
            pdf.ln(5)
            
            pdf.chapter_title("2. Informe Detallado de Agentes")
            # 9. LIMPIAR MARKDOWN BASICO PARA EL PDF 
            texto_limpio = str(resultado_final).replace('**', '').replace('### ', '\n\n')
            texto_limpio = texto_limpio.encode('latin-1', 'replace').decode('latin-1')
            pdf.chapter_body(texto_limpio)
            
            # 10. GUARDAR PDF TEMPORAL
            pdf_path = os.path.join(temp_dir, "reporte_dermarag.pdf")
            pdf.output(pdf_path)
            
            # 11. BOTON DESCARGA
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Exportar Reporte PDF (Formato APA)",
                    data=f,
                    file_name=f"Reporte_{id_paciente}.pdf",
                    mime="application/pdf"
                )

    else:
        st.warning("⚠️ Por favor sube una imagen para proceder.")

st.markdown("<div style='text-align: center; color: #9ca3af; padding: 20px;'>DermaRAG MVP v1.0 | Desarrollado con Mistral + EfficientNet-B4</div>", unsafe_allow_html=True)