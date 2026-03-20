# ==============================================================================
# 1. LIBRERIAS PARA HERRAMIENTA DE INGESTA RAG
# ==============================================================================
import os
import streamlit as st 
from crewai.tools import BaseTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==============================================================================
# 2. DEFINICION DE FUNCION DE GUIAS CLINICAS
# ==============================================================================
class BuscadorGuiasClinicas(BaseTool):
    name: str = "Buscador de Guías Clínicas"
    description: str = (
        "Úsala para buscar en documentos médicos PDF. "
        "ENTRADA: Solo escribe la frase de lo que buscas. Ej: 'margen melanoma', 'dosis nivolumab'. "
        "No uses JSON ni formatos complejos."
    )

    def _run(self, query: str) -> str:
        try:
            # --- NUEVO: Inicializar memoria RAG en la sesión ---
            if 'contextos_rag' not in st.session_state:
                st.session_state['contextos_rag'] = []
            # ---------------------------------------------------

# ==============================================================================
# 3. BLOQUE DE SEGURIDAD PARA INPUTS 
# ==============================================================================
            if isinstance(query, dict):
                query = query.get('query', str(query))
                if isinstance(query, dict): 
                    query = query.get('description', str(query))
            
            query = str(query).replace("{'query':", "").replace("}", "").strip()

            embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
            results_raw = db.similarity_search(query, k=10)
            
            if not results_raw:
                return "No se encontró información relevante en las guías para esta consulta."

            frases_basura = [
                "End-User License Agreement", "All Rights Reserved", "Printed by",
                "PLEASE NOTE that use of this NCCN Content", "may not distribute this Content",
                "National Comprehensive Cancer Network, Inc.", "ME-D" 
            ]

            contexto = f"RESULTADOS (FILTRADOS) DE LA BASE DE DATOS PARA: '{query}'\n\n"
            contador_validos = 0
            max_resultados_utiles = 4

            for doc in results_raw:
                contenido = doc.page_content
                
                # Filtros
                if any(basura in contenido for basura in frases_basura):
                    continue
                if len(contenido) < 50:
                    continue

                # --- NUEVO: Guardar el texto limpio para RAGas ---
                st.session_state['contextos_rag'].append(contenido)
                # -------------------------------------------------

                fuente = doc.metadata.get('source', 'Guía desconocida')
                nombre_archivo = os.path.basename(fuente) 
                pagina = doc.metadata.get('page', '?')
                
                contexto += f"--- FRAGMENTO {contador_validos+1} (Fuente: {nombre_archivo}, Pág: {pagina}) ---\n"
                contexto += f"{contenido}\n\n"
                
                contador_validos += 1
                if contador_validos >= max_resultados_utiles:
                    break
            
            if contador_validos == 0:
                return "Se encontraron fragmentos, pero todos fueron descartados por ser texto legal (Disclaimers/Copyright)."

            return contexto

        except Exception as e:
            return f"Error al consultar la base de datos: {str(e)}"