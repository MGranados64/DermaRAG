# ==============================================================================
# 1. LIBRERIAS PARA HERRAMIENTA DE INGESTA RAG
# ==============================================================================
import os
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
# ==============================================================================
# 3. BLOQUE DE SEGURIDAD PARA INPUTS 
# ==============================================================================
            # 1. Si el modelo envía un diccionario o JSON malformado en lugar de texto plano:
            if isinstance(query, dict):
                # 1.1 Intenta sacar el valor si viene en formato {'query': 'valor'}
                query = query.get('query', str(query))
                if isinstance(query, dict): # 1.2 Si sigue siendo dict 
                    query = query.get('description', str(query))
            
            # 2. Limpieza final de string por si quedan llaves sueltas
            query = str(query).replace("{'query':", "").replace("}", "").strip()

            # 3. Configuración del modelo de embeddings (Multilingüe)
            embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            # 4. Conexión a la DB
            db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
            
            # 5. RECUPERACIÓN AMPLIADA (k=10)
            results_raw = db.similarity_search(query, k=10)
            
            if not results_raw:
                return "No se encontró información relevante en las guías para esta consulta."

            # 6. DEFINICIÓN DE "BASURA" (FILTROS)
            frases_basura = [
                "End-User License Agreement",
                "All Rights Reserved",
                "Printed by",
                "PLEASE NOTE that use of this NCCN Content",
                "may not distribute this Content",
                "National Comprehensive Cancer Network, Inc.",
                "ME-D" 
            ]

            # 7. PROCESO DE LIMPIEZA Y FORMATEO
            contexto = f"RESULTADOS (FILTRADOS) DE LA BASE DE DATOS PARA: '{query}'\n\n"
            contador_validos = 0
            max_resultados_utiles = 4

            for doc in results_raw:
                contenido = doc.page_content
                
                # 7.1 Filtro de Contenido Prohibido
                if any(basura in contenido for basura in frases_basura):
                    continue
                
                # 7.2 Filtro de Longitud
                if len(contenido) < 50:
                    continue

                # 7.3 Si pasa los filtros, lo agregamos al reporte final
                fuente = doc.metadata.get('source', 'Guía desconocida')
                nombre_archivo = os.path.basename(fuente) 
                pagina = doc.metadata.get('page', '?')
                
                contexto += f"--- FRAGMENTO {contador_validos+1} (Fuente: {nombre_archivo}, Pág: {pagina}) ---\n"
                contexto += f"{contenido}\n\n"
                
                contador_validos += 1
                
                if contador_validos >= max_resultados_utiles:
                    break
            
            # 8. Validación final
            if contador_validos == 0:
                return "Se encontraron fragmentos, pero todos fueron descartados por ser texto legal (Disclaimers/Copyright)."

            return contexto

        except Exception as e:
            return f"Error al consultar la base de datos: {str(e)}"