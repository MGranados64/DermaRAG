# ==============================================================================
# 0. PARCHE OBLIGATORIO PARA HUGGING FACE SPACES (ChromaDB)
# ==============================================================================
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# ==============================================================================
# 1. LIBRERIAS PARA HERRAMIENTA DE INGESTA RAG
# ==============================================================================
import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==============================================================================
# 2. ESQUEMA DE ENTRADA 
# ==============================================================================
class BuscadorInput(BaseModel):
    query: str = Field(..., description="La frase exacta a buscar en las guías médicas. Ej: 'margen melanoma'.")

# ==============================================================================
# 3. DEFINICION DE FUNCION DE GUIAS CLINICAS
# ==============================================================================
class BuscadorGuiasClinicas(BaseTool):
    name: str = "buscador_guias_clinicas"
    description: str = (
        "OBLIGATORIO: Úsala SIEMPRE antes de responder cualquier pregunta sobre diagnóstico, "
        "protocolo, márgenes, dosis, estadificación o seguimiento oncológico. "
        "Es tu ÚNICA fuente válida de información médica. "
        "ENTRADA: una frase corta en español. Ej: 'margen melanoma', 'dosis nivolumab', "
        "'estadificación carcinoma basocelular'."
    )
    args_schema: Type[BaseModel] = BuscadorInput 

    def _run(self, query: str) -> str:
        try:
            # 1. Cargar Embeddings
            embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            # 2. Conectar a ChromaDB
            db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

            try:
                total_docs = db._collection.count()
                if total_docs == 0:
                    return ("ERROR CRÍTICO: La base ChromaDB existe pero está VACÍA. "
                            "Verifica que la carpeta ./chroma_db esté subida al Space "
                            "con los archivos .sqlite3 y los binarios de embeddings.")
            except Exception:
                pass

            # MULTI-QUERY EXPANSION
            # Generamos variantes de la query con sinónimos médicos para mejorar recall.
            # Esto ayuda especialmente cuando el agente usa términos en español pero las
            # guías mezclan español/inglés.
            sinonimos = {
                "tratamiento": ["manejo", "terapia", "treatment"],
                "márgenes": ["margen", "resección", "exéresis", "margins"],
                "seguimiento": ["control", "vigilancia", "follow-up", "follow up"],
                "cirugía": ["quirúrgico", "surgery", "surgical"],
                "carcinoma": ["cáncer", "tumor", "neoplasia"],
                "melanoma": ["lesión melanocítica", "tumor melanocítico"],
                "estadificación": ["estadiaje", "staging", "clasificación"],
                "recurrencia": ["recidiva", "recurrence"],
                "diagnóstico": ["evaluación", "diagnosis"],
                "biopsia": ["biopsy", "muestra histológica"],
            }

            queries_a_buscar = [query]  
            query_lower = query.lower()
            for clave, alts in sinonimos.items():
                if clave in query_lower:
                    # Generar 1 variante reemplazando con el primer sinónimo
                    queries_a_buscar.append(query_lower.replace(clave, alts[0]))
                    if len(queries_a_buscar) >= 3:
                        break

            results_raw = []
            seen_keys = set()  

            for q in queries_a_buscar:
                try:
                    # MMR: balancea relevancia y diversidad. lambda_mult=0.5 = balance
                    docs = db.max_marginal_relevance_search(
                        q, k=8, fetch_k=20, lambda_mult=0.5
                    )
                except Exception:
                    # Fallback a búsqueda normal si MMR no está disponible
                    docs = db.similarity_search(q, k=8)

                for doc in docs:
                    source = doc.metadata.get('source', '?')
                    page = doc.metadata.get('page', '?')
                    key = (source, page)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        results_raw.append(doc)

            if not results_raw:
                return "No se encontró información relevante en las guías para esta consulta."

            # 3. Filtros de "Basura" Legal y de formato
            frases_basura = [
                "End-User License Agreement", "All Rights Reserved", "Printed by",
                "PLEASE NOTE that use of this NCCN Content", "may not distribute this Content",
                "National Comprehensive Cancer Network, Inc.", "ME-D",
                "FUCS", "INC - CDFLLA", "ISBN", "Copyright", "www.",
                "http://", "https://", "@gmail", "@hotmail",
            ]

            # 4. Procesar Resultados con DIVERSIDAD POR FUENTE (round-robin)
            # Agrupamos los fragmentos válidos por archivo fuente
            por_fuente = {}
            for doc in results_raw:
                contenido = doc.page_content

                if any(basura in contenido for basura in frases_basura):
                    continue
                if len(contenido) < 150:
                    continue

                num_digitos = sum(c.isdigit() for c in contenido)
                if num_digitos / max(len(contenido), 1) > 0.15:
                    continue

                palabras = [p for p in contenido.split() if len(p) > 3]
                if len(palabras) < 25:
                    continue

                fuente = doc.metadata.get('source', 'desconocida')
                nombre_archivo = os.path.basename(fuente.replace('\\', '/'))
                if nombre_archivo not in por_fuente:
                    por_fuente[nombre_archivo] = []
                por_fuente[nombre_archivo].append(doc)

            # Round-robin: tomar 1 fragmento de cada fuente, luego otro, etc.
            # Esto garantiza diversidad de PDFs en lugar de saturar con uno solo.
            fragmentos_finales = []
            max_resultados_utiles = 6
            while len(fragmentos_finales) < max_resultados_utiles and any(por_fuente.values()):
                for fuente in list(por_fuente.keys()):
                    if por_fuente[fuente]:
                        fragmentos_finales.append(por_fuente[fuente].pop(0))
                        if len(fragmentos_finales) >= max_resultados_utiles:
                            break

            if not fragmentos_finales:
                return "Se encontraron fragmentos, pero todos fueron descartados por ser texto legal o disclaimers."

            # 5. Construir el contexto final
            contexto = f"RESULTADOS DE LA BASE DE DATOS PARA: '{query}'\n"
            contexto += f"(Búsqueda multi-query con expansión de sinónimos, fragmentos diversificados por fuente)\n\n"

            for i, doc in enumerate(fragmentos_finales):
                contenido = doc.page_content
                fuente = doc.metadata.get('source', 'Guía desconocida')
                nombre_archivo = os.path.basename(fuente.replace('\\', '/'))
                pagina = doc.metadata.get('page', '?')

                # Guardado para RAGas
                try:
                    with open("memoria_rag.txt", "a", encoding="utf-8") as f:
                        f.write(contenido + "\n\n")
                except Exception:
                    pass

                contexto += f"--- FRAGMENTO {i+1} (Fuente: {nombre_archivo}, Pág: {pagina}) ---\n"
                contexto += f"{contenido}\n\n"

            return contexto

        except Exception as e:
            return f"Error al consultar la base de datos: {str(e)}"