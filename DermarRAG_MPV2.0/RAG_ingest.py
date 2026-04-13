# ==============================================================================
# 1. LIBRERIAS PARA HERRAMIENTA DE INGESTA RAG
# ==============================================================================
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

RUTA_DOCS = "./guias_clinicas"
RUTA_DB = "./chroma_db"
# ==============================================================================
# 2. DEFINICION DE FUNCION DE INGESTA
# ==============================================================================
def crear_vectorstore():
    # 1. Limpieza previa si existe
    if os.path.exists(RUTA_DB):
        shutil.rmtree(RUTA_DB)

    print("📚 Cargando guías clínicas (Inglés)...")
    loader = PyPDFDirectoryLoader(RUTA_DOCS)
    docs = loader.load()
    
    # 2. Chunking: Aumentamos un poco el tamaño para tener más contexto, BATCH SIZE
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)
    print(f"🧩 Se generaron {len(splits)} fragmentos.")

    # 3. MODELO MULTILINGÜE 
    print("🧠 Descargando modelo multilingüe...")
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("💾 Indexando vectores...")
    Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=RUTA_DB
    )
    print("✅ ¡Base de datos lista para cruzar Español <-> Inglés!")

if __name__ == "__main__":
    crear_vectorstore()