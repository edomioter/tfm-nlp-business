# seed.py
import sys
import os
import json

# --- 1. Configuración de Rutas (Path Resolution) ---
# Obtenemos la ruta absoluta de la carpeta donde está este archivo (seed.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

import config
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

def load_examples_from_json(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encuentra el archivo: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def seed_knowledge_base():
    print("--- Carga de Conocimiento desde JSON ---")
    
    # 1. Cargar desde JSON
    json_path = os.path.join(current_dir, "data", "examples.json")
    examples_data = load_examples_from_json(json_path)
    print(f"Leídos {len(examples_data)} ejemplos del archivo JSON.")

    documents = [
        Document(text=ex["question"], metadata={"cypher": ex["cypher"]})
        for ex in examples_data
    ]

    # 2. Configurar Modelo de Embedding
    embed_model = GoogleGenAIEmbedding(
        model_name=config.EMBEDDING_MODEL,
        api_key=config.API_KEY
    )

    # 3. Conectar y Poblar Neo4j Vector
    vector_store = Neo4jVectorStore(
        username=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        url=config.NEO4J_URL,
        index_name="cypher_examples_index",
        embedding_dimension=768
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    print("Indexación completada exitosamente.")

if __name__ == "__main__":
    seed_knowledge_base()