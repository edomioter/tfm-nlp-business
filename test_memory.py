#!/usr/bin/env python
"""
Script de prueba para verificar la memoria del agente.
Simula una conversación con preguntas de seguimiento.
"""
import sys
import asyncio
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
import config as config
from agent.agent_factory import create_graph_agent


async def test_memory():
    """Prueba la memoria del agente con múltiples intercambios."""
    print("=== TEST DE MEMORIA DEL AGENTE ===\n")

    # Configuración
    llm = GoogleGenAI(
        model=config.MODEL_NAME,
        api_key=config.API_KEY,
        temperature=0
    )

    embed_model = GoogleGenAIEmbedding(
        model_name=config.EMBEDDING_MODEL,
        api_key=config.API_KEY
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    graph_store = Neo4jGraphStore(
        username=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        url=config.NEO4J_URL,
        refresh_schema=False
    )

    vector_store = Neo4jVectorStore(
        username=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        url=config.NEO4J_URL,
        index_name="cypher_examples_index",
        embedding_dimension=768
    )

    # Crear agente y memoria
    agent = create_graph_agent(
        graph_store=graph_store,
        vector_store=vector_store,
        llm=llm,
        schema_str=config.GRAPH_SCHEMA
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    # Conversación de prueba
    test_conversation = [
        "Hola, ¿cuál es tu nombre?",
        "¿Qué tipo de datos puedes analizar?",
        "¿Recuerdas cómo te llamas?",
        "Si te preguntara por el número de usuarios, ¿usarías alguna herramienta?",
    ]

    print("Iniciando conversación de prueba:\n")

    for i, user_msg in enumerate(test_conversation, 1):
        print(f"[{i}] Usuario: {user_msg}")

        try:
            response = await agent.run(user_msg=user_msg, memory=memory)
            print(f"[{i}] Agente: {response}\n")

            # Pequeña pausa entre mensajes
            await asyncio.sleep(1)

        except Exception as e:
            print(f"[{i}] Error: {e}\n")
            if "429" in str(e):
                print("⚠️  Límite de API alcanzado. Esperando 40 segundos...")
                await asyncio.sleep(40)
                # Reintentar
                try:
                    response = await agent.run(user_msg=user_msg, memory=memory)
                    print(f"[{i}] Agente (reintento): {response}\n")
                except Exception as e2:
                    print(f"[{i}] Error en reintento: {e2}\n")

    print("\n=== VERIFICACIÓN DE MEMORIA ===")
    print("✓ Si el agente recordó su nombre en la pregunta 3, la memoria funciona correctamente.")
    print("✓ Si el agente mencionó en la pregunta 4 que usaría la herramienta neo4j_data_tool,")
    print("  demuestra que mantiene contexto sobre sus capacidades.")


if __name__ == "__main__":
    asyncio.run(test_memory())
