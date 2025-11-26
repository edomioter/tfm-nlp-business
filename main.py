# main.py
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

async def main():
    print("--- Sistema Agéntico TFM (Chat + Graph RAG) ---")

    try:
        # 1. Infraestructura (Modelos y Conexiones)
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

        # 2. Creación del Agente
        agent = create_graph_agent(
            graph_store=graph_store,
            vector_store=vector_store,
            llm=llm,
            schema_str=config.GRAPH_SCHEMA
        )

        # 3. Crear Memoria Persistente con límite de tokens
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    except Exception as e:
        print(f"Error de inicio: {e}")
        sys.exit(1)

    print("\n[INFO] Agente listo con memoria y acceso a datos.")
    print("Prueba: '¿Cuántos clientes hay?' y luego '¿Por qué dijiste eso?'\n")

    while True:
        user_input = input("Usuario >> ")
        if user_input.lower() in ["salir", "exit"]:
            break

        try:
            # run() ejecuta el workflow del agente de forma asíncrona
            # Pasamos la memoria para mantener el contexto entre conversaciones
            response = await agent.run(user_msg=user_input, memory=memory)
            print(f"Agente >> {response}\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())