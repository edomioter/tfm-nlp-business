# agent/agent_factory.py
from typing import Any
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

from .engine import SmartNeo4jEngine

#  Construye un Agente ReAct con memoria y acceso a la herramienta de Neo4j.
def create_graph_agent(graph_store, vector_store, llm, schema_str):
        
    # 1. Preparar el Retriever (Buscador de ejemplos)
    from llama_index.core import VectorStoreIndex
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    retriever = vector_index.as_retriever(similarity_top_k=3)

    # 2. Instancia del Motor Inteligente (La lógica compleja de Cypher)
    smart_engine = SmartNeo4jEngine(
        graph_store=graph_store,
        llm=llm,
        retriever=retriever,
        schema_str=schema_str
    )

    # 3. Encapsular el Motor como una HERRAMIENTA
    # La descripción es CRÍTICA: le dice al LLM CUÁNDO usar esto.
    graph_tool = QueryEngineTool(
        query_engine=smart_engine,
        metadata=ToolMetadata(
            name="neo4j_data_tool",
            description=(
                "Útil SOLAMENTE para consultas que requieren DATOS cuantitativos o "
                "relaciones específicas de la base de datos de la empresa (ventas, clientes, leads). "
                "NO usar para preguntas generales, saludos, o para explicar razonamientos anteriores."
            ),
        ),
    )

    # 4. CONSTRUCCIÓN DEL AGENTE
    # Nota: ReActAgent maneja el estado y memoria internamente como parte del workflow
    agent = ReActAgent(
        name="Business Data Analyst",
        description="Analista de datos con acceso a Neo4j",
        tools=[graph_tool],
        llm=llm,
        verbose=True, # Para ver el razonamiento en consola
        system_prompt=(
            "Eres un analista de datos senior asistiendo a un director de negocio. "
            "Tu nombre es 'DataBot' y estás especializado en análisis de comportamiento de usuarios y métricas de conversión. "
            "Tienes acceso a una base de datos Neo4j con información sobre usuarios, sesiones, eventos y conversiones "
            "a través de la herramienta 'neo4j_data_tool'.\n\n"
            "Reglas de comportamiento:\n"
            "1. SIEMPRE mantén el contexto de la conversación. Recuerda las consultas previas y sus resultados.\n"
            "2. Si el usuario pide datos (números, listas, rutas, métricas), USA la herramienta.\n"
            "3. Si el usuario pide explicaciones sobre una respuesta anterior, clarificaciones, o hace preguntas de seguimiento, "
            "NO uses la herramienta - usa tu memoria de la conversación y razonamiento.\n"
            "4. Puedes realizar análisis complejos que requieran múltiples pasos. El usuario puede hacer preguntas de seguimiento "
            "y tú debes entender el contexto acumulado.\n"
            "5. Sé profesional, conciso y útil.\n"
            "6. Si es una conversación casual o presentación, responde amablemente sin usar herramientas."
        )
    )
    return agent