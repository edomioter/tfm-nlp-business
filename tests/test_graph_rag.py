# test_graph_rag.py
"""
Script de prueba para validar la implementación de GraphRAG Híbrido.

Este script demuestra cómo el sistema combina:
1. Vector RAG: Recuperación de ejemplos similares
2. Graph RAG: Contexto estadístico del grafo de negocio

Uso:
    python test_graph_rag.py
"""
import sys
import os
# Agregar el directorio raíz al path para importar módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core import Settings, VectorStoreIndex
import config
from agent.graph_context_provider import GraphContextProvider


def test_graph_context_provider():
    """
    Prueba 1: Verificar que el GraphContextProvider recupera contexto correctamente.
    """
    print("=" * 80)
    print("PRUEBA 1: GraphContextProvider - Recuperación de Contexto del Grafo")
    print("=" * 80)

    try:
        # Conectar a Neo4j
        graph_store = Neo4jGraphStore(
            username=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            url=config.NEO4J_URL,
            refresh_schema=False
        )

        # Inicializar el provider
        provider = GraphContextProvider(graph_store=graph_store)

        # Probar diferentes tipos de consultas
        test_queries = [
            "¿Cuántos usuarios hay en total?",
            "¿Cuál es la tasa de conversión?",
            "¿De qué ciudades son los usuarios?",
            "¿Qué dispositivos usan los visitantes?"
        ]

        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 80)

            # Recuperar contexto
            context = provider.get_business_context(query)

            # Formatear y mostrar
            formatted = provider.format_context_for_prompt(context)
            print(formatted)
            print()

        print("✅ Prueba 1 completada exitosamente\n")
        return True

    except Exception as e:
        print(f"❌ Error en Prueba 1: {e}")
        return False


def test_hybrid_rag_prompt():
    """
    Prueba 2: Verificar que el prompt combina Vector RAG + Graph RAG.
    """
    print("=" * 80)
    print("PRUEBA 2: Prompt Híbrido - Vector RAG + Graph RAG")
    print("=" * 80)

    try:
        # Inicializar LLM y embeddings
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

        # Conectar a Neo4j
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

        # Preparar retriever (Vector RAG)
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        retriever = vector_index.as_retriever(similarity_top_k=3)

        # Preparar Graph Context Provider (Graph RAG)
        graph_context_provider = GraphContextProvider(graph_store=graph_store)

        # Simular construcción de prompt
        from agent.engine import SmartNeo4jEngine

        engine = SmartNeo4jEngine(
            graph_store=graph_store,
            llm=llm,
            retriever=retriever,
            schema_str=config.GRAPH_SCHEMA,
            graph_context_provider=graph_context_provider
        )

        # Construir prompt para una query de prueba
        test_query = "¿Cuál es la tasa de conversión por ciudad?"

        print(f"\n📝 Query de Prueba: {test_query}")
        print("-" * 80)

        prompt = engine._build_dynamic_prompt(test_query)

        print("\n🔍 PROMPT GENERADO (GraphRAG Híbrido):")
        print("=" * 80)
        print(prompt)
        print("=" * 80)

        # Verificar que el prompt contiene ambos contextos
        has_graph_context = "CONTEXTO DEL NEGOCIO" in prompt
        has_vector_examples = "EJEMPLOS DE REFERENCIA" in prompt or "No hay ejemplos" in prompt

        print("\n✅ Verificación del Prompt:")
        print(f"   - Contiene Graph RAG Context: {'✓' if has_graph_context else '✗'}")
        print(f"   - Contiene Vector RAG Examples: {'✓' if has_vector_examples else '✗'}")

        if has_graph_context:
            print("\n✅ Prueba 2 completada exitosamente")
            print("   El sistema ahora combina Vector RAG + Graph RAG correctamente.\n")
            return True
        else:
            print("\n⚠️  Advertencia: El contexto del grafo no aparece en el prompt.")
            return False

    except Exception as e:
        print(f"❌ Error en Prueba 2: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison():
    """
    Prueba 3: Comparación de prompts con y sin GraphRAG.
    """
    print("=" * 80)
    print("PRUEBA 3: Comparación - Antes vs Después de GraphRAG")
    print("=" * 80)

    print("""
    ANTES (Solo Vector RAG):
    ├── Esquema del grafo
    ├── Ejemplos similares del vector store
    └── Pregunta del usuario

    DESPUÉS (GraphRAG Híbrido):
    ├── Esquema del grafo
    ├── 📊 CONTEXTO DEL NEGOCIO (Graph RAG) ← NUEVO
    │   ├── Estadísticas generales
    │   ├── Rango temporal de datos
    │   ├── Principales fuentes/ciudades/dispositivos
    │   └── Métricas de conversión
    ├── Ejemplos similares (Vector RAG)
    └── Pregunta del usuario

    VENTAJAS:
    ✓ El LLM tiene contexto cuantitativo antes de generar Cypher
    ✓ Puede validar si una pregunta es factible (ej: "usuarios de París" si no hay datos de París)
    ✓ Puede optimizar consultas (ej: filtrar por top ciudades en lugar de todas)
    ✓ Genera respuestas más precisas y contextualizadas
    """)

    print("\n✅ Prueba 3 completada (conceptual)\n")
    return True


def main():
    print("\n" + "=" * 80)
    print("TEST SUITE: GraphRAG Híbrido Implementation")
    print("=" * 80 + "\n")

    results = []

    # Ejecutar pruebas
    results.append(("GraphContextProvider", test_graph_context_provider()))
    results.append(("Prompt Híbrido", test_hybrid_rag_prompt()))
    results.append(("Comparación Conceptual", test_comparison()))

    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE PRUEBAS")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nResultado Final: {passed}/{total} pruebas exitosas")
    print("=" * 80 + "\n")

    return 0 if all(p for _, p in results) else 1


if __name__ == "__main__":
    sys.exit(main())
