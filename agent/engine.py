# agent/engine.py
from typing import Any, Tuple, Optional
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.retrievers import BaseRetriever
from neo4j.exceptions import CypherSyntaxError, ClientError
from .graph_context_provider import GraphContextProvider


#    Motor SOTA que combina:
#    1. RAG Dinámico (busca ejemplos en Neo4j Vector).
#    2. Prompting Adaptativo (cambia según contexto).
#    3. Self-Healing (corrige su propio código Cypher).
#    4. GraphRAG (contexto estadístico del grafo).

class SmartNeo4jEngine(CustomQueryEngine):

    graph_store: Neo4jGraphStore
    llm: Any
    retriever: BaseRetriever  # Interfaz genérica para buscar ejemplos
    schema_str: str
    max_retries: int = 3
    graph_context_provider: Optional[GraphContextProvider] = None  # GraphRAG provider

    def _validate_syntax(self, cypher: str) -> Tuple[bool, str]:
        # Valida la sintaxis usando EXPLAIN sin ejecutar.
        driver = self.graph_store._driver
        try:
            with driver.session() as session:
                session.run(f"EXPLAIN {cypher}")
            return True, ""
        except (CypherSyntaxError, ClientError) as e:
            return False, str(e)
        except Exception as e:
            return False, f"System Error: {e}"



    #   Construye el prompt. Si el retriever encuentra ejemplos útiles,
    #   los inyecta. Si no, omite la sección de ejemplos por completo.
    #   Ahora también integra contexto del grafo (GraphRAG).
    def _build_dynamic_prompt(self, query_str: str) -> str:
            from config import BASE_CYPHER_TEMPLATE

            # 1. Recuperar ejemplos semánticamente similares (Top-k) - Vector RAG
            nodes = self.retriever.retrieve(query_str)

            # 2. Formatear ejemplos (si existen)
            examples_block = ""
            if nodes:
                examples_text = "\n".join(
                    [f"Pregunta: {n.text}\nCypher: {n.metadata['cypher']}" for n in nodes]
                )
                examples_block = (
                    f"--- EJEMPLOS DE REFERENCIA (Úsalos como guía de estilo y lógica) ---\n"
                    f"{examples_text}\n"
                    f"-------------------------------------------------------------------\n\n"
                )

            # 3. Recuperar contexto del grafo (GraphRAG)
            graph_context_block = ""
            if self.graph_context_provider:
                try:
                    context = self.graph_context_provider.get_business_context(query_str)
                    graph_context_block = self.graph_context_provider.format_context_for_prompt(context)
                    graph_context_block += "\n"
                except Exception as e:
                    print(f"[GraphRAG] Error recuperando contexto: {e}")

            # 4. Ensamblar Prompt Final usando template centralizado
            prompt = BASE_CYPHER_TEMPLATE.format(
                schema=self.schema_str,
                graph_context=graph_context_block,
                examples=examples_block,
                query_str=query_str
            )
            return prompt

    async def _abuild_dynamic_prompt(self, query_str: str) -> str:
            """Versión asíncrona de _build_dynamic_prompt."""
            from config import BASE_CYPHER_TEMPLATE

            # 1. Recuperar ejemplos semánticamente similares (Top-k) - Asíncrono - Vector RAG
            nodes = await self.retriever.aretrieve(query_str)

            # 2. Formatear ejemplos (si existen)
            examples_block = ""
            if nodes:
                examples_text = "\n".join(
                    [f"Pregunta: {n.text}\nCypher: {n.metadata['cypher']}" for n in nodes]
                )
                examples_block = (
                    f"--- EJEMPLOS DE REFERENCIA (Úsalos como guía de estilo y lógica) ---\n"
                    f"{examples_text}\n"
                    f"-------------------------------------------------------------------\n\n"
                )

            # 3. Recuperar contexto del grafo (GraphRAG)
            graph_context_block = ""
            if self.graph_context_provider:
                try:
                    context = self.graph_context_provider.get_business_context(query_str)
                    graph_context_block = self.graph_context_provider.format_context_for_prompt(context)
                    graph_context_block += "\n"
                except Exception as e:
                    print(f"[GraphRAG] Error recuperando contexto: {e}")

            # 4. Ensamblar Prompt Final usando template centralizado
            prompt = BASE_CYPHER_TEMPLATE.format(
                schema=self.schema_str,
                graph_context=graph_context_block,
                examples=examples_block,
                query_str=query_str
            )
            return prompt

    #   Bucle de Auto-Corrección (Self-Healing Loop)
    def _generate_cypher_with_retry(self, initial_prompt: str) -> str:
            current_prompt = initial_prompt
            last_cypher = ""

            for attempt in range(self.max_retries):
                # A. Generación
                response = self.llm.complete(current_prompt).text
                # Limpieza
                cypher = response.replace("```cypher", "").replace("```", "").strip()

                # B. Validación
                is_valid, error_msg = self._validate_syntax(cypher)

                if is_valid:
                    return cypher # Éxito

                # C. Refinamiento (Si falla)
                print(f"[Self-Correction] Intento {attempt+1} falló: {error_msg}")
                last_cypher = cypher

                # Actualizamos el prompt con el error para que el LLM lo arregle
                current_prompt = (
                    f"{initial_prompt}\n\n"
                    f"INTENTO ANTERIOR FALLIDO: {last_cypher}\n"
                    f"ERROR REPORTADO POR NEO4J: {error_msg}\n"
                    "CORRIGE LA CONSULTA:"
                )

            raise ValueError(f"No se pudo generar Cypher válido tras {self.max_retries} intentos.")

    async def _agenerate_cypher_with_retry(self, initial_prompt: str) -> str:
            """Versión asíncrona del bucle de auto-corrección."""
            current_prompt = initial_prompt
            last_cypher = ""

            for attempt in range(self.max_retries):
                # A. Generación asíncrona
                response = await self.llm.acomplete(current_prompt)
                # Limpieza
                cypher = response.text.replace("```cypher", "").replace("```", "").strip()

                # B. Validación
                is_valid, error_msg = self._validate_syntax(cypher)

                if is_valid:
                    return cypher # Éxito

                # C. Refinamiento (Si falla)
                print(f"[Self-Correction] Intento {attempt+1} falló: {error_msg}")
                last_cypher = cypher

                # Actualizamos el prompt con el error para que el LLM lo arregle
                current_prompt = (
                    f"{initial_prompt}\n\n"
                    f"INTENTO ANTERIOR FALLIDO: {last_cypher}\n"
                    f"ERROR REPORTADO POR NEO4J: {error_msg}\n"
                    "CORRIGE LA CONSULTA:"
                )

            raise ValueError(f"No se pudo generar Cypher válido tras {self.max_retries} intentos.")

    def custom_query(self, query_str: str) -> Response:
            try:
                # 1. Construcción Inteligente del Prompt
                prompt = self._build_dynamic_prompt(query_str)

                # 2. Generación y Validación Robusta
                final_cypher = self._generate_cypher_with_retry(prompt)
                print(f"[DEBUG] Executing Cypher: {final_cypher}")

                # 3. Ejecución
                results = self.graph_store.query(final_cypher)

                # 4. Síntesis
                return Response(response=str(results))

            except Exception as e:
                return Response(response=f"Error procesando solicitud: {str(e)}")

    async def acustom_query(self, query_str: str) -> Response:
            """Versión asíncrona del método custom_query para compatibilidad con agentes async."""
            try:
                # 1. Construcción Inteligente del Prompt (Asíncrona)
                prompt = await self._abuild_dynamic_prompt(query_str)

                # 2. Generación y Validación Robusta (Asíncrona)
                final_cypher = await self._agenerate_cypher_with_retry(prompt)
                print(f"[DEBUG] Executing Cypher: {final_cypher}")

                # 3. Ejecución
                results = self.graph_store.query(final_cypher)

                # 4. Síntesis
                return Response(response=str(results))

            except Exception as e:
                return Response(response=f"Error procesando solicitud: {str(e)}")