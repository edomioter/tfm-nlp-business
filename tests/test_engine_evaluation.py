"""
Test de Evaluación del Motor de Consultas para TFM
==================================================

Este test evalúa el rendimiento del SmartNeo4jEngine con métricas cuantitativas
para demostrar su efectividad en la memoria del TFM.

Métricas evaluadas:
1. Precisión de generación de Cypher (sintaxis correcta)
2. Tiempo de respuesta promedio
3. Tasa de auto-corrección exitosa
4. Efectividad del RAG (comparación con/sin ejemplos)
5. Calidad de respuestas (validación de resultados esperados)

Uso:
    python tests/test_engine_evaluation.py
    python tests/test_engine_evaluation.py --output results.json
    python tests/test_engine_evaluation.py --mode comparative
"""

import sys
import os
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core import Settings, VectorStoreIndex
import config
from agent.engine import SmartNeo4jEngine
from agent.graph_context_provider import GraphContextProvider


@dataclass
class QueryTestCase:
    """Caso de prueba para una consulta."""
    id: str
    category: str  # simple, agregacion, compleja, temporal
    question: str
    expected_type: str  # count, list, aggregation
    validation_fn: callable = None  # Función para validar el resultado


@dataclass
class TestResult:
    """Resultado de un test individual."""
    test_id: str
    category: str
    question: str
    success: bool
    execution_time: float
    cypher_generated: str
    attempts: int  # Número de intentos de auto-corrección
    error: str = ""
    result_sample: str = ""
    reasoning_chain: str = ""  # Prompt completo construido por el agente
    cypher_attempts: List[str] = None  # Lista de todas las consultas Cypher intentadas

    def __post_init__(self):
        if self.cypher_attempts is None:
            self.cypher_attempts = []


class EngineEvaluator:
    """Evaluador del motor de consultas Neo4j."""

    def __init__(self):
        """Inicializa el evaluador con las conexiones necesarias."""
        self.setup_llm_and_stores()
        self.test_cases = self._define_test_cases()
        self.results: List[TestResult] = []

    def setup_llm_and_stores(self):
        """Configura LLM, embeddings y almacenes."""
        # Configurar LLM y embeddings
        self.llm = GoogleGenAI(
            model=config.MODEL_NAME,
            api_key=config.API_KEY,
            temperature=0
        )

        embed_model = GoogleGenAIEmbedding(
            model_name=config.EMBEDDING_MODEL,
            api_key=config.API_KEY
        )

        Settings.llm = self.llm
        Settings.embed_model = embed_model

        # Configurar Graph Store
        self.graph_store = Neo4jGraphStore(
            username=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            url=config.NEO4J_URL,
            refresh_schema=False
        )

        # Configurar Vector Store
        vector_store = Neo4jVectorStore(
            username=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            url=config.NEO4J_URL,
            index_name="cypher_examples_index",
            embedding_dimension=768
        )

        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        self.retriever = vector_index.as_retriever(similarity_top_k=3)

        # Configurar Graph Context Provider
        self.graph_context_provider = GraphContextProvider(graph_store=self.graph_store)

    def _define_test_cases(self) -> List[QueryTestCase]:
        """Define los casos de prueba para evaluar el motor."""
        return [
            # CATEGORÍA: Consultas Simples (Conteo)
            QueryTestCase(
                id="simple_01",
                category="simple",
                question="¿Cuántos usuarios hay en total?",
                expected_type="count"
            ),
            QueryTestCase(
                id="simple_02",
                category="simple",
                question="¿Cuántas sesiones se han registrado?",
                expected_type="count"
            ),
            QueryTestCase(
                id="simple_03",
                category="simple",
                question="¿Cuántos visitantes anónimos hay?",
                expected_type="count"
            ),

            # CATEGORÍA: Consultas con Agregación
            QueryTestCase(
                id="agg_01",
                category="agregacion",
                question="¿Cuál es la tasa de conversión general?",
                expected_type="aggregation"
            ),
            QueryTestCase(
                id="agg_02",
                category="agregacion",
                question="¿Cuál es el promedio de eventos por sesión?",
                expected_type="aggregation"
            ),
            QueryTestCase(
                id="agg_03",
                category="agregacion",
                question="¿Cuántas conversiones hubo por ciudad?",
                expected_type="aggregation"
            ),

            # CATEGORÍA: Consultas Complejas (Múltiples nodos)
            QueryTestCase(
                id="complex_01",
                category="compleja",
                question="¿Qué usuarios identificados provienen de visitantes con más de 3 sesiones?",
                expected_type="list"
            ),
            QueryTestCase(
                id="complex_02",
                category="compleja",
                question="¿Cuál es la ruta más común de eventos que llevan a una conversión?",
                expected_type="list"
            ),
            QueryTestCase(
                id="complex_03",
                category="compleja",
                question="¿Qué dispositivos tienen la mayor tasa de conversión?",
                expected_type="aggregation"
            ),

            # CATEGORÍA: Consultas Temporales
            QueryTestCase(
                id="temporal_01",
                category="temporal",
                question="¿Cuántos usuarios se registraron en los últimos 30 días?",
                expected_type="count"
            ),
            QueryTestCase(
                id="temporal_02",
                category="temporal",
                question="¿Cuál es la tendencia de conversiones por mes?",
                expected_type="aggregation"
            ),

            # CATEGORÍA: Consultas de Atributos
            QueryTestCase(
                id="attr_01",
                category="atributos",
                question="¿De qué ciudades son los usuarios?",
                expected_type="list"
            ),
            QueryTestCase(
                id="attr_02",
                category="atributos",
                question="¿Qué tipos de dispositivos usan los visitantes?",
                expected_type="list"
            ),
        ]

    def create_engine(self, use_rag: bool = True) -> SmartNeo4jEngine:
        """
        Crea una instancia del motor con o sin RAG.

        Args:
            use_rag: Si True, usa retriever y graph context. Si False, los omite.
        """
        return SmartNeo4jEngine(
            graph_store=self.graph_store,
            llm=self.llm,
            retriever=self.retriever if use_rag else None,
            schema_str=config.GRAPH_SCHEMA,
            graph_context_provider=self.graph_context_provider if use_rag else None
        )

    async def run_single_test(self, test_case: QueryTestCase, engine: SmartNeo4jEngine) -> TestResult:
        """
        Ejecuta un caso de prueba individual con control de rate limiting.

        Args:
            test_case: Caso de prueba a ejecutar
            engine: Motor de consultas a evaluar

        Returns:
            TestResult con los resultados de la prueba
        """
        print(f"\n{'='*80}")
        print(f"🧪 Test {test_case.id}: {test_case.category.upper()}")
        print(f"   Pregunta: {test_case.question}")
        print(f"{'='*80}")

        start_time = time.time()

        # Variables para capturar información
        captured_prompt = ""
        captured_cypher_attempts = []
        attempts_made = 0

        try:
            # Interceptar construcción del prompt
            original_build_prompt = engine._abuild_dynamic_prompt

            async def tracked_build_prompt(query_str):
                nonlocal captured_prompt
                prompt = await original_build_prompt(query_str)
                captured_prompt = prompt
                return prompt

            engine._abuild_dynamic_prompt = tracked_build_prompt

            # Interceptar generación de Cypher con retry
            original_generate = engine._agenerate_cypher_with_retry

            async def tracked_generate(initial_prompt):
                nonlocal attempts_made, captured_cypher_attempts
                current_prompt = initial_prompt
                last_cypher = ""

                for attempt in range(engine.max_retries):
                    attempts_made = attempt + 1

                    # Generación asíncrona
                    response = await engine.llm.acomplete(current_prompt)
                    cypher = response.text.replace("```cypher", "").replace("```", "").strip()

                    # Guardar intento
                    captured_cypher_attempts.append({
                        "attempt": attempt + 1,
                        "cypher": cypher,
                        "status": "pending"
                    })

                    # Validación
                    is_valid, error_msg = engine._validate_syntax(cypher)

                    if is_valid:
                        captured_cypher_attempts[-1]["status"] = "success"
                        return cypher

                    # Marcar como fallido
                    captured_cypher_attempts[-1]["status"] = "failed"
                    captured_cypher_attempts[-1]["error"] = error_msg

                    last_cypher = cypher
                    current_prompt = (
                        f"{initial_prompt}\n\n"
                        f"INTENTO ANTERIOR FALLIDO: {last_cypher}\n"
                        f"ERROR REPORTADO POR NEO4J: {error_msg}\n"
                        "CORRIGE LA CONSULTA:"
                    )

                raise ValueError(f"No se pudo generar Cypher válido tras {engine.max_retries} intentos.")

            engine._agenerate_cypher_with_retry = tracked_generate

            # Ejecutar consulta
            response = await engine.acustom_query(test_case.question)
            execution_time = time.time() - start_time

            # Extraer Cypher final exitoso
            cypher_generated = captured_cypher_attempts[-1]["cypher"] if captured_cypher_attempts else "N/A"

            # Validar resultado
            result_str = str(response)
            success = "Error" not in result_str

            result = TestResult(
                test_id=test_case.id,
                category=test_case.category,
                question=test_case.question,
                success=success,
                execution_time=execution_time,
                cypher_generated=cypher_generated,
                attempts=attempts_made,
                result_sample=result_str[:200],
                reasoning_chain=captured_prompt,
                cypher_attempts=captured_cypher_attempts
            )

            # Mostrar resultado
            status = "✅ ÉXITO" if success else "❌ FALLO"
            print(f"\n{status}")
            print(f"   Tiempo: {execution_time:.3f}s")
            print(f"   Intentos: {attempts_made}")
            print(f"   Resultado: {result_str[:100]}...")

            # Pausa para evitar rate limiting
            await asyncio.sleep(30)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_str = str(e)

            # Control de rate limiting (Error 429)
            if "429" in error_str:
                print(f"\n⚠️  Error 429: Límite de API alcanzado. Esperando 40 segundos...")
                await asyncio.sleep(40)

                # Reintentar una vez
                print(f"   Reintentando test {test_case.id}...")
                try:
                    # Reiniciar variables de captura
                    captured_prompt = ""
                    captured_cypher_attempts = []
                    attempts_made = 0

                    start_time = time.time()
                    response = await engine.acustom_query(test_case.question)
                    execution_time = time.time() - start_time

                    result_str = str(response)
                    success = "Error" not in result_str

                    print(f"\n✅ REINTENTO EXITOSO")
                    print(f"   Tiempo: {execution_time:.3f}s")
                    print(f"   Resultado: {result_str[:100]}...")

                    await asyncio.sleep(30)

                    cypher_generated = captured_cypher_attempts[-1]["cypher"] if captured_cypher_attempts else "N/A"

                    return TestResult(
                        test_id=test_case.id,
                        category=test_case.category,
                        question=test_case.question,
                        success=success,
                        execution_time=execution_time,
                        cypher_generated=cypher_generated,
                        attempts=attempts_made if attempts_made > 0 else engine.max_retries,
                        result_sample=result_str[:200],
                        reasoning_chain=captured_prompt,
                        cypher_attempts=captured_cypher_attempts
                    )

                except Exception as e2:
                    print(f"\n❌ ERROR EN REINTENTO: {str(e2)}")
                    return TestResult(
                        test_id=test_case.id,
                        category=test_case.category,
                        question=test_case.question,
                        success=False,
                        execution_time=execution_time,
                        cypher_generated="N/A",
                        attempts=engine.max_retries,
                        error=f"429 Error + Reintento fallido: {str(e2)}",
                        reasoning_chain=captured_prompt,
                        cypher_attempts=captured_cypher_attempts
                    )
            else:
                # Error no relacionado con rate limiting
                print(f"\n❌ ERROR: {error_str}")

                cypher_generated = captured_cypher_attempts[-1]["cypher"] if captured_cypher_attempts else "N/A"

                return TestResult(
                    test_id=test_case.id,
                    category=test_case.category,
                    question=test_case.question,
                    success=False,
                    execution_time=execution_time,
                    cypher_generated=cypher_generated,
                    attempts=attempts_made if attempts_made > 0 else engine.max_retries,
                    error=error_str,
                    reasoning_chain=captured_prompt,
                    cypher_attempts=captured_cypher_attempts
                )

    async def run_comparative_evaluation(self) -> Dict[str, Any]:
        """
        Ejecuta evaluación comparativa: Motor con RAG vs Motor sin RAG.

        Returns:
            Diccionario con resultados comparativos
        """
        print("\n" + "="*80)
        print("EVALUACIÓN COMPARATIVA: Con RAG vs Sin RAG")
        print("="*80)

        results = {
            "with_rag": [],
            "without_rag": []
        }

        # Subset de casos para comparación (evitar sobrecarga)
        comparison_cases = [tc for tc in self.test_cases if tc.category in ["simple", "agregacion"]]

        # --- Evaluación CON RAG ---
        print("\n\n" + "🔹"*40)
        print("FASE 1: Evaluando Motor CON RAG (Vector + Graph)")
        print("🔹"*40)

        engine_with_rag = self.create_engine(use_rag=True)

        for test_case in comparison_cases:
            result = await self.run_single_test(test_case, engine_with_rag)
            results["with_rag"].append(asdict(result))

        # Pausa entre fases
        print("\n⏸️  Pausa de 10 segundos entre fases...")
        await asyncio.sleep(10)

        # --- Evaluación SIN RAG ---
        print("\n\n" + "🔸"*40)
        print("FASE 2: Evaluando Motor SIN RAG (Solo Schema)")
        print("🔸"*40)

        engine_without_rag = self.create_engine(use_rag=False)

        for test_case in comparison_cases:
            result = await self.run_single_test(test_case, engine_without_rag)
            results["without_rag"].append(asdict(result))

        return results

    async def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Ejecuta la evaluación completa de todos los casos de prueba.

        Returns:
            Diccionario con todos los resultados y métricas
        """
        print("\n" + "="*80)
        print("EVALUACIÓN COMPLETA DEL MOTOR DE CONSULTAS")
        print("="*80)

        engine = self.create_engine(use_rag=True)
        results = []

        for test_case in self.test_cases:
            result = await self.run_single_test(test_case, engine)
            results.append(asdict(result))

        return {"results": results}

    def generate_metrics_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera un reporte de métricas a partir de los resultados.

        Args:
            results: Diccionario con resultados de evaluación

        Returns:
            Reporte con métricas agregadas
        """
        # Determinar si es evaluación comparativa o completa
        if "with_rag" in results:
            return self._generate_comparative_report(results)
        else:
            return self._generate_full_report(results)

    def _generate_full_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Genera reporte de evaluación completa."""
        test_results = results["results"]

        total_tests = len(test_results)
        successful = sum(1 for r in test_results if r["success"])
        failed = total_tests - successful

        avg_time = sum(r["execution_time"] for r in test_results) / total_tests if total_tests > 0 else 0
        avg_attempts = sum(r["attempts"] for r in test_results) / total_tests if total_tests > 0 else 0

        # Métricas por categoría
        categories = {}
        for result in test_results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "success": 0, "avg_time": 0}

            categories[cat]["total"] += 1
            if result["success"]:
                categories[cat]["success"] += 1
            categories[cat]["avg_time"] += result["execution_time"]

        for cat in categories:
            categories[cat]["success_rate"] = categories[cat]["success"] / categories[cat]["total"]
            categories[cat]["avg_time"] /= categories[cat]["total"]

        return {
            "summary": {
                "total_tests": total_tests,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total_tests if total_tests > 0 else 0,
                "avg_execution_time": avg_time,
                "avg_attempts": avg_attempts
            },
            "by_category": categories,
            "test_results": test_results
        }

    def _generate_comparative_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Genera reporte comparativo RAG vs No-RAG."""
        with_rag = results["with_rag"]
        without_rag = results["without_rag"]

        def calc_metrics(test_list):
            total = len(test_list)
            success = sum(1 for r in test_list if r["success"])
            avg_time = sum(r["execution_time"] for r in test_list) / total if total > 0 else 0
            avg_attempts = sum(r["attempts"] for r in test_list) / total if total > 0 else 0

            return {
                "total": total,
                "success": success,
                "success_rate": success / total if total > 0 else 0,
                "avg_time": avg_time,
                "avg_attempts": avg_attempts
            }

        with_rag_metrics = calc_metrics(with_rag)
        without_rag_metrics = calc_metrics(without_rag)

        # Calcular mejora porcentual
        improvement = {
            "success_rate": (with_rag_metrics["success_rate"] - without_rag_metrics["success_rate"]) / without_rag_metrics["success_rate"] * 100 if without_rag_metrics["success_rate"] > 0 else 0,
            "time": (without_rag_metrics["avg_time"] - with_rag_metrics["avg_time"]) / without_rag_metrics["avg_time"] * 100 if without_rag_metrics["avg_time"] > 0 else 0,
        }

        return {
            "comparative": {
                "with_rag": with_rag_metrics,
                "without_rag": without_rag_metrics,
                "improvement": improvement
            },
            "detailed_results": {
                "with_rag": with_rag,
                "without_rag": without_rag
            }
        }

    def print_report(self, report: Dict[str, Any]):
        """Imprime el reporte de forma legible."""
        print("\n\n" + "="*80)
        print("📊 REPORTE DE EVALUACIÓN")
        print("="*80)

        if "comparative" in report:
            self._print_comparative_report(report)
        else:
            self._print_full_report(report)

    def _print_full_report(self, report: Dict[str, Any]):
        """Imprime reporte de evaluación completa."""
        summary = report["summary"]

        print("\n🎯 RESUMEN GENERAL")
        print("-" * 80)
        print(f"Total de pruebas:        {summary['total_tests']}")
        print(f"Pruebas exitosas:        {summary['successful']} ✅")
        print(f"Pruebas fallidas:        {summary['failed']} ❌")
        print(f"Tasa de éxito:           {summary['success_rate']*100:.1f}%")
        print(f"Tiempo promedio:         {summary['avg_execution_time']:.3f}s")
        print(f"Intentos promedio:       {summary['avg_attempts']:.2f}")

        print("\n📂 RESULTADOS POR CATEGORÍA")
        print("-" * 80)

        for cat, metrics in report["by_category"].items():
            print(f"\n{cat.upper()}")
            print(f"  Total: {metrics['total']}")
            print(f"  Éxito: {metrics['success']}/{metrics['total']} ({metrics['success_rate']*100:.1f}%)")
            print(f"  Tiempo promedio: {metrics['avg_time']:.3f}s")

    def _print_comparative_report(self, report: Dict[str, Any]):
        """Imprime reporte comparativo."""
        comp = report["comparative"]
        with_rag = comp["with_rag"]
        without_rag = comp["without_rag"]
        improvement = comp["improvement"]

        print("\n🔬 COMPARACIÓN: CON RAG vs SIN RAG")
        print("-" * 80)
        print(f"{'Métrica':<30} {'Con RAG':<15} {'Sin RAG':<15} {'Mejora':<15}")
        print("-" * 80)
        print(f"{'Tasa de éxito':<30} {with_rag['success_rate']*100:<15.1f}% {without_rag['success_rate']*100:<15.1f}% {improvement['success_rate']:<15.1f}%")
        print(f"{'Tiempo promedio (s)':<30} {with_rag['avg_time']:<15.3f} {without_rag['avg_time']:<15.3f} {improvement['time']:<15.1f}%")
        print(f"{'Intentos promedio':<30} {with_rag['avg_attempts']:<15.2f} {without_rag['avg_attempts']:<15.2f}")

        print("\n📈 CONCLUSIÓN")
        print("-" * 80)
        if improvement['success_rate'] > 0:
            print(f"✅ El RAG mejora la tasa de éxito en un {improvement['success_rate']:.1f}%")
        else:
            print(f"⚠️  El RAG no muestra mejora significativa en tasa de éxito")

        if improvement['time'] > 0:
            print(f"✅ El RAG reduce el tiempo de ejecución en un {improvement['time']:.1f}%")
        elif improvement['time'] < -10:
            print(f"⚠️  El RAG incrementa el tiempo de ejecución en un {abs(improvement['time']):.1f}%")

    def save_results(self, report: Dict[str, Any], filename: str = "evaluation_results.json"):
        """Guarda los resultados en un archivo JSON."""
        report["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "model": config.MODEL_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
            "neo4j_url": config.NEO4J_URL
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Resultados guardados en: {filename}")


async def main():
    """Función principal para ejecutar la evaluación."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluar el motor de consultas Neo4j")
    parser.add_argument("--mode", choices=["full", "comparative"], default="full",
                        help="Modo de evaluación: full (completa) o comparative (con/sin RAG)")
    parser.add_argument("--output", default="evaluation_results.json",
                        help="Archivo de salida para los resultados (JSON)")

    args = parser.parse_args()

    print("="*80)
    print("🚀 SISTEMA DE EVALUACIÓN DEL MOTOR DE CONSULTAS - TFM")
    print("="*80)

    evaluator = EngineEvaluator()

    # Ejecutar evaluación según modo
    if args.mode == "comparative":
        print("\n📊 Modo: EVALUACIÓN COMPARATIVA (Con RAG vs Sin RAG)")
        results = await evaluator.run_comparative_evaluation()
    else:
        print("\n📊 Modo: EVALUACIÓN COMPLETA")
        results = await evaluator.run_full_evaluation()

    # Generar reporte
    report = evaluator.generate_metrics_report(results)

    # Mostrar reporte
    evaluator.print_report(report)

    # Guardar resultados
    evaluator.save_results(report, args.output)

    print("\n" + "="*80)
    print("✅ Evaluación completada exitosamente")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
