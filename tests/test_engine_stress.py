"""
Test de Estrés del Motor de Consultas - TFM
============================================

Este test evalúa los LÍMITES del SmartNeo4jEngine con consultas extremadamente complejas
que desafían la capacidad del sistema. El objetivo es obtener una tasa de éxito realista
(60-80%) que demuestre tanto las fortalezas como las debilidades del motor.

Categorías de pruebas desafiantes:
1. Agregaciones multinivel (nested aggregations)
2. Análisis de rutas complejas (path analysis)
3. Consultas temporales con ventanas móviles
4. Análisis de cohortes y segmentación avanzada
5. Consultas ambiguas que requieren interpretación
6. Subconsultas complejas con múltiples JOINs
7. Análisis de funnels de conversión
8. Patrones opcionales y negaciones
9. Consultas con condiciones múltiples encadenadas
10. Análisis de comportamiento anómalo

Uso:
    python tests/test_engine_stress.py
    python tests/test_engine_stress.py --output stress_results.json
"""

import sys
import os
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any
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
class StressTestCase:
    """Caso de prueba de estrés."""
    id: str
    category: str
    difficulty: str  # "hard", "very_hard", "extreme"
    question: str
    expected_type: str
    notes: str = ""  # Por qué es difícil


@dataclass
class StressTestResult:
    """Resultado de un test de estrés."""
    test_id: str
    category: str
    difficulty: str
    question: str
    success: bool
    execution_time: float
    cypher_generated: str
    attempts: int
    error: str = ""
    result_sample: str = ""
    notes: str = ""
    reasoning_chain: str = ""  # Prompt completo construido por el agente
    cypher_attempts: List[Dict[str, Any]] = None  # Lista de todas las consultas Cypher intentadas

    def __post_init__(self):
        if self.cypher_attempts is None:
            self.cypher_attempts = []


class EngineStressTester:
    """Probador de estrés del motor de consultas."""

    def __init__(self):
        """Inicializa el probador con las conexiones necesarias."""
        self.setup_llm_and_stores()
        self.test_cases = self._define_stress_test_cases()
        self.results: List[StressTestResult] = []

    def setup_llm_and_stores(self):
        """Configura LLM, embeddings y almacenes."""
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

        # Graph Store
        self.graph_store = Neo4jGraphStore(
            username=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            url=config.NEO4J_URL,
            refresh_schema=False
        )

        # Vector Store
        vector_store = Neo4jVectorStore(
            username=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            url=config.NEO4J_URL,
            index_name="cypher_examples_index",
            embedding_dimension=768
        )

        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        self.retriever = vector_index.as_retriever(similarity_top_k=3)

        # Graph Context Provider
        self.graph_context_provider = GraphContextProvider(graph_store=self.graph_store)

    def _define_stress_test_cases(self) -> List[StressTestCase]:
        """Define casos de prueba extremadamente desafiantes."""
        return [
            # ===== CATEGORÍA: Agregaciones Multinivel =====
            StressTestCase(
                id="stress_agg_01",
                category="agregacion_multinivel",
                difficulty="hard",
                question="¿Cuál es el promedio de eventos por sesión para usuarios que han realizado al menos una conversión, agrupado por ciudad y dispositivo?",
                expected_type="aggregation",
                notes="Requiere múltiples niveles de agregación y filtrado"
            ),
            StressTestCase(
                id="stress_agg_02",
                category="agregacion_multinivel",
                difficulty="very_hard",
                question="De los visitantes que se convirtieron en usuarios, ¿cuántas sesiones promedio tuvieron antes y después de su identificación?",
                expected_type="aggregation",
                notes="Requiere partición temporal basada en evento de identificación"
            ),
            StressTestCase(
                id="stress_agg_03",
                category="agregacion_multinivel",
                difficulty="extreme",
                question="¿Cuál es la tasa de conversión por cohorte mensual de registro, considerando solo conversiones que ocurrieron dentro de los primeros 30 días?",
                expected_type="aggregation",
                notes="Requiere análisis de cohortes con ventanas temporales"
            ),

            # ===== CATEGORÍA: Análisis de Rutas =====
            StressTestCase(
                id="stress_path_01",
                category="analisis_rutas",
                difficulty="hard",
                question="¿Cuántos pasos promedio hay entre la primera sesión de un visitante y su primera conversión?",
                expected_type="aggregation",
                notes="Requiere cálculo de longitud de path"
            ),
            StressTestCase(
                id="stress_path_02",
                category="analisis_rutas",
                difficulty="very_hard",
                question="¿Qué secuencia de tipos de eventos (page_view, click, etc.) es más probable que lleve a una conversión, considerando los 5 eventos previos?",
                expected_type="list",
                notes="Requiere análisis de secuencias y patrones"
            ),
            StressTestCase(
                id="stress_path_03",
                category="analisis_rutas",
                difficulty="extreme",
                question="Identifica usuarios que visitaron desde mobile, luego desktop, y finalmente convirtieron desde mobile nuevamente",
                expected_type="list",
                notes="Requiere pattern matching de secuencias específicas"
            ),

            # ===== CATEGORÍA: Funnels de Conversión =====
            StressTestCase(
                id="stress_funnel_01",
                category="funnel_conversion",
                difficulty="hard",
                question="¿Cuál es la tasa de abandono en cada paso del funnel: visita → sesión con eventos → sesión con más de 3 eventos → conversión?",
                expected_type="aggregation",
                notes="Requiere construcción de funnel con drop-off rates"
            ),
            StressTestCase(
                id="stress_funnel_02",
                category="funnel_conversion",
                difficulty="very_hard",
                question="¿En qué punto del customer journey (número de sesión) ocurren la mayoría de las conversiones?",
                expected_type="aggregation",
                notes="Requiere enumerar sesiones por usuario y analizar patrones"
            ),

            # ===== CATEGORÍA: Análisis Temporal Complejo =====
            StressTestCase(
                id="stress_temporal_01",
                category="temporal_complejo",
                difficulty="hard",
                question="¿Cuál es la diferencia en días promedio entre el registro de un usuario y su primera conversión?",
                expected_type="aggregation",
                notes="Requiere cálculo de diferencias temporales"
            ),
            StressTestCase(
                id="stress_temporal_02",
                category="temporal_complejo",
                difficulty="very_hard",
                question="¿Qué día de la semana tiene la mayor tasa de conversión, considerando solo sesiones que comenzaron en horario laboral (9am-6pm)?",
                expected_type="aggregation",
                notes="Requiere extracción de día de semana y hora"
            ),
            StressTestCase(
                id="stress_temporal_03",
                category="temporal_complejo",
                difficulty="extreme",
                question="Identifica usuarios con comportamiento estacional: que tienen picos de actividad cada 7 días aproximadamente",
                expected_type="list",
                notes="Requiere análisis de periodicidad"
            ),

            # ===== CATEGORÍA: Consultas Ambiguas =====
            StressTestCase(
                id="stress_ambig_01",
                category="ambiguedad",
                difficulty="hard",
                question="¿Qué usuarios son más valiosos?",
                expected_type="list",
                notes="Ambiguo: ¿valiosos por conversiones? ¿por engagement? Requiere interpretación"
            ),
            StressTestCase(
                id="stress_ambig_02",
                category="ambiguedad",
                difficulty="very_hard",
                question="Muéstrame los usuarios problemáticos",
                expected_type="list",
                notes="Muy ambiguo: ¿problemáticos cómo? ¿muchas sesiones sin conversión? ¿bounce alto?"
            ),
            StressTestCase(
                id="stress_ambig_03",
                category="ambiguedad",
                difficulty="extreme",
                question="¿Hay algún patrón raro en los datos?",
                expected_type="list",
                notes="Extremadamente abierto, requiere análisis exploratorio"
            ),

            # ===== CATEGORÍA: Negaciones y Patrones Opcionales =====
            StressTestCase(
                id="stress_neg_01",
                category="negacion",
                difficulty="hard",
                question="¿Qué visitantes nunca se identificaron pero tienen más de 5 sesiones?",
                expected_type="list",
                notes="Requiere negación con EXISTS"
            ),
            StressTestCase(
                id="stress_neg_02",
                category="negacion",
                difficulty="very_hard",
                question="¿Qué usuarios se registraron pero nunca han realizado una conversión en los últimos 60 días?",
                expected_type="list",
                notes="Requiere negación con filtro temporal"
            ),

            # ===== CATEGORÍA: Análisis de Segmentación =====
            StressTestCase(
                id="stress_seg_01",
                category="segmentacion",
                difficulty="hard",
                question="Agrupa usuarios en tres segmentos: alto valor (3+ conversiones), medio valor (1-2 conversiones), bajo valor (0 conversiones) y cuenta cuántos hay en cada uno",
                expected_type="aggregation",
                notes="Requiere CASE statements o múltiples agregaciones"
            ),
            StressTestCase(
                id="stress_seg_02",
                category="segmentacion",
                difficulty="very_hard",
                question="¿Qué características tienen en común los usuarios del top 10% en términos de engagement (número de eventos)?",
                expected_type="aggregation",
                notes="Requiere percentiles y análisis comparativo"
            ),

            # ===== CATEGORÍA: Consultas Multi-Concepto =====
            StressTestCase(
                id="stress_multi_01",
                category="multi_concepto",
                difficulty="very_hard",
                question="¿Cuál es la combinación de ciudad y dispositivo con mejor ROI, definido como tasa de conversión multiplicada por número de eventos promedio?",
                expected_type="aggregation",
                notes="Combina múltiples métricas en cálculo personalizado"
            ),
            StressTestCase(
                id="stress_multi_02",
                category="multi_concepto",
                difficulty="extreme",
                question="Identifica usuarios que muestran comportamiento de 'power user': más de 10 sesiones, promedio de más de 5 eventos por sesión, al menos una conversión, y han usado tanto mobile como desktop",
                expected_type="list",
                notes="Requiere múltiples condiciones complejas en diferentes niveles"
            ),
        ]

    def create_engine(self) -> SmartNeo4jEngine:
        """Crea una instancia del motor."""
        return SmartNeo4jEngine(
            graph_store=self.graph_store,
            llm=self.llm,
            retriever=self.retriever,
            schema_str=config.GRAPH_SCHEMA,
            graph_context_provider=self.graph_context_provider
        )

    async def run_single_test(self, test_case: StressTestCase, engine: SmartNeo4jEngine) -> StressTestResult:
        """Ejecuta un caso de prueba individual."""
        print(f"\n{'='*80}")
        print(f"🔥 Test {test_case.id}: {test_case.difficulty.upper()}")
        print(f"   Categoría: {test_case.category}")
        print(f"   Pregunta: {test_case.question}")
        print(f"   Desafío: {test_case.notes}")
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

            result = StressTestResult(
                test_id=test_case.id,
                category=test_case.category,
                difficulty=test_case.difficulty,
                question=test_case.question,
                success=success,
                execution_time=execution_time,
                cypher_generated=cypher_generated,
                attempts=attempts_made,
                result_sample=result_str[:200],
                notes=test_case.notes,
                reasoning_chain=captured_prompt,
                cypher_attempts=captured_cypher_attempts
            )

            # Mostrar resultado
            status = "✅ ÉXITO" if success else "❌ FALLO"
            print(f"\n{status}")
            print(f"   Tiempo: {execution_time:.3f}s")
            print(f"   Resultado: {result_str[:150]}...")

            # Pausa para evitar rate limiting
            await asyncio.sleep(30)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_str = str(e)

            # Manejo de rate limiting
            if "429" in error_str:
                print(f"\n⚠️  Error 429: Esperando 40 segundos...")
                await asyncio.sleep(40)

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
                    await asyncio.sleep(30)

                    cypher_generated = captured_cypher_attempts[-1]["cypher"] if captured_cypher_attempts else "N/A"

                    return StressTestResult(
                        test_id=test_case.id,
                        category=test_case.category,
                        difficulty=test_case.difficulty,
                        question=test_case.question,
                        success=success,
                        execution_time=execution_time,
                        cypher_generated=cypher_generated,
                        attempts=attempts_made if attempts_made > 0 else engine.max_retries,
                        result_sample=result_str[:200],
                        notes=test_case.notes,
                        reasoning_chain=captured_prompt,
                        cypher_attempts=captured_cypher_attempts
                    )

                except Exception as e2:
                    print(f"\n❌ ERROR EN REINTENTO: {str(e2)}")

            print(f"\n❌ ERROR: {error_str}")

            cypher_generated = captured_cypher_attempts[-1]["cypher"] if captured_cypher_attempts else "N/A"

            return StressTestResult(
                test_id=test_case.id,
                category=test_case.category,
                difficulty=test_case.difficulty,
                question=test_case.question,
                success=False,
                execution_time=execution_time,
                cypher_generated=cypher_generated,
                attempts=attempts_made if attempts_made > 0 else engine.max_retries,
                error=error_str,
                notes=test_case.notes,
                reasoning_chain=captured_prompt,
                cypher_attempts=captured_cypher_attempts
            )

    async def run_stress_evaluation(self) -> Dict[str, Any]:
        """Ejecuta la evaluación de estrés completa."""
        print("\n" + "="*80)
        print("🔥 EVALUACIÓN DE ESTRÉS DEL MOTOR DE CONSULTAS")
        print("="*80)

        engine = self.create_engine()
        results = []

        for test_case in self.test_cases:
            result = await self.run_single_test(test_case, engine)
            results.append(asdict(result))

        return {"results": results}

    def generate_metrics_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Genera un reporte de métricas."""
        test_results = results["results"]

        total_tests = len(test_results)
        successful = sum(1 for r in test_results if r["success"])
        failed = total_tests - successful

        avg_time = sum(r["execution_time"] for r in test_results) / total_tests if total_tests > 0 else 0

        # Métricas por dificultad
        by_difficulty = {}
        for result in test_results:
            diff = result["difficulty"]
            if diff not in by_difficulty:
                by_difficulty[diff] = {"total": 0, "success": 0, "avg_time": 0}

            by_difficulty[diff]["total"] += 1
            if result["success"]:
                by_difficulty[diff]["success"] += 1
            by_difficulty[diff]["avg_time"] += result["execution_time"]

        for diff in by_difficulty:
            by_difficulty[diff]["success_rate"] = by_difficulty[diff]["success"] / by_difficulty[diff]["total"]
            by_difficulty[diff]["avg_time"] /= by_difficulty[diff]["total"]

        # Métricas por categoría
        by_category = {}
        for result in test_results:
            cat = result["category"]
            if cat not in by_category:
                by_category[cat] = {"total": 0, "success": 0, "avg_time": 0}

            by_category[cat]["total"] += 1
            if result["success"]:
                by_category[cat]["success"] += 1
            by_category[cat]["avg_time"] += result["execution_time"]

        for cat in by_category:
            by_category[cat]["success_rate"] = by_category[cat]["success"] / by_category[cat]["total"]
            by_category[cat]["avg_time"] /= by_category[cat]["total"]

        return {
            "summary": {
                "total_tests": total_tests,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total_tests if total_tests > 0 else 0,
                "avg_execution_time": avg_time
            },
            "by_difficulty": by_difficulty,
            "by_category": by_category,
            "test_results": test_results
        }

    def print_report(self, report: Dict[str, Any]):
        """Imprime el reporte."""
        print("\n\n" + "="*80)
        print("📊 REPORTE DE EVALUACIÓN DE ESTRÉS")
        print("="*80)

        summary = report["summary"]

        print("\n🎯 RESUMEN GENERAL")
        print("-" * 80)
        print(f"Total de pruebas:        {summary['total_tests']}")
        print(f"Pruebas exitosas:        {summary['successful']} ✅")
        print(f"Pruebas fallidas:        {summary['failed']} ❌")
        print(f"Tasa de éxito:           {summary['success_rate']*100:.1f}%")
        print(f"Tiempo promedio:         {summary['avg_execution_time']:.3f}s")

        print("\n🔥 RESULTADOS POR DIFICULTAD")
        print("-" * 80)

        for diff, metrics in report["by_difficulty"].items():
            print(f"\n{diff.upper()}")
            print(f"  Total: {metrics['total']}")
            print(f"  Éxito: {metrics['success']}/{metrics['total']} ({metrics['success_rate']*100:.1f}%)")
            print(f"  Tiempo promedio: {metrics['avg_time']:.3f}s")

        print("\n📂 RESULTADOS POR CATEGORÍA")
        print("-" * 80)

        for cat, metrics in report["by_category"].items():
            print(f"\n{cat.upper()}")
            print(f"  Total: {metrics['total']}")
            print(f"  Éxito: {metrics['success']}/{metrics['total']} ({metrics['success_rate']*100:.1f}%)")
            print(f"  Tiempo promedio: {metrics['avg_time']:.3f}s")

    def save_results(self, report: Dict[str, Any], filename: str = "stress_test_results.json"):
        """Guarda los resultados en un archivo JSON."""
        report["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "stress_test",
            "model": config.MODEL_NAME,
            "embedding_model": config.EMBEDDING_MODEL,
            "neo4j_url": config.NEO4J_URL
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Resultados guardados en: {filename}")


async def main():
    """Función principal."""
    import argparse

    parser = argparse.ArgumentParser(description="Test de estrés del motor de consultas")
    parser.add_argument("--output", default="stress_test_results.json",
                        help="Archivo de salida para los resultados (JSON)")

    args = parser.parse_args()

    print("="*80)
    print("🔥 SISTEMA DE TEST DE ESTRÉS - TFM")
    print("="*80)

    tester = EngineStressTester()

    # Ejecutar evaluación de estrés
    results = await tester.run_stress_evaluation()

    # Generar reporte
    report = tester.generate_metrics_report(results)

    # Mostrar reporte
    tester.print_report(report)

    # Guardar resultados
    tester.save_results(report, args.output)

    print("\n" + "="*80)
    print("✅ Evaluación de estrés completada")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
