# Tests de Evaluación del Motor de Consultas

Este directorio contiene los tests diseñados para evaluar el rendimiento y efectividad del `SmartNeo4jEngine` en el contexto del TFM.

## Archivos de Test

### 1. `test_engine_evaluation.py`

Test principal de evaluación cuantitativa del motor de consultas. Diseñado específicamente para generar métricas que puedan incluirse en la memoria del TFM.

#### Características

- **Control de Rate Limiting**: Maneja automáticamente errores 429 de la API con esperas y reintentos
- **Evaluación por Categorías**: Organiza tests por complejidad (simple, agregación, compleja, temporal, atributos)
- **Métricas Cuantitativas**:
  - Tasa de éxito en generación de Cypher
  - Tiempo promedio de respuesta
  - Número de intentos de auto-corrección
  - Comparación con/sin RAG

#### Uso

```bash
# Evaluación completa (todos los casos de prueba)
python tests/test_engine_evaluation.py

# Evaluación comparativa (con RAG vs sin RAG)
python tests/test_engine_evaluation.py --mode comparative

# Guardar resultados en archivo específico
python tests/test_engine_evaluation.py --output mis_resultados.json

# Combinación de opciones
python tests/test_engine_evaluation.py --mode comparative --output comparativa_rag.json
```

#### Opciones de Línea de Comandos

- `--mode`: Modo de evaluación
  - `full`: Evaluación completa de todos los casos de prueba (por defecto)
  - `comparative`: Comparación entre motor con RAG vs sin RAG
- `--output`: Nombre del archivo JSON para guardar resultados (por defecto: `evaluation_results.json`)

#### Casos de Prueba Incluidos

El test evalúa **13 casos de prueba** organizados en 5 categorías:

1. **Consultas Simples (3 casos)**
   - Conteo de usuarios
   - Conteo de sesiones
   - Conteo de visitantes anónimos

2. **Consultas con Agregación (3 casos)**
   - Tasa de conversión general
   - Promedio de eventos por sesión
   - Conversiones por ciudad

3. **Consultas Complejas (3 casos)**
   - Usuarios identificados con múltiples sesiones
   - Rutas de eventos hacia conversión
   - Dispositivos con mayor tasa de conversión

4. **Consultas Temporales (2 casos)**
   - Usuarios registrados en últimos 30 días
   - Tendencia de conversiones por mes

5. **Consultas de Atributos (2 casos)**
   - Ciudades de usuarios
   - Tipos de dispositivos

#### Métricas Reportadas

El sistema genera un reporte JSON con las siguientes métricas:

```json
{
  "summary": {
    "total_tests": 13,
    "successful": 12,
    "failed": 1,
    "success_rate": 0.923,
    "avg_execution_time": 2.45,
    "avg_attempts": 1.1
  },
  "by_category": {
    "simple": {
      "total": 3,
      "success": 3,
      "success_rate": 1.0,
      "avg_time": 1.8
    },
    ...
  }
}
```

#### Control de Rate Limiting (429 Errors)

El test incluye manejo automático de límites de API:

1. **Detección**: Captura errores 429 de Google Generative AI
2. **Espera**: Pausa de 40 segundos antes de reintentar
3. **Reintento**: Ejecuta el test nuevamente
4. **Pausas Preventivas**: 2-3 segundos entre tests, 10 segundos entre fases

Esto es especialmente útil para planes de billing limitados.

#### Formato de Salida

El test produce:

1. **Salida por consola**: Progreso en tiempo real con emojis y formato claro
2. **Archivo JSON**: Resultados estructurados para análisis posterior
3. **Reporte de métricas**: Tabla comparativa y conclusiones

Ejemplo de salida por consola:

```
================================================================================
🧪 Test simple_01: SIMPLE
   Pregunta: ¿Cuántos usuarios hay en total?
================================================================================

✅ ÉXITO
   Tiempo: 1.234s
   Intentos: 1
   Resultado: [{'count': 150}]...

⏸️  Pausa para evitar rate limiting...
```

### 2. `test_graph_rag.py`

Test de validación de la implementación de GraphRAG híbrido (Vector RAG + Graph RAG).

#### Uso

```bash
python tests/test_graph_rag.py
```

#### Pruebas Incluidas

1. **GraphContextProvider**: Verifica recuperación de contexto estadístico del grafo
2. **Prompt Híbrido**: Valida integración de Vector RAG + Graph RAG en el prompt
3. **Comparación Conceptual**: Explica diferencias antes/después de GraphRAG

### 3. `test_memory.py`

Test de verificación del sistema de memoria conversacional del agente.

#### Uso

```bash
python tests/test_memory.py
```

#### Características

- Simula conversación multi-turno
- Verifica que el agente recuerde contexto previo
- Incluye control de rate limiting

## Recomendaciones para Ejecución

### Para TFM / Presentación

Si necesitas generar métricas para tu memoria o presentación:

```bash
# Ejecuta la evaluación comparativa
python tests/test_engine_evaluation.py --mode comparative --output resultados_tfm.json
```

Esto genera un archivo JSON con métricas cuantitativas que puedes:
- Incluir en tablas de tu memoria
- Usar para crear gráficas (Excel, Python, R)
- Citar en la sección de resultados experimentales

### Para Desarrollo / Debug

Si estás desarrollando o debuggeando el motor:

```bash
# Evaluación completa con todos los casos
python tests/test_engine_evaluation.py --output debug.json
```

### Consideraciones de Rate Limiting

Si tienes un plan de billing limitado:

1. Ejecuta tests durante horas de bajo uso
2. Usa el modo `comparative` primero (menos casos)
3. El sistema maneja automáticamente los 429, pero puede tardar más
4. Considera ejecutar subconjuntos de tests modificando `_define_test_cases()`

## Estructura de Resultados JSON

```json
{
  "summary": {
    "total_tests": int,
    "successful": int,
    "failed": int,
    "success_rate": float,
    "avg_execution_time": float,
    "avg_attempts": float
  },
  "by_category": {
    "categoria": {
      "total": int,
      "success": int,
      "success_rate": float,
      "avg_time": float
    }
  },
  "test_results": [
    {
      "test_id": str,
      "category": str,
      "question": str,
      "success": bool,
      "execution_time": float,
      "cypher_generated": str,
      "attempts": int,
      "error": str,
      "result_sample": str
    }
  ],
  "metadata": {
    "timestamp": str,
    "model": str,
    "embedding_model": str,
    "neo4j_url": str
  }
}
```

## Troubleshooting

### Error: "Module not found"

Asegúrate de ejecutar desde el directorio raíz del proyecto:

```bash
cd /workspaces/tfm-nlp-business
python tests/test_engine_evaluation.py
```

### Error: "Connection refused" (Neo4j)

Verifica que Neo4j esté corriendo y las credenciales en `.env` sean correctas:

```bash
# Verificar variables de entorno
cat .env | grep NEO4J
```

### Error: "API Key invalid"

Verifica tu API key de Google Generative AI en `.env`:

```bash
cat .env | grep LLM_API_KEY
```

### Demasiados errores 429

Opciones:
1. Espera a que se resetee tu cuota (usualmente 1 minuto)
2. Reduce el número de casos de prueba
3. Aumenta las pausas entre tests (edita línea 295: `await asyncio.sleep(2)` → `await asyncio.sleep(5)`)

## Contribuir

Para agregar nuevos casos de prueba:

1. Edita `_define_test_cases()` en `test_engine_evaluation.py`
2. Agrega un nuevo `QueryTestCase` con:
   - `id`: Identificador único
   - `category`: Categoría del test
   - `question`: Pregunta en lenguaje natural
   - `expected_type`: Tipo de resultado esperado
3. Ejecuta el test para validar

Ejemplo:

```python
QueryTestCase(
    id="custom_01",
    category="personalizada",
    question="¿Cuál es mi nueva métrica?",
    expected_type="aggregation"
)
```
