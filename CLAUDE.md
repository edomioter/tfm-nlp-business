# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's Thesis project implementing a conversational assistant for business data analysis and monitoring. The system uses a RAG (Retrieval-Augmented Generation) architecture with Neo4j graph database for analyzing user behavior, conversions, and business metrics.

## Environment Setup

### Required Environment Variables (.env)

The project requires a `.env` file at the root with the following variables:
- `LLM_API_KEY`: API key for the LLM provider (Google Generative AI)
- `LLM_MODEL`: Model name (e.g., "gemini-1.5-pro")
- `EMBEDDING_MODEL`: Embedding model name (e.g., "models/embedding-001")
- `NEO4J_URL`: Neo4j database URL (e.g., "bolt://localhost:7687")
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password

All variables are validated at startup by `config.py`. Missing required variables will raise `ValueError` with a descriptive message.

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Database Initialization

```bash
# Seed the vector store with example Cypher queries
python seed.py
```

This loads examples from `data/examples.json` and creates embeddings in Neo4j's vector store under the index `cypher_examples_index`.

### Running the Application

```bash
python main.py
```

The interactive chat interface will start. Type queries in Spanish, or type `salir`/`exit` to quit.

## Architecture

### Core Components

1. **Configuration Layer** (`config.py`)
   - Loads environment variables via `python-dotenv`
   - Defines the Neo4j graph schema (`GRAPH_SCHEMA`) with node types (User, Visitor, Session, Event, Conversion) and relationships
   - Contains prompt templates for Cypher generation (`CYPHER_GEN_TEMPLATE`)

2. **Agent Factory** (`agent/agent_factory.py`)
   - Creates a `BaseWorkflowAgent` (ReAct pattern) with chat memory
   - Wraps the SmartNeo4jEngine as a tool with specific instructions on when to use it
   - Memory limit: 3000 tokens to prevent context overflow

3. **Smart Engine** (`agent/engine.py`)
   - `SmartNeo4jEngine`: Custom query engine implementing:
     - **Dynamic RAG**: Retrieves similar Cypher examples from vector store (top-k=3)
     - **Adaptive Prompting**: Injects retrieved examples into the prompt when relevant
     - **Self-Healing**: Validates Cypher with `EXPLAIN` and auto-corrects syntax errors (max 3 retries)
   - Key methods:
     - `_validate_syntax()`: Uses Neo4j EXPLAIN to validate without execution
     - `_build_dynamic_prompt()`: Constructs prompt with optional examples section
     - `_generate_cypher_with_retry()`: Iterative generation with error feedback

4. **Data Seeding** (`seed.py`)
   - Loads Cypher query examples from `data/examples.json`
   - Creates embeddings using Google Generative AI
   - Stores in Neo4j vector store for semantic retrieval

### Graph Schema

The knowledge graph models user behavior tracking:
- **Nodes**: User, Visitor (anonymous), Session, Event, Conversion
- **Key Relationships**:
  - `(:Visitor)-[:IDENTIFIED_AS]->(:User)`: Links anonymous browsing to registered user
  - `(:Session)-[:CONTIENE]->(:Event)`: Granular session events
  - `(:Session)-[:CONVERSION_REALIZADA]->(:Conversion)`: Tracks sales

The schema enforces that conversions are pre-filtered events (success URL), and the `IDENTIFIED_AS` relationship enables journey analysis across anonymous and authenticated states.

### Agent Behavior

The agent follows specific rules defined in `agent_factory.py:44-58`:
1. Use `neo4j_data_tool` ONLY for quantitative queries requiring database access
2. Do NOT use the tool for greetings, explanations of previous responses, or general conversation
3. Leverage chat memory (`ChatMemoryBuffer`) to answer follow-up questions without re-querying

## Development Notes

### Adding New Cypher Examples

Edit `data/examples.json` with the format:
```json
{
  "question": "Business question in Spanish",
  "cypher": "MATCH ... RETURN ..."
}
```

Then re-run `python seed.py` to update the vector store.

### Modifying the Schema

1. Update `GRAPH_SCHEMA` in `config.py`
2. Ensure prompt templates reference the new schema elements
3. Update example queries in `data/examples.json` if needed

### Debugging Cypher Generation

The engine prints debug information:
- `[Self-Correction]`: Shows retry attempts and errors
- `[DEBUG] Executing Cypher`: Final validated query before execution

Set `verbose=True` in the agent (already enabled) to see ReAct reasoning steps.

### Known Issue in main.py

Line 75 has `asyncio.run(main())` which will cause the main function to run twice. This is likely a bug that should be removed.
