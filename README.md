# tfm-nlp-business

Repository for the Master's Thesis consisting of a conversational assistant for analysis and monitoring of a company's data.

## Overview

This project implements a conversational assistant powered by RAG (Retrieval-Augmented Generation) and Neo4j graph database for analyzing business data, user behavior, conversions, and metrics.

### Key Features

- **Smart Neo4j Query Engine** with self-healing capabilities
- **Hybrid RAG Architecture**: Vector RAG + Graph RAG for contextual query generation
- **Conversational Agent** with memory and tool usage
- **Comprehensive Evaluation System** for measuring performance

## Quick Start

### Prerequisites

- Python 3.8+
- Neo4j database (local or cloud)
- Google Generative AI API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd tfm-nlp-business

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your credentials
```

### Configuration

Create a `.env` file with the following variables:

```bash
LLM_API_KEY=your_google_api_key
LLM_MODEL=gemini-1.5-pro
EMBEDDING_MODEL=models/embedding-001
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### Initialize Database

```bash
# Seed the vector store with example Cypher queries
python seed.py
```

### Run the Application

```bash
# Start the conversational assistant
python main.py
```

## Testing and Evaluation

This project includes a comprehensive test suite for evaluating the query engine performance.

### Running Tests

```bash
# Full evaluation (all test cases)
python tests/test_engine_evaluation.py

# Comparative evaluation (with RAG vs without RAG)
python tests/test_engine_evaluation.py --mode comparative

# Save results to custom file
python tests/test_engine_evaluation.py --output my_results.json
```

### Visualizing Results

Generate charts and tables from test results:

```bash
# Requires: pip install matplotlib pandas
python tests/visualize_results.py evaluation_results.json

# Custom output directory
python tests/visualize_results.py evaluation_results.json --output graficas/
```

This generates:
- PNG charts (category analysis, comparative analysis, success distribution)
- CSV files for Excel analysis
- LaTeX tables for thesis document

### Available Tests

1. **test_engine_evaluation.py**: Quantitative evaluation with 13 test cases
2. **test_graph_rag.py**: Validation of hybrid RAG implementation
3. **test_memory.py**: Conversational memory verification

See [tests/README.md](tests/README.md) for detailed documentation.

## Project Structure

```
tfm-nlp-business/
├── agent/                      # Agent implementation
│   ├── agent_factory.py       # Agent creation and configuration
│   ├── engine.py              # Smart Neo4j query engine
│   └── graph_context_provider.py  # Graph RAG provider
├── data/                      # Data files
│   └── examples.json          # Cypher query examples for RAG
├── tests/                     # Test suite
│   ├── README.md              # Test documentation
│   ├── test_engine_evaluation.py  # Main evaluation test
│   ├── test_graph_rag.py      # RAG validation
│   ├── test_memory.py         # Memory validation
│   └── visualize_results.py   # Result visualization
├── config.py                  # Configuration and prompts
├── seed.py                    # Database seeding script
├── main.py                    # Main application entry point
└── CLAUDE.md                  # Development guidelines
```

## Architecture

### Components

1. **SmartNeo4jEngine** (`agent/engine.py`)
   - Dynamic RAG: Retrieves similar Cypher examples
   - Adaptive Prompting: Context-aware prompt generation
   - Self-Healing: Validates and auto-corrects Cypher syntax
   - Graph RAG: Injects business context from graph statistics

2. **Agent Factory** (`agent/agent_factory.py`)
   - Creates ReAct-pattern agent with chat memory
   - Configures tools and decision-making logic

3. **Graph Context Provider** (`agent/graph_context_provider.py`)
   - Retrieves statistical context from the graph
   - Provides business metrics for informed query generation

### Graph Schema

The Neo4j knowledge graph models:

- **Nodes**: User, Visitor, Session, Event, Conversion
- **Relationships**:
  - `(:Visitor)-[:IDENTIFIED_AS]->(:User)`
  - `(:Session)-[:CONTIENE]->(:Event)`
  - `(:Session)-[:CONVERSION_REALIZADA]->(:Conversion)`

## Development

### Adding New Test Cases

Edit `tests/test_engine_evaluation.py` and add to `_define_test_cases()`:

```python
QueryTestCase(
    id="custom_01",
    category="custom",
    question="Your question here?",
    expected_type="count"  # or "list", "aggregation"
)
```

### Adding Cypher Examples

Edit `data/examples.json`:

```json
{
  "question": "Business question in Spanish",
  "cypher": "MATCH ... RETURN ..."
}
```

Then re-run: `python seed.py`

### Rate Limiting

The test suite includes automatic handling of API rate limits (429 errors):
- Automatic retry with 40-second wait
- Preventive pauses between tests
- Suitable for limited billing plans

## Documentation

- [CLAUDE.md](CLAUDE.md) - Development guidelines and architecture details
- [tests/README.md](tests/README.md) - Complete test documentation

## License

This project is part of a Master's Thesis.

## Contact

For questions or issues, please open an issue in this repository.
