# LLM-Driven Analytics Template — Design Spec

**Date:** 2026-03-26
**Status:** Approved

## Goal

Enhance analytics-env with LLM-driven analytics capabilities: multi-provider support, example notebooks for three workflow patterns (code generation, data Q&A with RAG, automated pipelines), all built on LangChain.

## 1. Dependency Changes

### Remove from `[llm]`
- `crewai` — overlaps with LangChain agents
- `llama-index` — overlaps with LangChain RAG
- `aisuite[all]` — overlaps with LangChain provider abstraction

### Add to `[llm]`
- `langchain-openai` — OpenAI provider (GPT-4o, embeddings)
- `langchain-anthropic` — Anthropic provider (Claude)
- `langchain-ollama` — local models via Ollama
- `chromadb` — file-based vector store for RAG
- `langchain-chroma` — LangChain ChromaDB integration

### Keep in `[llm]`
- `langchain`
- `langchain-experimental` (pandas agent)
- `langchain-google-genai`
- `langchain-google-vertexai`
- `tavily-python` (web search tool for agents)

### Final `[llm]` group
```
chromadb
langchain
langchain-anthropic
langchain-chroma
langchain-experimental
langchain-google-genai
langchain-google-vertexai
langchain-ollama
langchain-openai
tavily-python
```

## 2. Example Notebooks

### `notebooks/llm_code_generation.ipynb` — Code generation for analytics

**Purpose:** Demonstrate LLM-assisted data analysis via code generation.

**Structure:**
1. Provider configuration cell (swap pattern — see Section 3)
2. Load sample dataset: `seaborn.load_dataset("tips")`
3. Create a LangChain pandas DataFrame agent using `langchain-experimental`
4. Demo prompts:
   - "What's the average tip by day of the week?"
   - "Create a scatter plot of total bill vs tip, colored by time"
   - "Which server had the highest average tip percentage?"
5. Show the generated code and output for each

### `notebooks/llm_data_qa_rag.ipynb` — Data Q&A with RAG

**Purpose:** Demonstrate retrieval-augmented Q&A over data with context from a data dictionary.

**Structure:**
1. Provider configuration cell (LLM + embeddings swap pattern)
2. Load sample dataset: `seaborn.load_dataset("penguins")`
3. Define a data dictionary as a markdown string (column descriptions, common analysis patterns, domain context about penguin species)
4. Split data dictionary into chunks, embed into ChromaDB
5. Build a retrieval chain: query → retrieve relevant context → augment prompt → LLM answer
6. Demo without RAG vs with RAG (show the quality difference)
7. Conversational follow-up questions demonstrating context retention

### `notebooks/llm_automated_pipeline.ipynb` — LLM-powered data pipeline

**Purpose:** Demonstrate chaining LLM steps for structured data processing.

**Structure:**
1. Provider configuration cell
2. Define synthetic data: 10-15 hardcoded product reviews (text, rating)
3. Build LCEL pipeline:
   - Step 1: Classify sentiment (positive/negative/neutral)
   - Step 2: Extract key entities (product features mentioned)
   - Step 3: Summarize each review in one sentence
   - Step 4: Output structured JSON
4. Parse results into a pandas DataFrame
5. Show batch processing across all reviews
6. Demonstrate structured output parsing with Pydantic models

## 3. Provider Configuration Pattern

Each notebook starts with this cell:

```python
from dotenv import load_dotenv
load_dotenv()

# Provider configuration — uncomment your preferred provider
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o")

# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3.2")
```

Embeddings (in RAG notebook):

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# from langchain_ollama import OllamaEmbeddings
# embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

API keys loaded from `.env` via `python-dotenv`. No utility module — config is self-contained in each notebook.

## 4. Sample Data

- **Code generation notebook:** `seaborn.load_dataset("tips")` — built-in, no files needed
- **Data Q&A notebook:** `seaborn.load_dataset("penguins")` + inline markdown data dictionary string
- **Pipeline notebook:** 10-15 hardcoded product review strings with ratings in a list of dicts

All self-contained in notebooks. No files in `data/`, no downloads.

## 5. .env.example Updates

Add:
```
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

Existing entries remain unchanged.

## 6. README Updates

Add after "Dependency Groups" section:

### LLM Workflows section
- Brief description of three patterns with links to notebooks
- Table: notebook name | workflow pattern | what it demonstrates

### Provider Setup section
- Table: provider | env var | notes
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`
- Ollama: no key needed, local

### Local Models section
- Ollama install link
- Embeddings: ~2GB RAM (nomic-embed-text)
- Generation: 8GB+ RAM, GPU recommended (llama3.2, mistral, etc.)
- Note: cloud providers recommended for production workloads

## Out of Scope

- Custom utility module / `src/` package
- Fine-tuning or model training
- Deployment / serving infrastructure
- Multi-agent orchestration (removed CrewAI)
- Evaluation / benchmarking framework
