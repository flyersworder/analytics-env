# LLM-Driven Analytics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-provider LLM support and three example notebooks (code generation, RAG Q&A, automated pipeline) to the analytics-env template.

**Architecture:** Consolidate the `[llm]` dependency group around LangChain with 4 cloud providers + Ollama local. Three self-contained notebooks demonstrate distinct LLM-driven analytics patterns. No shared utility module — each notebook is standalone.

**Tech Stack:** LangChain, langchain-experimental, langchain-openai, langchain-anthropic, langchain-google-genai, langchain-ollama, ChromaDB, langchain-chroma, seaborn (sample data), Pydantic (structured output)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `pyproject.toml:31-39` | Update `[llm]` dependency group |
| Modify | `.env.example` | Add OPENAI_API_KEY, ANTHROPIC_API_KEY |
| Modify | `README.md` | Add LLM workflows, provider setup, local models sections |
| Create | `notebooks/llm_code_generation.ipynb` | Code generation with pandas agent |
| Create | `notebooks/llm_data_qa_rag.ipynb` | RAG Q&A with ChromaDB |
| Create | `notebooks/llm_automated_pipeline.ipynb` | LCEL pipeline with structured output |

---

### Task 1: Update Dependencies and Config

**Files:**
- Modify: `pyproject.toml:31-39`
- Modify: `.env.example`

- [ ] **Step 1: Update `[llm]` group in pyproject.toml**

Replace lines 31-39 in `pyproject.toml` (the `llm = [...]` block) with:

```toml
llm = [
    "chromadb>=1.0.0",
    "langchain>=0.3.13",
    "langchain-anthropic>=0.3.0",
    "langchain-chroma>=0.2.0",
    "langchain-experimental>=0.3.4",
    "langchain-google-genai>=2.0.7",
    "langchain-google-vertexai>=2.0.9",
    "langchain-ollama>=0.3.0",
    "langchain-openai>=0.3.0",
    "tavily-python>=0.5.0",
]
```

- [ ] **Step 2: Add new API key entries to `.env.example`**

Add these two lines to `.env.example` (before the existing entries):

```
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

The full file becomes:

```
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
GOOGLE_MAPS_API_KEY=your-google-maps-api-key
PATENT_SEARCH_API_KEY=your-patent-search-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

- [ ] **Step 3: Run `uv sync --extra llm --extra dev` to verify resolution**

```bash
uv sync --extra llm --extra dev
```

Expected: resolves and installs without errors. New packages installed include `langchain-anthropic`, `langchain-openai`, `langchain-ollama`, `langchain-chroma`, `chromadb`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock .env.example
git commit -m "Consolidate LLM deps around LangChain with multi-provider support

Replace crewai, llama-index, aisuite with langchain provider packages:
langchain-openai, langchain-anthropic, langchain-ollama, langchain-chroma.
Add OPENAI_API_KEY and ANTHROPIC_API_KEY to .env.example."
```

---

### Task 2: Code Generation Notebook

**Files:**
- Create: `notebooks/llm_code_generation.ipynb`

- [ ] **Step 1: Create `notebooks/llm_code_generation.ipynb`**

Create a Jupyter notebook with the following cells:

**Cell 1 (markdown):**
```markdown
# LLM-Assisted Code Generation for Analytics

This notebook demonstrates using an LLM to generate pandas code for data analysis.
The LLM receives your natural language questions and writes Python code to answer them.

## Provider Configuration

Uncomment your preferred provider below. API keys are loaded from `.env`.
```

**Cell 2 (code):**
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

**Cell 3 (markdown):**
```markdown
## Load Sample Data

We use the `tips` dataset from seaborn — restaurant tipping data with columns:
total_bill, tip, sex, smoker, day, time, size.
```

**Cell 4 (code):**
```python
import seaborn as sns

df = sns.load_dataset("tips")
df.head()
```

**Cell 5 (markdown):**
```markdown
## Create a Pandas DataFrame Agent

The agent can write and execute pandas code to answer questions about the data.
```

**Cell 6 (code):**
```python
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
)
```

**Cell 7 (markdown):**
```markdown
## Ask Questions in Natural Language

The agent generates and executes pandas code to answer each question.
```

**Cell 8 (code):**
```python
agent.invoke("What is the average tip percentage by day of the week? Show as a table.")
```

**Cell 9 (code):**
```python
agent.invoke(
    "Create a scatter plot of total_bill vs tip, colored by time (Lunch vs Dinner). "
    "Use matplotlib and add a title."
)
```

**Cell 10 (code):**
```python
agent.invoke(
    "Which combination of day and time has the highest average tip percentage? "
    "Show your reasoning step by step."
)
```

**Cell 11 (markdown):**
```markdown
## Notes

- The agent executes generated code in a sandboxed environment
- `allow_dangerous_code=True` is required for code execution — review generated code in production
- Works with any LangChain-compatible LLM provider (swap the provider cell above)
- For complex analyses, break your question into smaller steps for better results
```

- [ ] **Step 2: Verify the notebook parses correctly**

```bash
uv run python -c "import nbformat; nbformat.read('notebooks/llm_code_generation.ipynb', as_version=4); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run existing tests to ensure the new notebook is picked up**

```bash
uv run pytest tests/test_notebooks.py -v
```

Expected: 5 tests pass (4 existing + 1 new for `llm_code_generation.ipynb`).

- [ ] **Step 4: Commit**

```bash
git add notebooks/llm_code_generation.ipynb
git commit -m "Add LLM code generation example notebook

Demonstrates pandas DataFrame agent with multi-provider
config pattern using the tips dataset from seaborn."
```

---

### Task 3: Data Q&A with RAG Notebook

**Files:**
- Create: `notebooks/llm_data_qa_rag.ipynb`

- [ ] **Step 1: Create `notebooks/llm_data_qa_rag.ipynb`**

Create a Jupyter notebook with the following cells:

**Cell 1 (markdown):**
```markdown
# Data Q&A with Retrieval-Augmented Generation (RAG)

This notebook demonstrates how RAG improves LLM answers about your data.
We embed a data dictionary into ChromaDB, then the LLM retrieves relevant
context before answering — producing more accurate, grounded responses.

## Provider Configuration

Uncomment your preferred LLM and embedding providers below.
```

**Cell 2 (code):**
```python
from dotenv import load_dotenv

load_dotenv()

# LLM configuration — uncomment your preferred provider

from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o")

# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3.2")
```

**Cell 3 (code):**
```python
# Embedding configuration — uncomment your preferred provider

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# from langchain_ollama import OllamaEmbeddings
# embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

**Cell 4 (markdown):**
```markdown
## Load Sample Data

The Palmer Penguins dataset — measurements of penguin species from three Antarctic islands.
```

**Cell 5 (code):**
```python
import seaborn as sns

df = sns.load_dataset("penguins").dropna()
print(f"{len(df)} rows, {len(df.columns)} columns")
df.head()
```

**Cell 6 (markdown):**
```markdown
## Define the Data Dictionary

This is the domain knowledge we want the LLM to have access to.
In a real project, this could come from a wiki, a docs site, or a markdown file.
```

**Cell 7 (code):**
```python
data_dictionary = """
# Palmer Penguins Dataset — Data Dictionary

## Overview
This dataset contains measurements for three penguin species (Adelie, Chinstrap, Gentoo)
observed on three islands (Torgersen, Dream, Biscoe) in the Palmer Archipelago, Antarctica.
Data was collected by Dr. Kristen Gorman from 2007-2009.

## Columns

- **species**: Penguin species. One of: Adelie, Chinstrap, Gentoo.
  Adelie are the smallest and most widespread. Gentoo are the largest with
  distinctive orange bills. Chinstrap have a thin black line under the chin.
- **island**: Island name. Torgersen and Dream have Adelie penguins.
  Dream also has Chinstrap. Biscoe has Adelie and Gentoo.
- **bill_length_mm**: Length of the bill (culmen) in millimeters. Ranges ~32-60mm.
  Longer bills are typical of Gentoo and Chinstrap species.
- **bill_depth_mm**: Depth of the bill in millimeters. Ranges ~13-22mm.
  Adelie and Chinstrap have deeper bills relative to length than Gentoo.
- **flipper_length_mm**: Flipper length in millimeters. Ranges ~170-235mm.
  Gentoo have the longest flippers, correlating with their larger body size.
- **body_mass_g**: Body mass in grams. Ranges ~2700-6300g.
  Males are generally heavier than females across all species.
- **sex**: Penguin sex (Male or Female). Sexual dimorphism is most pronounced
  in body mass and flipper length.

## Common Analysis Patterns
- Species classification based on morphological measurements
- Sexual dimorphism analysis within and across species
- Island-species distribution patterns
- Bill shape (length vs depth ratio) as a species discriminator
- Body mass prediction from other measurements
"""
```

**Cell 8 (markdown):**
```markdown
## Build the RAG Vector Store

Split the data dictionary into chunks, embed them, and store in ChromaDB.
```

**Cell 9 (code):**
```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.create_documents([data_dictionary])

vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(f"Indexed {len(chunks)} chunks into ChromaDB")
```

**Cell 10 (markdown):**
```markdown
## Compare: Without RAG vs With RAG

First, let's ask the LLM directly (no context). Then ask with RAG retrieval.
```

**Cell 11 (code):**
```python
question = "Which penguin species has the deepest bill relative to its length, and why might that matter?"

# Without RAG — LLM uses only its training data
response_no_rag = llm.invoke(question)
print("WITHOUT RAG:")
print(response_no_rag.content)
```

**Cell 12 (code):**
```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are a data analyst assistant. Use the following context from the data "
    "dictionary to answer questions about the penguins dataset. If the context "
    "doesn't contain the answer, say so.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response_rag = rag_chain.invoke({"input": question})
print("WITH RAG:")
print(response_rag["answer"])
```

**Cell 13 (markdown):**
```markdown
## Ask More Questions

The RAG chain grounds answers in your data dictionary, producing more specific
and accurate responses than the LLM alone.
```

**Cell 14 (code):**
```python
for q in [
    "What islands can I find Chinstrap penguins on?",
    "If I want to predict body mass, which features should I use and why?",
    "How can I tell Adelie and Chinstrap apart using bill measurements?",
]:
    result = rag_chain.invoke({"input": q})
    print(f"Q: {q}")
    print(f"A: {result['answer']}\n")
```

**Cell 15 (markdown):**
```markdown
## Notes

- ChromaDB stores vectors in-memory by default — for persistence, pass `persist_directory`
- Embedding cost is minimal: the data dictionary is small (a few hundred tokens)
- For larger data dictionaries, tune `chunk_size` and `k` for better retrieval
- Ollama embeddings (`nomic-embed-text`) work fully offline — no API key needed
```

- [ ] **Step 2: Verify the notebook parses correctly**

```bash
uv run python -c "import nbformat; nbformat.read('notebooks/llm_data_qa_rag.ipynb', as_version=4); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_notebooks.py -v
```

Expected: 6 tests pass (5 previous + 1 new for `llm_data_qa_rag.ipynb`).

- [ ] **Step 4: Commit**

```bash
git add notebooks/llm_data_qa_rag.ipynb
git commit -m "Add RAG data Q&A example notebook

Demonstrates ChromaDB vector store with data dictionary retrieval,
comparing LLM answers with and without RAG context."
```

---

### Task 4: Automated Pipeline Notebook

**Files:**
- Create: `notebooks/llm_automated_pipeline.ipynb`

- [ ] **Step 1: Create `notebooks/llm_automated_pipeline.ipynb`**

Create a Jupyter notebook with the following cells:

**Cell 1 (markdown):**
```markdown
# LLM-Powered Data Pipeline

This notebook demonstrates chaining LLM steps to process text data at scale:
classify sentiment, extract entities, and summarize — outputting structured results
as a pandas DataFrame.

## Provider Configuration
```

**Cell 2 (code):**
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

**Cell 3 (markdown):**
```markdown
## Sample Data: Product Reviews

Synthetic reviews for demonstrating the pipeline. In production, this would
come from a database, CSV, or API.
```

**Cell 4 (code):**
```python
reviews = [
    {"id": 1, "text": "Absolutely love this coffee maker! Brews fast and the temperature is perfect every time. Best purchase this year.", "rating": 5},
    {"id": 2, "text": "The blender stopped working after two weeks. Motor burned out and customer service was unhelpful. Total waste of money.", "rating": 1},
    {"id": 3, "text": "Decent toaster for the price. Toast comes out evenly but the timer is a bit unreliable. Gets the job done.", "rating": 3},
    {"id": 4, "text": "This air fryer changed how I cook. Everything comes out crispy with barely any oil. Easy to clean too.", "rating": 5},
    {"id": 5, "text": "Microwave works fine but the door handle feels cheap and the beeping is extremely loud. Annoying but functional.", "rating": 3},
    {"id": 6, "text": "Returned immediately. The electric kettle leaked from day one. Dangerous product that should be recalled.", "rating": 1},
    {"id": 7, "text": "Solid stand mixer. Heavy and sturdy, handles thick dough without struggling. A bit noisy on high speed though.", "rating": 4},
    {"id": 8, "text": "The rice cooker makes perfect rice every time. Set it and forget it. Wish it had a larger capacity though.", "rating": 4},
    {"id": 9, "text": "Dishwasher leaves spots on glasses and doesn't dry properly. For this price I expected much better performance.", "rating": 2},
    {"id": 10, "text": "Incredible espresso machine. Pulls shots like a cafe. The learning curve is steep but worth every penny.", "rating": 5},
]
```

**Cell 5 (markdown):**
```markdown
## Define Structured Output Schema

Using Pydantic models to ensure the LLM returns structured, typed data.
```

**Cell 6 (code):**
```python
from pydantic import BaseModel, Field


class ReviewAnalysis(BaseModel):
    """Structured analysis of a single product review."""

    sentiment: str = Field(description="One of: positive, negative, neutral")
    product_features: list[str] = Field(
        description="Key product features or aspects mentioned in the review"
    )
    summary: str = Field(description="One-sentence summary of the review")
```

**Cell 7 (markdown):**
```markdown
## Build the LCEL Pipeline

LangChain Expression Language (LCEL) lets us compose prompt → LLM → parser
into a single chain using the `|` (pipe) operator.
```

**Cell 8 (code):**
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a product review analyst. Analyze the given review and extract "
            "structured information. Be concise and precise.",
        ),
        (
            "human",
            "Analyze this product review:\n\n{review_text}",
        ),
    ]
)

chain = prompt | llm.with_structured_output(ReviewAnalysis)
```

**Cell 9 (markdown):**
```markdown
## Process a Single Review

Test the chain on one review before running the full batch.
```

**Cell 10 (code):**
```python
result = chain.invoke({"review_text": reviews[0]["text"]})
print(f"Sentiment: {result.sentiment}")
print(f"Features: {result.product_features}")
print(f"Summary: {result.summary}")
```

**Cell 11 (markdown):**
```markdown
## Batch Process All Reviews

Use `chain.batch()` to process all reviews efficiently.
```

**Cell 12 (code):**
```python
inputs = [{"review_text": r["text"]} for r in reviews]
results = chain.batch(inputs)

print(f"Processed {len(results)} reviews")
```

**Cell 13 (markdown):**
```markdown
## Results as a DataFrame

Combine the structured LLM output with the original review data.
```

**Cell 14 (code):**
```python
import pandas as pd

analysis_df = pd.DataFrame(
    {
        "id": [r["id"] for r in reviews],
        "rating": [r["rating"] for r in reviews],
        "sentiment": [r.sentiment for r in results],
        "product_features": [", ".join(r.product_features) for r in results],
        "summary": [r.summary for r in results],
    }
)

analysis_df
```

**Cell 15 (code):**
```python
print("Sentiment distribution:")
print(analysis_df["sentiment"].value_counts().to_string())
print(f"\nAverage rating by sentiment:")
print(analysis_df.groupby("sentiment")["rating"].mean().to_string())
```

**Cell 16 (markdown):**
```markdown
## Notes

- `with_structured_output()` uses the LLM's native tool/function calling — no regex parsing
- `chain.batch()` processes inputs concurrently for better throughput
- Pydantic validation ensures type safety — malformed LLM output raises clear errors
- This pattern scales to thousands of records with `chain.batch(inputs, config={"max_concurrency": 5})`
- Works with any provider that supports structured output (OpenAI, Anthropic, Google, newer Ollama models)
```

- [ ] **Step 2: Verify the notebook parses correctly**

```bash
uv run python -c "import nbformat; nbformat.read('notebooks/llm_automated_pipeline.ipynb', as_version=4); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_notebooks.py -v
```

Expected: 7 tests pass (6 previous + 1 new for `llm_automated_pipeline.ipynb`).

- [ ] **Step 4: Commit**

```bash
git add notebooks/llm_automated_pipeline.ipynb
git commit -m "Add LLM automated pipeline example notebook

Demonstrates LCEL chain with Pydantic structured output for
batch processing product reviews: classify, extract, summarize."
```

---

### Task 5: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the LLM row in the Dependency Groups table**

In the existing Dependency Groups table, replace the LLM row:

Old:
```
| **LLM** | `uv sync --extra llm` | LangChain, CrewAI, LlamaIndex, aisuite, Tavily |
```

New:
```
| **LLM** | `uv sync --extra llm` | LangChain, langchain-openai, langchain-anthropic, langchain-ollama, langchain-chroma, ChromaDB |
```

- [ ] **Step 2: Add LLM Workflows section after Dependency Groups**

Insert after the Dependency Groups table and before "## Development Workflow":

```markdown
## LLM Workflows

The template includes example notebooks for three LLM-driven analytics patterns:

| Notebook | Pattern | What it demonstrates |
|----------|---------|---------------------|
| [llm_code_generation.ipynb](notebooks/llm_code_generation.ipynb) | Code generation | Pandas DataFrame agent — ask questions in English, get code + results |
| [llm_data_qa_rag.ipynb](notebooks/llm_data_qa_rag.ipynb) | Data Q&A with RAG | ChromaDB vector store + data dictionary retrieval for grounded answers |
| [llm_automated_pipeline.ipynb](notebooks/llm_automated_pipeline.ipynb) | Automated pipeline | LCEL chain with Pydantic structured output for batch text processing |

### Provider Setup

Each notebook supports multiple LLM providers. Set the relevant API key in `.env`:

| Provider | Env variable | Notes |
|----------|-------------|-------|
| OpenAI | `OPENAI_API_KEY` | GPT-4o, text-embedding-3-small |
| Anthropic | `ANTHROPIC_API_KEY` | Claude Sonnet |
| Google | `GOOGLE_API_KEY` | Gemini 2.5 Flash |
| Ollama | (none — local) | [Install Ollama](https://ollama.com), then `ollama pull llama3.2` |

### Local Models with Ollama

For fully offline operation, use [Ollama](https://ollama.com) as both LLM and embedding provider:

- **Embeddings** (`nomic-embed-text`): ~2GB RAM, runs on CPU
- **Generation** (`llama3.2`, `mistral`): 8GB+ RAM recommended, GPU significantly improves speed
- Cloud providers are recommended for production workloads
```

- [ ] **Step 3: Run linting to verify README formatting**

```bash
prek run --all-files
```

Expected: all hooks pass.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Add LLM workflows, provider setup, and local models to README

Documents three example notebooks, multi-provider config,
and Ollama hardware requirements."
```

---

## Post-Implementation

After all 5 tasks are committed, run final validation:

```bash
prek run --all-files
uv run pytest -v
```

Both should pass. The test suite should show 7 passing tests (4 original notebooks + 3 new LLM notebooks).
