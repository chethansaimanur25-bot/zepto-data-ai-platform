# Support assistant (Module 3)

A small RAG service for Zepto policy questions: local embeddings, ChromaDB, a LangGraph router and a FastAPI endpoint. The graded baseline runs fully offline with `MOCK_LLM` unset.

## Setup and run (local)
From the repo root, with the virtual environment active and `pip install -r requirements.txt` done:

```
uvicorn main:app --app-dir support_assistant --port 8000
```

Then open http://127.0.0.1:8000/docs or POST to `/ask`. The first start downloads the `all-MiniLM-L6-v2` embedding model and builds the ChromaDB index. `python support_assistant/test_retrieval.py` runs 8 retrieval checks and `python support_assistant/test_graph.py` runs the routing checks.

## Architecture: ingestion -> embedding -> retrieval -> generation

```
docs/doc_01..08.txt -> ingest.py (chunk + embed) -> ChromaDB "zepto_policies"
user query -> classify_intent --policy_question--> retrieve_and_answer -> AskResponse
                             \--general_question--> direct_answer      -> AskResponse
```

1. **Ingestion and chunking** (`ingest.py`, `load_chunks`): each of the 8 corpus files is one chunk, because every document is about 330-560 characters, far below the embedding model's input limit. The chunk id is the file name (`doc_01` to `doc_08`).
2. **Embedding and storage** (`ingest.py`, `ingest`): `sentence-transformers` with `all-MiniLM-L6-v2` embeds each chunk locally (no API, no account) with normalised vectors, and the vectors are stored in a persistent ChromaDB collection named `zepto_policies` configured for cosine similarity. The index is created on first use and is not committed to Git.
3. **Retrieval** (`graph.py`, node `retrieve_and_answer`, which calls `ingest.py:retrieve`): embeds the query and returns the top-3 chunks by cosine similarity. This always runs for real in both modes.
4. **Generation** (`graph.py`): `retrieve_and_answer` builds the answer for policy questions and `direct_answer` for everything else. The structured prompt used by the real-LLM path is `prompts.py:PROMPT_TEMPLATE`. The final answer is validated by the Pydantic model `AskResponse` in `schemas.py` (`answer`, `sources`, `confidence`).

The LangGraph `StateGraph` (`graph.py`) has a `TypedDict` state and three nodes: `classify_intent`, `retrieve_and_answer`, `direct_answer`, joined by a conditional edge from `classify_intent`.

## The MOCK_LLM toggle
Only generation branches on `MOCK_LLM`; embedding, ChromaDB retrieval and the routing edge never do. With `MOCK_LLM` unset (or `1`) the service is deterministic and offline:
- `classify_intent`: keyword heuristic ("delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"), no LLM call.
- `retrieve_and_answer`: answer is `Based on the retrieved context: <first 200 characters of the top chunk>`, `sources` are the ids of the 3 retrieved chunks, `confidence` is 1.0.
- `direct_answer`: fixed string "I can only answer questions about Zepto policies right now.", empty `sources`, `confidence` 1.0.

With `MOCK_LLM=0` (optional extension, not part of the graded baseline) the same nodes call an LLM through Groq's free tier (needs a `GROQ_API_KEY` environment variable; never commit the key). Intent is classified by the LLM, answers are generated from the structured prompt grounded in the retrieved chunks, and the output is validated against `AskResponse`; invalid output is retried up to 2 more times with a corrective instruction before a clearly marked error response is returned (`generate_structured` in `graph.py`).

## Structured prompt
`prompts.py` contains the template with the five parts (role, context, task, format, length), a negative constraint ("Do not answer using information that is not present in the provided context") and a few-shot example.

## Example calls (MOCK_LLM at its default)
The raw JSON responses below were recorded by calling the running server (see `api_examples.txt`)
POST /ask
request : {"query": "What are your delivery charges?"}
status  : 200
response: {"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01","doc_05","doc_03"],"confidence":1.0}

POST /ask
request : {"query": "How do I get a refund?"}
status  : 200
response: {"answer":"Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop","sources":["doc_02","doc_06","doc_05"],"confidence":1.0}

POST /ask
request : {"query": "What is the capital of France?"}
status  : 200
response: {"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}


## Docker
```
docker build -t zepto-support support_assistant
docker run -p 7860:7860 zepto-support
```
Then POST to http://localhost:7860/ask. The image builds the ChromaDB index and caches the embedding model at build time and runs with `MOCK_LLM=1`.

Docker is not installed on the author's laptop. The image build and a container `/ask` call are verified by the GitHub Actions workflow `.github/workflows/docker-build.yml` (see the Actions tab: runs #4 and #5 are green).

## Known limitations
- The keyword router is specified by the brief and is deliberately simple. A policy question that contains none of the keywords (for example "Is there phone support?" or "What does Zepto Pass+ cost?") is routed to `direct_answer` in mock mode, even though retrieval would find the right document.
- Mock answers are a 200-character excerpt of the top chunk, so they can end mid-sentence.
- The Groq real-LLM path and a Hugging Face Spaces deployment were not attempted.