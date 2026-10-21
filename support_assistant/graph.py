import os
import re
from typing import List, TypedDict

import requests
from langgraph.graph import StateGraph, END

from ingest import retrieve
from prompts import build_prompt, CLASSIFY_PROMPT
from schemas import AskResponse

KEYWORDS = ["delivery", "return", "refund", "membership",
            "tracking", "cancel", "gift card", "support hours"]
GENERAL_MOCK_ANSWER = "I can only answer questions about Zepto policies right now."
MAX_RETRIES = 2


def is_mock() -> bool:
    """MOCK_LLM unset or 1 -> deterministic offline mock (graded baseline)."""
    return os.getenv("MOCK_LLM", "1") != "0"


class State(TypedDict, total=False):
    query: str
    intent: str
    chunks: List[dict]
    answer: str
    sources: List[str]
    confidence: float


def keyword_intent(query: str) -> str:
    q = query.lower()
    return "policy_question" if any(k in q for k in KEYWORDS) else "general_question"


# ---- optional real-LLM helpers (only used when MOCK_LLM=0) ----
def call_llm(prompt: str) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_response(raw: str) -> AskResponse:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return AskResponse.model_validate_json(raw)


def generate_structured(prompt: str) -> AskResponse:
    """Validate against the schema; retry up to 2 more times with a corrective instruction."""
    current, last_err = prompt, None
    for _ in range(1 + MAX_RETRIES):
        try:
            return parse_response(call_llm(current))
        except Exception as e:
            last_err = e
            current = (prompt + "\n\nYour previous output was invalid ("
                       + str(e)[:200].replace("\n", " ")
                       + "). Return ONLY valid JSON with fields answer, sources, confidence.")
    return AskResponse(
        answer=f"ERROR: could not produce a valid response ({last_err})",
        sources=[], confidence=0.0)


# ---- graph nodes ----
def classify_intent(state: State) -> dict:
    q = state["query"]
    if is_mock():
        return {"intent": keyword_intent(q)}          # no LLM call
    try:
        label = call_llm(CLASSIFY_PROMPT.format(question=q)).strip().lower()
        return {"intent": "policy_question" if "policy" in label else "general_question"}
    except Exception:
        return {"intent": keyword_intent(q)}


def retrieve_and_answer(state: State) -> dict:
    q = state["query"]
    chunks = retrieve(q, k=3)                         # real retrieval in BOTH modes
    ids = [c["id"] for c in chunks]
    if is_mock():
        snippet = chunks[0]["text"][:200]
        return {"chunks": chunks,
                "answer": f"Based on the retrieved context: {snippet}",
                "sources": ids, "confidence": 1.0}
    resp = generate_structured(build_prompt(q, chunks))
    return {"chunks": chunks, "answer": resp.answer,
            "sources": resp.sources, "confidence": resp.confidence}


def direct_answer(state: State) -> dict:
    if is_mock():
        return {"answer": GENERAL_MOCK_ANSWER, "sources": [], "confidence": 1.0}
    try:
        text = call_llm("Answer briefly. Do not claim to know Zepto's policies.\n\n"
                        "Question: " + state["query"])
        return {"answer": text.strip(), "sources": [], "confidence": 0.5}
    except Exception as e:
        return {"answer": f"ERROR: {e}", "sources": [], "confidence": 0.0}


def route(state: State) -> str:
    return ("retrieve_and_answer" if state["intent"] == "policy_question"
            else "direct_answer")


builder = StateGraph(State)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)
builder.set_entry_point("classify_intent")
builder.add_conditional_edges("classify_intent", route, {
    "retrieve_and_answer": "retrieve_and_answer",
    "direct_answer": "direct_answer",
})
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)
graph = builder.compile()


def run(query: str) -> AskResponse:
    out = graph.invoke({"query": query})
    return AskResponse(answer=out["answer"], sources=out.get("sources", []),
                       confidence=out["confidence"])