import os
from graph import graph, is_mock
from schemas import AskResponse

print("MOCK_LLM =", repr(os.getenv("MOCK_LLM")), "| mock mode:", is_mock())
print()

TESTS = [
    ("What are your delivery charges?", "policy_question", "doc_01"),
    ("Can I cancel my order after it is packed?", "policy_question", "doc_05"),
    ("How do I get a refund?", "policy_question", "doc_02"),
    ("What is the capital of France?", "general_question", None),
    ("Tell me a joke", "general_question", None),
]

ok_all = True
for query, want_intent, want_top in TESTS:
    out = graph.invoke({"query": query})
    resp = AskResponse(answer=out["answer"], sources=out.get("sources", []),
                       confidence=out["confidence"])
    ok = out["intent"] == want_intent and (
        want_top is None or (resp.sources and resp.sources[0] == want_top))
    ok_all &= bool(ok)
    print(("PASS" if ok else "FAIL"), "|", query)
    print("   intent:", out["intent"], "| sources:", resp.sources,
          "| confidence:", resp.confidence)
    print("   answer:", resp.answer[:110])

print()
print("ALL PASS" if ok_all else "SOME FAILED")