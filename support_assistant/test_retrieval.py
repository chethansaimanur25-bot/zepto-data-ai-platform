from ingest import retrieve

TESTS = [
    ("How much is the delivery fee for a small order?", "doc_01"),
    ("How long does a refund take to reach my bank account?", "doc_02"),
    ("What does Zepto Pass+ cost per month?", "doc_03"),
    ("My order has not moved for 25 minutes, what should I do?", "doc_04"),
    ("Can I cancel my order after it is packed?", "doc_05"),
    ("An item was damaged and my order was INR 1500, what do I need?", "doc_06"),
    ("Can I use two gift cards in one purchase?", "doc_07"),
    ("Is there phone support?", "doc_08"),
]

passed = 0
for query, expected in TESTS:
    hits = retrieve(query, k=3)
    top = hits[0]["id"]
    ok = top == expected
    passed += ok
    print(("PASS" if ok else "FAIL"), "|", query)
    print("   expected", expected, "| top-3:",
          [(h["id"], h["similarity"]) for h in hits])

print()
print(f"{passed}/{len(TESTS)} queries retrieved the expected document first")