"""Compute gold answers by running each question's reference query in Prolog.

Reads data/gym_questions.json and writes data/gym_dev.json, the same shape as the paper's
data files (id, question, answer) plus the gold query.
"""
import json
from prolog_solver import Prolog_Program

with open("data/gym_questions.json") as f:
    questions = json.load(f)

out = []
for q in questions:
    prog = Prolog_Program("Query:\n" + q["gold_query"])
    ans, err = prog.execute_program()
    assert ans is not None, f"{q['id']}: gold query failed: {err}"
    out.append({**q, "answer": prog.answer_mapping(ans)})
    print(f"{q['id']:7} {q['question'][:60]:60} -> {out[-1]['answer']}")

with open("data/gym_dev.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {len(out)} questions to data/gym_dev.json")
