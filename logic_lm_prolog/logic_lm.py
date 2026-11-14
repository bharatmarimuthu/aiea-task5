"""Logic-LM (Pan et al., 2023) reimplemented on the Task 4 gym-exercise Prolog KB.

Pipeline, following the paper's three stages and its self-refiner:
  1. Problem Formulator  LLM translates the English question into a Prolog goal
                         (paper: models/logic_program.py)
  2. Symbolic Reasoner   SWI-Prolog runs the goal against kb.pl through Janus
                         (paper: models/logic_inference.py + symbolic_solvers/)
  3. Result Interpreter  maps the solver output to an answer: true/false or a list
                         (paper: answer_mapping)
  +  Self-Refiner        if Prolog raises an error, the error message goes back to the LLM,
                         which rewrites the goal; up to --max_rounds times
                         (paper: models/self_refinement.py)
Baselines: --mode direct (standard prompting) and --mode cot (chain-of-thought), where the LLM
reads kb.pl and answers on its own. --mode oracle uses the gold queries instead of an LLM, to
test the pipeline without any model.

Metrics match the paper's models/evaluation.py: overall accuracy, Exe_Rate (share of programs
that ran) and Exe_Acc (accuracy on the ones that ran).

Examples:
  python logic_lm.py --mode oracle
  python logic_lm.py --mode logic-lm --max_rounds 3
  python logic_lm.py --mode direct
"""
import argparse
import json
import os
import re
import time

from llm import few_shot_messages
from prolog_solver import Prolog_Program

HERE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(HERE, "kb.pl")


def read(path):
    with open(os.path.join(HERE, path)) as f:
        return f.read()


# ---------------- stage 1: Problem Formulator ----------------
def formulate(llm, question):
    prompt = read("prompts/gym.txt").replace("[[QUESTION]]", question)
    return llm.generate(few_shot_messages(prompt))


def self_correct(llm, question, program, error_message):
    goal = program.split("Query:", 1)[-1].strip()
    prompt = (read("prompts/self-correct-gym.txt")
              .replace("[[QUESTION]]", question)
              .replace("[[PROGRAM]]", goal)
              .replace("[[ERROR MESSAGE]]", error_message))
    return llm.generate(few_shot_messages(prompt))


# ---------------- stages 2 + 3: Symbolic Reasoner + Result Interpreter ----------------
def safe_execute_program(logic_program):
    program = Prolog_Program(logic_program, KB_PATH)
    if not program.flag:
        return None, "parsing error", "Parsing Error", program.goal
    answer, error_message = program.execute_program()
    if answer is None:
        return None, "execution error", error_message, program.goal
    return program.answer_mapping(answer), "success", "", program.goal


# ---------------- baselines (paper: baselines/gpt3_baseline.py) ----------------
def parse_answer(text):
    lines = [l for l in text.strip().splitlines() if "answer:" in l.lower()]
    line = lines[-1] if lines else text.strip().splitlines()[-1] if text.strip() else ""
    after = line.lower().split("answer:", 1)[-1]
    if "[" in after:
        return sorted(set(re.findall(r"[a-z][a-z0-9_]*", after.split("[", 1)[1])))
    if "true" in after or "yes" in after:
        return "true"
    if "false" in after or "no" in after:
        return "false"
    return None


def run_baseline(llm, question, mode):
    prompt = read(f"prompts/{mode}.txt").replace("[[KB]]", read("kb.pl")).replace("[[QUESTION]]", question)
    raw = llm.generate(prompt, stop=False)
    return parse_answer(raw), raw


# ---------------- main loop ----------------
def run(args):
    with open(os.path.join(HERE, "data/gym_dev.json")) as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[: args.limit]

    llm = None
    if args.mode != "oracle":
        from llm import LLM
        llm = LLM(model=args.model)
        print(f"Model: {llm.model}  (endpoint: {llm.client.base_url})")
    model_name = "oracle" if llm is None else llm.model.replace("/", "_").replace(":", "_")

    outputs = []
    for ex in dataset:
        q, gold = ex["question"], ex["answer"]
        record = {"id": ex["id"], "hops": ex["hops"], "question": q, "answer": gold}
        t0 = time.time()

        if args.mode in ("direct", "cot"):
            pred, raw = run_baseline(llm, q, args.mode)
            record.update(predicted_answer=pred, raw_output=raw, flag="success")
        else:
            program = ("Query:\n" + ex["gold_query"]) if args.mode == "oracle" else formulate(llm, q)
            history = []
            pred, flag, err, goal = safe_execute_program(program)
            history.append({"round": 0, "goal": goal, "flag": flag, "error": err})
            rounds = 0
            while flag != "success" and llm is not None and rounds < args.max_rounds:
                rounds += 1
                program = self_correct(llm, q, program, err)
                pred, flag, err, goal = safe_execute_program(program)
                history.append({"round": rounds, "goal": goal, "flag": flag, "error": err})
            if flag != "success" and args.backup == "cot":
                pred, _ = run_baseline(llm, q, "cot")
            record.update(predicted_answer=pred, flag=flag, refine_rounds=rounds, history=history)

        record["correct"] = record["predicted_answer"] == gold
        outputs.append(record)

        mark = "✓" if record["correct"] else "✗"
        print(f"{mark} {ex['id']:7} {q[:58]:58} [{time.time() - t0:4.1f}s]")
        if args.verbose or not record["correct"]:
            for h in record.get("history", []):
                print(f"      round {h['round']}: {h['goal']!r} -> {h['flag']} {h['error']}")
            print(f"      predicted: {record['predicted_answer']}")
            print(f"      gold:      {gold}")

    # ---- evaluation (paper: models/evaluation.py full_evaluation) ----
    n = len(outputs)
    executable = [o for o in outputs if o["flag"] == "success"]
    acc = sum(o["correct"] for o in outputs) / n
    print(f"\nMode: {args.mode}   questions: {n}")
    print(f"Overall accuracy:               {acc:.3f}")
    if args.mode not in ("direct", "cot"):
        first_try = sum(o["history"][0]["flag"] == "success" for o in outputs) / n
        exe_acc = sum(o["correct"] for o in executable) / len(executable) if executable else 0.0
        print(f"Exe_Rate before self-refine:    {first_try:.3f}")
        print(f"Executable rate (Exe_Rate):     {len(executable) / n:.3f}")
        print(f"Executable accuracy (Exe_Acc):  {exe_acc:.3f}")

    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    out_path = os.path.join(HERE, "outputs", f"{args.mode}_{model_name}.json")
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Logic-LM on the gym-exercise Prolog KB")
    p.add_argument("--mode", default="logic-lm", choices=["logic-lm", "direct", "cot", "oracle"])
    p.add_argument("--model", default=None, help="overrides LLM_MODEL")
    p.add_argument("--max_rounds", type=int, default=3, help="self-refinement rounds (paper tests 0-3)")
    p.add_argument("--backup", default="none", choices=["none", "cot"],
                   help="if a program never runs: count it wrong, or fall back to CoT (paper's 'LLM' backup)")
    p.add_argument("--limit", type=int, default=0, help="only run the first N questions")
    p.add_argument("--verbose", action="store_true", help="print every goal and answer")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
