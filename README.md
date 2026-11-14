# AIEA Task 5: Logic-LM on a Prolog knowledge base

This repo reimplements **Logic-LM** (Pan et al., *Findings of EMNLP 2023*, [arXiv 2305.12295](https://arxiv.org/abs/2305.12295)) and runs it on my Task 4 gym-exercise knowledge base.

- **Part 1** runs the authors' own code ([teacherpeterpan/Logic-LLM](https://github.com/teacherpeterpan/Logic-LLM)) and reproduces the paper's Table 3.
- **Part 2** (`logic_lm_prolog/`) is my reimplementation. It follows the paper's three stages and its self-refiner, but the solver is SWI-Prolog through Janus instead of Pyke.

| Paper stage | Paper file | My version |
|---|---|---|
| Problem Formulator (LLM → logic) | `models/logic_program.py`, `prompts/ProntoQA.txt` | `logic_lm.py: formulate()`, `prompts/gym.txt`, `llm.py: few_shot_messages()` |
| Symbolic Reasoner | `symbolic_solvers/pyke_solver/pyke_solver.py` | `prolog_solver.py: Prolog_Program` |
| Result Interpreter | `answer_mapping()` | `Prolog_Program.answer_mapping()` |
| Self-Refiner | `models/self_refinement.py`, `prompts/self-correct-*.txt` | `logic_lm.py: self_correct()`, `prompts/self-correct-gym.txt` |
| Baselines | `baselines/gpt3_baseline.py` | `--mode direct`, `--mode cot` |
| Evaluation (Accuracy, Exe_Rate, Exe_Acc) | `models/evaluation.py` | end of `logic_lm.py: run()` |

---

## Walkthrough

Run every command from the repo root (`~/Projects/aiea-onboarding/aiea-task5`) unless a step says otherwise.

### Step 0: Python environment (one time)

The authors pinned 2023 packages (torch, CUDA) that won't install on a Mac with Python 3.14. Their Pyke solver also needs the `imp` module, which Python removed in 3.12. So use a Python 3.10 conda env with only the packages the solvers need:

```bash
conda create -y -n logiclm python=3.10
conda activate logiclm
pip install -r requirements.txt
```

You also need SWI-Prolog (already installed: `swipl --version` shows 10.0.2).

### Step 1: Run the paper's code (no LLM or API key needed)

The authors published the logic programs GPT-4 generated, so you can run the **solver + evaluation** half of their pipeline for free:

```bash
git clone https://github.com/teacherpeterpan/Logic-LLM.git   # skip if the folder exists
cd Logic-LLM
python models/logic_inference.py --model_name gpt-4 --dataset_name ProntoQA --split dev --backup_strategy random
python models/evaluation.py --model_name gpt-4 --dataset_name ProntoQA --split dev --backup random
cd ..
```

Repeat with `ProofWriter`, `LogicalDeduction`, or `AR-LSAT`. My results match Table 3 of the paper:

| Dataset | Exe_Rate | Exe_Acc |
|---|---|---|
| ProntoQA | 100% | 83.2 |
| ProofWriter | 99.0% | 79.6 |
| LogicalDeduction | 100% | 87.6 |
| AR-LSAT | 32.6% | 60.0 |

"Overall accuracy" varies a little between runs because failed programs get a random backup answer. FOLIO is skipped because it needs the Prover9 binary compiled. Generating new logic programs (`logic_program.py`) needs the retired `text-davinci-003` / 2023 `gpt-4` models and `openai==0.27`, which is why Part 2 has its own LLM wrapper.

### Step 2: Test my reimplementation without an LLM

```bash
cd logic_lm_prolog
python prolog_solver.py          # solver self-test: good queries, a typo, a syntax error, a blocked command
python make_gold.py              # runs the reference queries in Prolog -> data/gym_dev.json (14 questions)
python logic_lm.py --mode oracle # whole pipeline with the gold queries standing in for the LLM: 14/14
```

### Step 3: Connect an LLM (pick one)

**Option A: Ollama, free and local (recommended to start).**

```bash
brew install --cask ollama       # the app; plain `brew install ollama` compiles LLVM from source (hours)
open -a Ollama                   # starts the server; leave the app running
ollama pull llama3.2             # about 2 GB
```

No other setup is needed: the defaults in `llm.py` point at Ollama with `llama3.2`.

**Option B: an API key.** Create `logic_lm_prolog/.env` (it's in `.gitignore`, so never commit it):

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

For Anthropic, use `LLM_BASE_URL=https://api.anthropic.com/v1/` and `LLM_MODEL=claude-haiku-5-5`.

### Step 4: Run Logic-LM and the two baselines

From `logic_lm_prolog/`:

```bash
python logic_lm.py --mode logic-lm --verbose   # LLM writes Prolog, Prolog answers, up to 3 self-refine rounds
python logic_lm.py --mode direct               # baseline: LLM reads kb.pl and answers directly
python logic_lm.py --mode cot                  # baseline: same, but thinking step by step
```

Useful flags:
- `--limit 3`: run only the first 3 questions (a quick smoke test)
- `--max_rounds 0`: turn off self-refinement
- `--backup cot`: fall back to CoT when a program never runs (the paper's "LLM" backup)
- `--model qwen2.5-coder:7b`: try a different model

Each run prints ✓/✗ per question, then Accuracy, Exe_Rate (before and after self-refinement) and Exe_Acc. It saves everything to `outputs/<mode>_<model>.json`, including each round's Prolog goal and error message.

### Step 5: What to look for (for the deliverable)

- **Logic-LM vs. direct vs. CoT accuracy,** especially on the 3-hop questions (`hops` field: gym_10, 12, 13, 14). The actual numbers are in "Part 2 results" below.
- **Exe_Rate before vs. after self-refinement.** Does feeding back Prolog errors fix programs, as in the paper's Table 3?
- **Programs that ran but were wrong** (Exe_Acc < 1). These are the paper's "valid but semantically wrong" translations, the problem the Faithful NeSy team works on.

## Part 2 results (llama3.2, temperature 0)

Rerunning gave identical results. The run files are in `logic_lm_prolog/outputs/`.

| Method | Accuracy | gym_1–9 | gym_10–14 (harder) |
|---|---|---|---|
| Logic-LM | 10/14 (71%) | 9/9 | 1/5 |
| Direct | 6/14 (43%) | 5/9 | 1/5 |
| CoT | 4/14 (29%) | 4/9 | 0/5 |
| Oracle (gold queries) | 14/14 | 9/9 | 5/5 |

gym_10–14 are the four 3-hop questions (gym_10, 12, 13, 14) plus gym_11, a 2-hop question with a negation.

For Logic-LM, Exe_Rate went from 92.9% to 100% after self-refinement: one program, gym_12, needed one round. Exe_Acc was 71.4%.

**Logic-LM errors.** All four wrong answers came from programs that ran but meant something other than the question:
- gym_10: the program dropped the "compound" condition and used an empty equipment list.
- gym_11: the program added a `\+ compound_exercise(X)` the question never asked for.
- gym_12 ("a hamstring exercise that does *not* need a barbell"): the first program required a barbell and then put `\+` on the exercise itself (`\+ E`). Prolog tried to call the exercise name as a goal and raised an error. Self-refinement fixed the error but kept the flipped logic, and the refined program returned `[]` instead of `false`.
- gym_13: the program wrote `A, B ; C` without parentheses, so the operator precedence changed the meaning.

Two answers counted as correct were only right by luck. In gym_2 and gym_14, the program swapped the arguments (`targets(chest, plank)`). That query matched nothing, and "nothing" happened to equal the gold answer `false`.

So self-refinement fixes programs that fail to run, but it can't catch programs that run and mean the wrong thing. It can even produce one, as in gym_12.

**Baselines.** For 4 CoT questions (gym_2, 12, 13, 14) llama3.2 started writing Prolog code and repeated it until it hit the 512-token limit. It never wrote `Answer:`, so these count as failures. Unlike the paper's GPT-4 results, CoT scored below direct prompting with this small model.

**A prompting detail.** When the whole few-shot prompt was sent as one user message, llama3.2 copied the first example (`requires_equipment(push_up, barbell)`) for every question. `few_shot_messages()` sends the paper's prompts as a system message followed by alternating user/assistant example turns instead. The prompt files themselves are unchanged.

## Files

- `logic_lm_prolog/kb.pl`: copy of the Task 4 knowledge base
- `logic_lm_prolog/data/gym_questions.json`: 14 questions with reference Prolog queries; `make_gold.py` turns them into `gym_dev.json`
- `logic_lm_prolog/prompts/`: few-shot formulator prompt, self-correction prompt, and baseline prompts (paper's format: `------` separators, `[[QUESTION]]` slot)
- `logic_lm_prolog/prolog_solver.py`: Prolog solver class. It parses the LLM output, blocks unsafe predicates (`shell`, `assert`…), uses a 5 s time limit, and returns clean error messages for self-refinement
- `logic_lm_prolog/llm.py`: OpenAI-compatible client (Ollama, OpenAI, or Anthropic), plus `few_shot_messages()`, which splits a few-shot prompt into chat turns
- `logic_lm_prolog/logic_lm.py`: the pipeline, baselines and evaluation
