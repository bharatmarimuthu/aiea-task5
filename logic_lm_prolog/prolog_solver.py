"""Symbolic Reasoner for Logic-LM on a Prolog knowledge base.

Mirrors the interface of the paper's solver classes (e.g. Logic-LLM/models/symbolic_solvers/
pyke_solver/pyke_solver.py: Pyke_Program): construct it with the LLM's raw logic program,
check `.flag` to see whether it parsed, then call `execute_program()`, which returns
(answer, error_message). The answer is None when the solver could not run the program,
and the error message is what the self-refiner sends back to the LLM.

Instead of Pyke, this runs the query in SWI-Prolog through Janus.
"""
import re
import janus_swi as janus

# LLM-written code runs inside Prolog, so refuse goals that could touch the system.
BLOCKED = ["shell", "halt", "assert", "retract", "abolish", "consult", "open(", "process_",
           "delete_file", "make_directory", "tmp_file", "load_files", "use_module", "setenv"]

ANSWER_VAR = "X"   # convention from the prompt: "which" questions bind X
TIME_LIMIT = 5     # seconds per query, in case the LLM writes an infinite loop

_loaded_kb = None


def load_kb(path):
    global _loaded_kb
    if _loaded_kb != path:
        janus.consult(path)
        _loaded_kb = path


class Prolog_Program:
    def __init__(self, logic_program: str, kb_path: str = "kb.pl") -> None:
        self.logic_program = logic_program
        load_kb(kb_path)
        self.goal = None
        self.flag = self.parse_logic_program()

    # ---- parsing (paper: parse_logic_program / _parse_segment) ----
    def parse_logic_program(self):
        text = self.logic_program
        if "Query:" in text:
            text = text.split("Query:", 1)[1]
        text = text.split("------")[0]
        lines = [l.split(":::")[0].strip() for l in text.strip().splitlines()]
        lines = [l for l in lines if l and not l.startswith("```")]
        if not lines:
            return False
        goal = " ".join(lines).strip()
        goal = re.sub(r"^\?-\s*", "", goal).rstrip(".").strip()
        if not goal:
            return False
        self.goal = goal
        return True

    # ---- execution (paper: execute_program) ----
    # The goal is passed to Prolog as a string and parsed there with term_string/3, so its
    # variables stay inside Prolog. (Janus errors if a named variable comes back unbound,
    # the same "not sufficiently instantiated" issue as setof in Task 4.)
    QUERY = (
        "catch(( term_string(_G, GoalStr, [variable_names(_Vs)]),"
        "        ( memberchk('" + ANSWER_VAR + "'=_X, _Vs)"
        "        -> Mode = list, call_with_time_limit(" + str(TIME_LIMIT) + ", findall(_X, _G, _L0)),"
        "           sort(_L0, L), T = none"
        "        ;  Mode = bool, L = [],"
        "           ( call_with_time_limit(" + str(TIME_LIMIT) + ", _G) -> T = yes ; T = no ) ),"
        "        Err = none ),"
        "      _E, ( Mode = error, L = [], T = none, term_string(_E, Err) ))"
    )

    def execute_program(self):
        if self.goal is None:
            return None, "Parsing Error"
        lowered = self.goal.lower()
        for word in BLOCKED:
            if word in lowered:
                return None, f"goal uses a forbidden predicate ({word})"
        try:
            res = janus.query_once(self.QUERY, {"GoalStr": self.goal})
        except Exception as e:
            return None, self._clean(str(e))

        if not res.get("truth", True) or res["Mode"] == "error":
            return None, self._clean(res.get("Err", "unknown error"))
        if res["Mode"] == "list":
            return sorted(str(x) for x in res["L"]), ""
        return res["T"] == "yes", ""

    # ---- result interpreter (paper: answer_mapping) ----
    def answer_mapping(self, answer):
        if isinstance(answer, bool):
            return "true" if answer else "false"
        return answer

    @staticmethod
    def _clean(msg):
        msg = str(msg)
        m = re.search(r"existence_error\(procedure,\s*([^),]+)", msg)
        if m:
            return f"procedure {m.group(1).strip()} does not exist"
        m = re.search(r"syntax_error\(([^)]*)\)", msg)
        if m:
            return "syntax error: " + m.group(1).replace("_", " ")
        if "time_limit_exceeded" in msg:
            return f"query did not finish within {TIME_LIMIT} seconds"
        return msg.split("\n")[0][:300]


if __name__ == "__main__":
    # quick self-test, like the __main__ block in the paper's pyke_solver.py
    for prog in ["Query:\ntargets(X, glutes)",
                 "Query:\ncompound_exercise(bench_press)",
                 "Query:\ntarget(X, glutes)",
                 "Query:\nmuscle_workout(chest dumbbell, X)",
                 "Query:\nshell('ls')"]:
        p = Prolog_Program(prog)
        ans, err = p.execute_program()
        print(f"{p.goal!r:45} -> {p.answer_mapping(ans) if ans is not None else None}  {err}")
