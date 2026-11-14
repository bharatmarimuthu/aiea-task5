"""LLM wrapper: an updated version of the paper's OpenAIModel (Logic-LLM/models/utils.py).

The paper's code targets openai==0.27 and models that are now retired (text-davinci-003, the
2023 gpt-4). This uses the current OpenAI SDK against any OpenAI-compatible endpoint, so the
same code runs on:
  - Ollama on your Mac (free):  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=llama3.2
  - OpenAI:                     LLM_BASE_URL=https://api.openai.com/v1  LLM_MODEL=gpt-4o-mini
  - Anthropic:                  LLM_BASE_URL=https://api.anthropic.com/v1/  LLM_MODEL=claude-haiku-5-5
Settings come from environment variables or a .env file next to this script (never commit it).
"""
import os
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path=os.path.join(HERE, ".env")):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


class LLM:
    def __init__(self, model=None, max_new_tokens=512, stop_words=("------",)):
        _load_dotenv()
        self.model = model or os.environ.get("LLM_MODEL", "llama3.2")
        self.client = OpenAI(
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.environ.get("LLM_API_KEY", "ollama"),  # Ollama ignores the key
        )
        self.max_new_tokens = max_new_tokens
        self.stop_words = list(stop_words)

    def generate(self, prompt, temperature=0.0, stop=True):
        # temperature 0, as in the paper (§4: "we set the temperature to 0")
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            stop=self.stop_words if stop else None,
        )
        return (response.choices[0].message.content or "").strip()


def few_shot_messages(prompt, markers=("###", ">>> Correct Program:")):
    """Turn a prompt in the paper's format into chat turns.

    The paper's prompts were written for completion models (text-davinci-003): instructions,
    then examples separated by "------", then the new question left open. A small chat model
    given all of that as one message tends to copy the first example's answer. So: the
    instructions become the system message, each example becomes a user turn (question) and an
    assistant turn (answer), and the new question is the last user turn. Same content, same order.
    """
    blocks = [b.strip() for b in prompt.split("------")]
    messages = [{"role": "system", "content": blocks[0]}]
    for block in blocks[1:-1]:
        marker = next(m for m in markers if m in block)
        question, answer = block.rsplit(marker, 1)
        messages.append({"role": "user", "content": (question + marker).strip()})
        messages.append({"role": "assistant", "content": answer.strip()})
    messages.append({"role": "user", "content": blocks[-1]})
    return messages
