"""LLM-as-judge: score each generated story with Llama-3.3-70B via the Groq API.

Reads outputs/eval_generations.json, scores every story on six 1-5 dimensions
plus free-text strengths/weaknesses, and writes outputs/eval_scores.json.

The judge is instructed to grade FITNESS FOR A YOUNG CHILD'S STORY, not raw
sophistication, and to give honest strengths AND weaknesses for every model — so
a fluent-but-off-topic GPT-2 story does not automatically win.

Needs GROQ_API_KEY (free from console.groq.com). Put it in a .env file at the
repo root: GROQ_API_KEY=gsk_...   (this file loads .env itself; it is gitignored).

Run from the repo root:  python scripts/eval_judge.py
"""
import os
import sys
import json
import time
from pathlib import Path

from groq import Groq

REPO = Path(__file__).resolve().parents[1]
MODEL = "llama-3.3-70b-versatile"

DIMENSIONS = ["grammar", "fluency", "coherence",
              "contextual_correctness", "creativity", "plot_completion"]

SYSTEM_PROMPT = """You are evaluating continuations of a young child's story \
(TinyStories style: simple words, for a 3-4 year old). You are given the story \
PROMPT and a model's CONTINUATION.

Score the story 1-5 (5 = best) on each dimension:
- grammar: well-formed English
- fluency: reads smoothly and naturally, not stilted
- coherence: internally consistent (characters, objects, and logic stay consistent; no contradictions)
- contextual_correctness: faithful to the prompt; stays on-topic with what the prompt set up
- creativity: interesting and imaginative, not generic filler
- plot_completion: the story goes somewhere and reaches a small resolution

Grade FITNESS FOR A YOUNG CHILD'S STORY, not raw sophistication. A simple, \
correct, age-appropriate story scores high. A fluent but off-topic, dark, adult, \
or incoherent story scores low even if the vocabulary is advanced.

Be balanced and honest: identify genuine strengths AND genuine weaknesses for \
EVERY story, whatever the model.

Respond with ONLY a JSON object with these keys: grammar, fluency, coherence, \
contextual_correctness, creativity, plot_completion (integers 1-5), plus \
"strengths" and "weaknesses" (one short sentence each)."""


def judge_story(client, prompt, story):
    """Return the parsed score dict for one story, retrying on transient errors."""
    user = f"PROMPT:\n{prompt}\n\nCONTINUATION:\n{story}"
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                temperature=0.0,  # deterministic scoring
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:  # rate limit / transient / parse issue
            wait = 2 ** attempt
            print(f"  retry in {wait}s ({type(e).__name__}: {e})")
            time.sleep(wait)
    raise RuntimeError("judge failed after retries")


def load_dotenv(path):
    """Minimal .env loader (no dependency): set KEY=VALUE lines into os.environ."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def main():
    load_dotenv(REPO / ".env")
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set — add it to .env at the repo root.")

    client = Groq()  # reads GROQ_API_KEY from the environment
    records = json.load(open(REPO / "outputs" / "eval_generations.json"))

    scored = []
    for i, r in enumerate(records, 1):
        print(f"[{i}/{len(records)}] judging {r['model']}")
        result = judge_story(client, r["prompt"], r["story"])
        scored.append({**r, "scores": result})

    out_path = REPO / "outputs" / "eval_scores.json"
    with open(out_path, "w") as f:
        json.dump(scored, f, indent=2)
    print(f"\nwrote {len(scored)} judged stories to {out_path}")


if __name__ == "__main__":
    main()
