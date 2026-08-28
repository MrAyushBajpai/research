"""
Native-language Ablation Experiment
===================================
Runs the translated ARC-Easy questions (Finnish and Swahili) to test whether
the token bloat and accuracy drop in smaller models was purely a cross-lingual
prompting artifact (English prompt -> FI/SW output) or a genuine property of
generation in the target language (FI/SW prompt -> FI/SW output).

Only tests:
- Llama-3.1-8B (small) and Llama-3.3-70B (large)
- Commonsense task (ARC-Easy)
- Finnish (fi) and Swahili (sw)
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from scripts.logger     import ExperimentLogger
from run_experiment     import call_groq, check_commonsense, LANG_SELF_INSTRUCTION

try:
    from groq import Groq
except ImportError:
    raise SystemExit("groq package not installed. Run: pip install groq")

# Settings for the ablation
MODELS = {
    "llama3.1-8b":  "llama-3.1-8b-instant",
    "llama3.3-70b": "llama-3.3-70b-versatile",
}
LANGUAGES = ["fi", "sw"]
N_SAMPLES = 500
RESULTS_DIR = "results_ablation"

# Expect input in native language.
SYSTEM_PROMPTS = {
    "fi": (
        "Olet järkeilevä avustaja. "
        "Käyttäjän kysymys on suomeksi. "
        "Sinun ON kirjoitettava KOKO vastauksesi VAIN suomeksi - "
        "jokainen sana, jokainen päättelyvaihe, jokainen lause. "
        "Englannin kielen käyttö on EHDOTTOMASTI KIELLETTY. "
        "Mieti askel askeleelta ja anna sitten lopullinen vastauksesi yhtenä kirjaimena (A/B/C/D) "
        "viimeiselle riville etuliitteellä 'Answer:'. "
    ),
    "sw": (
        "Wewe ni msaidizi wa kufikiri. "
        "Swali la mtumiaji litakuwa kwa Kiswahili. "
        "LAZIMA uandike jibu lako LOTE kwa Kiswahili PEKEE - "
        "kila neno, kila hatua ya kufikiri, kila sentensi. "
        "Usiandike sentensi yoyote kwa Kiingereza. "
        "Fikiri hatua kwa hatua, kisha utoe jibu lako la mwisho kama herufi moja (A/B/C/D) "
        "kwenye mstari wa MWISHO ukitanguliwa na 'Answer:'. "
    )
}

def load_native_dataset(lang: str) -> list[dict]:
    path = Path(f"data/cache/commonsense_native_{lang}.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Missing translated dataset: {path}")
    
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset

def run_ablation():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set.")

    client = Groq(api_key=api_key)
    logger = ExperimentLogger(RESULTS_DIR)

    print(f"\n{'='*70}")
    print(f"Native Ablation Experiment")
    print(f"  Models    : {list(MODELS.keys())}")
    print(f"  Task      : commonsense")
    print(f"  Languages : {LANGUAGES}")
    print(f"  Results   : {RESULTS_DIR}/")
    print(f"{'='*70}\n")

    for lang in LANGUAGES:
        try:
            dataset = load_native_dataset(lang)
            print(f"Loaded {len(dataset)} {lang} native questions.")
        except FileNotFoundError as e:
            print(f"SKIPPING {lang}: {e}")
            continue

        for model_key, model_id in MODELS.items():
            run_key = f"{model_key}__commonsense_native__{lang}"
            
            status = logger.get_status(run_key)
            if status == "DONE":
                print(f"SKIP (already DONE): {run_key}")
                continue

            done_indices = logger.completed_indices(run_key)
            if done_indices:
                print(f"RESUME ({len(done_indices)} done): {run_key}")
            else:
                print(f"START: {run_key}")

            logger.mark_running(run_key)
            n_errors = 0

            for i, item in enumerate(dataset):
                if i in done_indices:
                    continue

                # The question is already translated
                user_prompt = item['question']
                sys_prompt = SYSTEM_PROMPTS[lang]

                result = call_groq(client, model_id, sys_prompt, user_prompt)

                if result["error"]:
                    n_errors += 1
                    print(f"  [{i+1:3d}/{N_SAMPLES}] ✗ ERROR: {result['error']}")
                    if n_errors >= 10:
                        logger.mark_failed(run_key, f"too_many_errors:{n_errors}")
                        break
                    continue

                n_errors = 0
                is_correct = check_commonsense(result["content"], item["answer"])

                record = {
                    "idx":               i,
                    "model":             model_key,
                    "task":              "commonsense_native",
                    "language":          lang,
                    "question_id":       item.get("id", str(i)),
                    "system_prompt":     sys_prompt,
                    "user_prompt":       user_prompt,
                    "response":          result["content"],
                    "expected_answer":   str(item["answer"]),
                    "correct":           is_correct,
                    "prompt_tokens":     result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens":      result["total_tokens"],
                    "latency_s":         result["latency_s"],
                    "finish_reason":     result["finish_reason"],
                    "response_length":   len(result["content"]),
                    "model_id":          model_id,
                    "timestamp_utc":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

                logger.append_record(run_key, record)

                status_char = "✓" if is_correct else "✗"
                trunc_flag  = " [TRUNC]" if result["finish_reason"] == "length" else ""
                print(f"  [{i+1:3d}/{N_SAMPLES}] {status_char} comp={result['completion_tokens']:4d} lat={result['latency_s']:.2f}s{trunc_flag}")

                time.sleep(1.2)

            else:
                logger.finalize_run(run_key)
                print(f"  → Run DONE: {run_key}")

if __name__ == "__main__":
    run_ablation()
