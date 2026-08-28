import os
import sys
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from scripts.logger import ExperimentLogger
from run_experiment import call_groq, check_commonsense, check_math, LANG_SELF_INSTRUCTION

try:
    from groq import Groq
except ImportError:
    raise SystemExit("groq package not installed.")

RESULTS_DIR = "results_qwen_ablation"
N_SAMPLES = 500

MODEL_KEY = "qwen3.6-27b"
MODEL_ID = "qwen/qwen3.6-27b"  # Groq API identifier used in original script

def load_native_dataset(lang: str) -> list[dict]:
    path = Path(f"data/cache/commonsense_native_{lang}.jsonl")
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset

def load_english_dataset(task: str) -> list[dict]:
    path = Path(f"data/cache/{task}_500.jsonl")
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset

def get_arc_system_prompt(input_lang_code, output_lang_code):
    lang_names = {"en": "English", "fi": "Finnish", "sw": "Swahili", "hi": "Hindi"}
    in_lang = lang_names[input_lang_code]
    out_lang = lang_names[output_lang_code]
    
    if output_lang_code == "en":
        output_constraint = (
            f"You MUST write your ENTIRE response in {out_lang} ONLY - "
            "every word, every reasoning step, every sentence. "
            f"Absolutely NO prose in {in_lang} or other languages."
        )
    else:
        output_constraint = (
            f"You MUST write your ENTIRE response in {out_lang} ONLY - "
            "every word, every reasoning step, every sentence. "
            f"Absolutely NO English prose or sentences are allowed."
        )

    prompt = (
        "You are a reasoning assistant. "
        f"The user's question will arrive in {in_lang}. "
        f"{output_constraint} "
        "Think step by step, then give your final answer as a single letter (A/B/C/D) "
        "on the LAST line prefixed with 'Answer:'. "
        f"{LANG_SELF_INSTRUCTION.get(output_lang_code, '')}"
    )
    return prompt

def get_math_system_prompt(output_lang_code):
    lang_names = {"en": "English", "fi": "Finnish", "sw": "Swahili", "hi": "Hindi"}
    out_lang = lang_names[output_lang_code]
    
    if output_lang_code == "en":
        output_constraint = (
            f"You MUST write your ENTIRE response in {out_lang} ONLY - "
            "every word, every reasoning step, every sentence."
        )
    else:
        output_constraint = (
            f"You MUST write your ENTIRE response in {out_lang} ONLY - "
            "every word, every reasoning step, every sentence. "
            "Absolutely NO English prose or sentences are allowed."
        )

    prompt = (
        "You are a math problem solver. "
        "The user's question will arrive in English. "
        f"{output_constraint} "
        "Solve the problem step by step. "
        "Give your final numerical answer on the LAST line, prefixed with 'Answer:' "
        "(the token 'Answer:' may stay in English; only the number follows it). "
        f"{LANG_SELF_INSTRUCTION.get(output_lang_code, '')}"
    )
    return prompt

def run():
    api_key = os.environ.get("GROQ_API_KEY", "")
    client = Groq(api_key=api_key)
    logger = ExperimentLogger(RESULTS_DIR)

    arc_configs = [
        ("en", "en"),
        ("en", "fi"),
        ("fi", "en"),
        ("fi", "fi"),
        ("en", "sw"),
        ("sw", "en"),
        ("sw", "sw"),
        ("en", "hi")
    ]
    
    gsm8k_configs = [
        ("en", "en"),
        ("en", "fi"),
        ("en", "sw")
    ]
    
    en_arc_dataset = load_english_dataset("commonsense")
    fi_arc_dataset = load_native_dataset("fi")
    sw_arc_dataset = load_native_dataset("sw")
    hi_arc_dataset = en_arc_dataset # We don't have native Hindi, so EN->HI uses EN input
    en_gsm8k_dataset = load_english_dataset("math")
    
    all_runs = []
    
    for in_l, out_l in arc_configs:
        dataset = en_arc_dataset
        if in_l == "fi": dataset = fi_arc_dataset
        elif in_l == "sw": dataset = sw_arc_dataset
        
        run_key = f"{MODEL_KEY}__commonsense__{in_l}_to_{out_l}"
        all_runs.append((run_key, "commonsense", in_l, out_l, dataset, check_commonsense, get_arc_system_prompt(in_l, out_l)))

    for in_l, out_l in gsm8k_configs:
        run_key = f"{MODEL_KEY}__math__{in_l}_to_{out_l}"
        all_runs.append((run_key, "math", in_l, out_l, en_gsm8k_dataset, check_math, get_math_system_prompt(out_l)))

    for run_key, task, in_l, out_l, dataset, checker, sys_prompt in all_runs:
        status = logger.get_status(run_key)
        if status == "DONE":
            print(f"SKIP (already DONE): {run_key}")
            continue

        done_indices = logger.completed_indices(run_key)
        print(f"START: {run_key} ({len(done_indices)} done)")
        logger.mark_running(run_key)
        n_errors = 0

        for i, item in enumerate(dataset):
            if i in done_indices:
                continue

            user_prompt = item['question']
            
            result = call_groq(client, MODEL_ID, sys_prompt, user_prompt)

            if result["error"]:
                n_errors += 1
                print(f"  [{i+1:3d}/{N_SAMPLES}] ✗ ERROR: {result['error']}")
                if n_errors >= 10:
                    logger.mark_failed(run_key, f"too_many_errors:{n_errors}")
                    break
                continue

            n_errors = 0
            is_correct = checker(result["content"], item["answer"])

            record = {
                "idx":               i,
                "model":             MODEL_KEY,
                "task":              task,
                "input_language":    in_l,
                "output_language":   out_l,
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
                "model_id":          MODEL_ID,
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
    run()
