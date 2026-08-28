import os
import json
import time
import random
from pathlib import Path
try:
    from groq import Groq, RateLimitError
except ImportError:
    raise SystemExit("groq package not installed. Run: pip install groq")

def translate_idiomatically(client: Groq, text: str, target_lang: str) -> str:
    """
    Uses an idiomatic localization prompt to avoid translationese.
    """
    sys_prompt = (
        f"You are an expert bilingual localization specialist. Your task is to translate the following "
        f"multiple-choice question from English into {target_lang}. "
        f"CRITICAL INSTRUCTION: Do NOT translate word-for-word. Avoid 'translationese' at all costs. "
        f"Rewrite the question and options so they sound completely natural, idiomatic, and culturally "
        f"appropriate to a native {target_lang} speaker, as if originally authored in {target_lang}. "
        f"Keep the option letters (A, B, C, D) intact."
    )
    
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_completion_tokens=300
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            wait_time = (2 ** attempt) * 5 + random.uniform(0, 2)
            print(f"    [Rate Limit] Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"    [Error] {e}")
            time.sleep(2)
            
    raise Exception("Max retries exceeded")

def run_translation():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Set GROQ_API_KEY in .env first.")
        return

    client = Groq(api_key=api_key)
    
    cache_path = Path("data/cache/commonsense_500.jsonl")
    records = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    languages = {"fi": "Finnish", "sw": "Swahili"}
    
    for lang_code, lang_name in languages.items():
        out_path = Path(f"data/cache/commonsense_native_{lang_code}.jsonl")
        
        # Load already translated IDs to allow resuming
        done_ids = set()
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        done_ids.add(json.loads(line).get("id"))
                        
        print(f"\nTranslating {len(records)} questions to {lang_name} (Resuming from {len(done_ids)} done)...")
        
        # Open in append mode to write incrementally
        with open(out_path, "a", encoding="utf-8") as f_out:
            for i, rec in enumerate(records):
                if rec["id"] in done_ids:
                    continue
                    
                print(f"  [{i+1}/{len(records)}] Translating ID: {rec['id']}")
                try:
                    native_q = translate_idiomatically(client, rec['question'], lang_name)
                    new_rec = rec.copy()
                    new_rec["question"] = native_q
                    
                    # Write immediately to disk
                    f_out.write(json.dumps(new_rec) + "\n")
                    f_out.flush()
                    
                    # Small courtesy delay between API calls
                    time.sleep(1.0)
                except Exception as e:
                    print(f"Error on {rec['id']}, stopping: {e}")
                    break

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_translation()
