import json
import pandas as pd
from pathlib import Path
import re
from collections import Counter

def ngrams(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def compute_repetition_metrics(results_dir="results"):
    path = Path(results_dir)
    jsonl_files = list(path.glob("*.jsonl"))
    
    if not jsonl_files:
        print("No JSONL files found.")
        return
        
    records = []
    for f in jsonl_files:
        if "llama4-scout__code" in f.name:
            continue
            
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    text = rec.get("response", "")
                    
                    words = re.findall(r'\w+', text.lower())
                    if len(words) < 3:
                        rep_rate_3 = 0.0
                        unique_ratio = 1.0
                    else:
                        tris = ngrams(words, 3)
                        unique_tris = set(tris)
                        rep_rate_3 = 1.0 - (len(unique_tris) / len(tris)) if tris else 0.0
                        
                        unique_words = set(words)
                        unique_ratio = len(unique_words) / len(words)
                        
                    records.append({
                        "model": rec["model"],
                        "task": rec["task"],
                        "language": rec["language"],
                        "tokens": rec.get("completion_tokens", 0),
                        "rep_rate_3gram": rep_rate_3,
                        "unique_word_ratio": unique_ratio,
                    })

    df = pd.DataFrame(records)
    
    # Aggregate
    agg = df.groupby(["model", "task", "language"]).agg({
        "tokens": "mean",
        "rep_rate_3gram": "mean",
        "unique_word_ratio": "mean"
    }).reset_index()
    
    # Focus on llama3.1-8b
    focus = agg[(agg["model"] == "llama3.1-8b") & (agg["task"] == "commonsense")]
    en = focus[focus["language"] == "en"].iloc[0]
    fi = focus[focus["language"] == "fi"].iloc[0]
    
    print("=== Repetition Metrics (llama3.1-8b commonsense) ===")
    print(f"English: Tokens={en['tokens']:.1f}, 3-Gram Repetition={en['rep_rate_3gram']:.3f}, Unique Word Ratio={en['unique_word_ratio']:.3f}")
    print(f"Finnish: Tokens={fi['tokens']:.1f}, 3-Gram Repetition={fi['rep_rate_3gram']:.3f}, Unique Word Ratio={fi['unique_word_ratio']:.3f}")
    
    agg.to_csv("results/table_repetition.csv", index=False)
    print("\nFull table exported to results/table_repetition.csv")

if __name__ == "__main__":
    compute_repetition_metrics()
