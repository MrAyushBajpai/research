import json
import pandas as pd
from pathlib import Path

def load_data(results_dir, task="commonsense", models=None, languages=None):
    path = Path(results_dir)
    jsonl_files = list(path.glob("*.jsonl"))
    records = []
    
    for f in jsonl_files:
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if rec["task"] not in task and not rec["task"].startswith(task):
                        continue
                    if models and rec["model"] not in models:
                        continue
                    if languages and rec["language"] not in languages:
                        continue
                        
                    records.append({
                        "model": rec["model"],
                        "language": rec["language"],
                        "tokens": rec.get("completion_tokens", 0),
                        "correct": rec.get("correct", False),
                        "truncated": rec.get("finish_reason") == "length"
                    })
    return pd.DataFrame(records)

def compute_ablation_metrics():
    models = ["llama3.1-8b", "llama3.3-70b"]
    langs = ["fi", "sw"]
    
    # 1. Load Ablation (Native -> Native)
    df_ablation = load_data("results_ablation", task="commonsense_native", models=models, languages=langs)
    
    # 2. Load Original (English -> Native)
    df_cross = load_data("results", task="commonsense", models=models, languages=langs)
    
    # 3. Load English Baseline (English -> English)
    df_en = load_data("results", task="commonsense", models=models, languages=["en"])
    
    # Aggregate
    def agg(df):
        return df.groupby(["model", "language"]).agg({
            "tokens": "mean",
            "correct": "mean",
            "truncated": "mean"
        }).reset_index()

    agg_ablation = agg(df_ablation).assign(Condition="Native Prompt")
    agg_cross = agg(df_cross).assign(Condition="English Prompt")
    agg_en = agg(df_en).assign(Condition="Baseline (EN)")
    
    combined = pd.concat([agg_en, agg_cross, agg_ablation], ignore_index=True)
    
    print("=== Native Language Ablation Results ===")
    for model in models:
        print(f"\nModel: {model}")
        print(f"{'Condition':<20} | {'Lang':<5} | {'Tokens':<8} | {'Accuracy':<8} | {'Truncation':<10}")
        print("-" * 60)
        
        # English baseline
        en = combined[(combined["model"] == model) & (combined["Condition"] == "Baseline (EN)")]
        if not en.empty:
            en = en.iloc[0]
            print(f"{en['Condition']:<20} | {en['language']:<5} | {en['tokens']:<8.1f} | {en['correct']:.1%}   | {en['truncated']:.1%}")
            
        for lang in langs:
            cross = combined[(combined["model"] == model) & (combined["Condition"] == "English Prompt") & (combined["language"] == lang)]
            native = combined[(combined["model"] == model) & (combined["Condition"] == "Native Prompt") & (combined["language"] == lang)]
            
            if not cross.empty:
                c = cross.iloc[0]
                print(f"{c['Condition']:<20} | {c['language']:<5} | {c['tokens']:<8.1f} | {c['correct']:.1%}   | {c['truncated']:.1%}")
            if not native.empty:
                n = native.iloc[0]
                print(f"{n['Condition']:<20} | {n['language']:<5} | {n['tokens']:<8.1f} | {n['correct']:.1%}   | {n['truncated']:.1%}")
                
    combined.to_csv("results_ablation/table_ablation.csv", index=False)
    print("\nFull table exported to results_ablation/table_ablation.csv")

if __name__ == "__main__":
    compute_ablation_metrics()
