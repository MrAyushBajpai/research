import os
import glob
import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

TOKEN_PRICE_PER_M = {
    "llama3.3-70b": {"input": 0.59,  "output": 0.79},
    "llama4-scout":  {"input": 0.11,  "output": 0.34},
    "llama3.1-8b":   {"input": 0.05,  "output": 0.08},
    "gpt-oss-20b":   {"input": 0.075, "output": 0.30},
    "gpt-oss-120b":  {"input": 0.15,  "output": 0.60}
}

def load_data():
    rows = []
    for filepath in glob.glob("results/*.jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                model = d["model"]
                if model not in TOKEN_PRICE_PER_M: continue
                lang = d["language"]
                task = d["task"]
                qid = d["question_id"]
                tot_tok = d["total_tokens"]
                prompt_tok = d["prompt_tokens"]
                comp_tok = d["completion_tokens"]
                correct = 1 if d["correct"] else 0
                
                prices = TOKEN_PRICE_PER_M[model]
                cost = (prompt_tok/1e6)*prices["input"] + (comp_tok/1e6)*prices["output"]
                
                # Check truncation correctly. Some scripts might have "finish_reason" == "length" or "max_tokens"
                fr = d.get("finish_reason")
                truncated = 1 if fr in ("length", "max_tokens") else 0
                
                rows.append({
                    "model": model,
                    "language": lang,
                    "task": task,
                    "question_id": qid,
                    "total_tokens": tot_tok,
                    "cost": cost,
                    "correct": correct,
                    "truncated": truncated
                })
    return pd.DataFrame(rows)

from mixed_effects_interaction import fit_and_test_interaction
from bootstrap_rnorm import bootstrap_all_cells

def main():
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows.")
    
    # Run interaction test
    print("\n--- Mixed Effects Interaction Test ---")
    df_notrunc = df[df["truncated"] == 0]
    full, reduced, lr_stat, df_diff, p_val = fit_and_test_interaction(df_notrunc)
    stat_str = f"Non-truncated (n={len(df_notrunc)}): LR stat={lr_stat:.1f}, df={df_diff}, p={p_val:.4g}"
    print(stat_str)
    
    with open("results/mixed_effects_stats.txt", "w", encoding="utf-8") as f:
        f.write(stat_str + "\n")
    
    # Run bootstrap rnorm
    print("\n--- Bootstrap Rnorm ---")
    rnorm_df = bootstrap_all_cells(df, n_boot=2000, seed=42)
    # Sort to show worst ones
    res = rnorm_df.sort_values("rnorm", ascending=False)
    print(res.head(15).to_string(index=False))
    
    res.to_csv("results/bootstrap_rnorm_results.csv", index=False)
    print("\nSaved full bootstrap results to results/bootstrap_rnorm_results.csv")
    print("Saved stats to results/mixed_effects_stats.txt")

if __name__ == "__main__":
    main()
