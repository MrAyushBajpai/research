import glob
import json
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy.stats as st
import sys

def tobit_log_likelihood(params, X, y, cens):
    # params: [beta_0, beta_1, ..., beta_k, sigma]
    beta = params[:-1]
    sigma = params[-1]
    
    if sigma <= 0:
        return 1e10
        
    mu = np.dot(X, beta)
    
    # Uncensored
    uncens = (cens == 1)
    ll_uncens = np.sum(st.norm.logpdf(y[uncens], loc=mu[uncens], scale=sigma))
    
    # Censored (right censored at y)
    cens_mask = (cens == 0)
    # log(1 - CDF((y - mu) / sigma)) = log(SF((y - mu) / sigma))
    ll_cens = np.sum(st.norm.logsf(y[cens_mask], loc=mu[cens_mask], scale=sigma))
    
    return -(ll_uncens + ll_cens)

def main():
    print("Loading data...")
    records = []
    for f in glob.glob("results/*.jsonl"):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "total_tokens" not in rec or "finish_reason" not in rec:
                        continue
                        
                    records.append({
                        "model": rec.get("model", rec.get("model_id")),
                        "language": rec.get("language", "en"),
                        "task": rec.get("task", "unknown"),
                        "total_tokens": rec["total_tokens"],
                        "status": 1 if rec["finish_reason"] != "length" else 0
                    })
                except Exception:
                    pass
                    
    df = pd.DataFrame(records)
    df["model"] = df["model"].apply(lambda x: x.split("/")[-1] if "/" in x else x)
    
    print(f"Loaded {len(df)} records.")
    
    # Create main effects dummy variables
    # Dropping first column of each categorical to avoid collinearity
    X_df = pd.get_dummies(df[["model", "language", "task"]], drop_first=True)
    X_df.insert(0, 'Intercept', 1.0)
    X = np.asarray(X_df.values, dtype=float)
    
    # Target is log(total_tokens)
    y = np.log(df["total_tokens"].values)
    cens = df["status"].values
    
    # OLS for initial guess
    beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - np.dot(X, beta_init)
    sigma_init = np.std(residuals)
    
    init_params = np.append(beta_init, sigma_init)
    
    print("Fitting Tobit model (Main Effects)...")
    res = minimize(
        tobit_log_likelihood, 
        init_params, 
        args=(X, y, cens),
        method='L-BFGS-B',
        bounds=[(None, None)] * len(beta_init) + [(1e-5, None)]
    )
    
    if res.success:
        print("Tobit model fitted successfully.")
        
        # Save results
        out_df = pd.DataFrame({
            "Feature": list(X_df.columns) + ["sigma"],
            "Coefficient": res.x
        })
        out_df.to_csv("results/tobit_robustness_results.csv", index=False)
        print("Saved to results/tobit_robustness_results.csv")
    else:
        print("Tobit optimization failed:", res.message)

if __name__ == '__main__':
    main()
