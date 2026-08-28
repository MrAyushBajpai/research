# Reproducibility Code and Data

This repository contains the code and raw data to reproduce the experiments and analyses in the submission. 

## Structure
- `data/`: Contains the evaluation datasets (MBPP, GSM8K, ARC-Easy).
- `scripts/`: Contains the evaluation metric utilities, prompt builders, and dataset loaders.
- `results/`: Contains the raw generated `.jsonl` outputs from all models across 11 languages, and summarized `.csv` tables.
- `results_ablation/`: Raw outputs for the native language prompting ablation.
- `results_qwen_ablation/`: Raw outputs for the cross-family Qwen ablation.
- `run_experiment.py`: Main execution script to query the Groq API.
- `analyze.py`: Generates the figures and tables from the `results/` folder.
- `mcnemar_test.py` & `extra_stats_tests.py`: Scripts used to compute statistical significance (Cochran's Q, McNemar's, Wilcoxon, Mixed-effects regression).

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. To reproduce the tables and plots from the raw generation logs:
   ```bash
   python analyze.py --results_dir results --plots_dir results/plots
   ```

3. To re-run the statistical tests:
   ```bash
   python extra_stats_tests.py
   python mcnemar_test.py
   ```

*(Note: If you intend to re-run the actual generation via `run_experiment.py`, you will need a valid Groq API key set as `GROQ_API_KEY` in your environment variables. See `.env.example`)*
