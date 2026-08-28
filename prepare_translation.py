import json
from pathlib import Path

def prepare_translation():
    # Load the 500 ARC-Easy questions
    cache_path = Path("data/cache/commonsense_500.jsonl")
    if not cache_path.exists():
        print("Cache file not found!")
        return

    records = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    # Save as a clean JSON for upload
    upload_path = Path("data/cache/commonsense_500_to_translate.json")
    with open(upload_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    prompt = """
I am running an NLP experiment and need to translate a benchmark dataset (ARC-Easy) of 500 multiple-choice questions from English into Finnish and Swahili.

I have attached a JSON file: `commonsense_500_to_translate.json`. It contains a list of objects with the following keys: `id`, `question`, and `answer`. The `question` field contains the question text AND the four multiple-choice options (A, B, C, D). 

Please translate the `question` field for all 500 items into Finnish, and separately into Swahili. 
DO NOT translate the `id` or the `answer` fields (they must remain exactly as they are).
DO NOT translate the letters 'A)', 'B)', 'C)', 'D)'.

Please output two separate JSON files (or code blocks containing the JSON):
1. `commonsense_native_fi.jsonl` (Finnish translations, formatted as JSON Lines)
2. `commonsense_native_sw.jsonl` (Swahili translations, formatted as JSON Lines)

The format for each line in your output must be valid JSON matching the input schema exactly, for example:
{"id": "ARC-Challenge-Test-1", "question": "Mikä on pääkaupunki...\nA) ...\nB) ...\nC) ...\nD) ...", "answer": "B"}
"""
    
    prompt_path = Path("data/cache/translation_prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    
    print(f"Saved JSON payload to: {upload_path}")
    print(f"Saved prompt instructions to: {prompt_path}")
    print("\nPlease upload 'commonsense_500_to_translate.json' to ChatGPT/Claude/Gemini and paste the text from 'translation_prompt.txt'!")

if __name__ == "__main__":
    prepare_translation()
