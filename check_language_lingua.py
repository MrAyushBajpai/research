import json
import glob
import os
import re
import csv
import sys
from lingua import Language, LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_all_languages().build()

lang_map = {
    'ar': Language.ARABIC,
    'de': Language.GERMAN,
    'en': Language.ENGLISH,
    'es': Language.SPANISH,
    'fi': Language.FINNISH,
    'fr': Language.FRENCH,
    'hi': Language.HINDI,
    'ko': Language.KOREAN,
    'sw': Language.SWAHILI,
    'tr': Language.TURKISH,
    'zh': Language.CHINESE
}

results_summary = {}

out_csv = "D:/multilang-token-efficiency/results/language_detection_accuracy.csv"

# Process up to 150 items per file
SAMPLE_SIZE = 150

with open(out_csv, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['model', 'task', 'target_lang', 'total', 'match', 'accuracy'])
    
    files = glob.glob("D:/multilang-token-efficiency/results/*.jsonl")
    for filepath in files:
        filename = os.path.basename(filepath)
        parts = filename.replace(".jsonl", "").split("__")
        if len(parts) != 3:
            continue
        model, task, target_lang = parts
        
        expected = lang_map.get(target_lang)
        if not expected:
            continue
            
        total = 0
        match = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= SAMPLE_SIZE:
                    break
                data = json.loads(line)
                response = data.get("response", "")
                
                if not response.strip():
                    continue
                    
                text_only = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
                if not text_only.strip():
                    text_only = response
                
                detected = detector.detect_language_of(text_only)
                
                if detected == expected:
                    match += 1
                elif target_lang == 'en' and "Answer:" in text_only:
                     match += 1
                elif target_lang == 'en' and detected == Language.GERMAN and len(text_only.split()) < 5:
                     # Very short strings like "Answer: B" often get detected as German
                     match += 1
                     
                total += 1

        if total > 0:
            acc = match / total
            writer.writerow([model, task, target_lang, total, match, acc])
            f_out.flush()
            
            key = target_lang
            if key not in results_summary:
                results_summary[key] = {'total': 0, 'match': 0}
            results_summary[key]['total'] += total
            results_summary[key]['match'] += match
            print(f"Processed {filename}: {acc:.2%}")

print("\nAccuracy by Language:")
global_match = 0
global_total = 0
for lang in sorted(results_summary.keys()):
    t = results_summary[lang]['total']
    m = results_summary[lang]['match']
    global_match += m
    global_total += t
    print(f"{lang}: {m/t:.2%} ({m}/{t})")

print(f"\nOverall Accuracy: {global_match / global_total:.2%} ({global_match}/{global_total})")
