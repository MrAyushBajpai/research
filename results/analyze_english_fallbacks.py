import json
import glob
import os
import re
import csv
from lingua import Language, LanguageDetectorBuilder

def main():
    print("Loading language detector...")
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

    REFUSAL_PATTERNS = [
        r"as an ai", r"i'm sorry", r"i am sorry", r"i cannot", r"i can't",
        r"apologize", r"apologies", r"i don't understand", r"language model",
        r"i am a large language model", r"i am unable to", r"does not support",
        r"i'm an ai", r"unfortunately", r"i can only"
    ]

    SHORT_ANSWER_PATTERNS = [
        r"^answer:\s*[a-d]$", r"^answer:\s*\d+", r"^the answer is",
        r"^answer$"
    ]

    def check_pattern(text, patterns):
        text_lower = text.lower()
        for p in patterns:
            if re.search(p, text_lower):
                return True
        return False

    all_fallbacks = []
    files = glob.glob("D:/multilang-token-efficiency/results/*.jsonl")
    
    total_files = len(files)
    print(f"Starting analysis of {total_files} files...")

    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        parts = filename.replace(".jsonl", "").split("__")
        if len(parts) != 3:
            continue
            
        model, task, target_lang = parts
        
        # We only care about non-English targets falling back to English
        if target_lang == 'en':
            continue
            
        expected = lang_map.get(target_lang)
        if not expected:
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                response = data.get("response", "")
                
                if not response.strip():
                    continue
                    
                text_only = re.sub(r'```.*?```', '', response, flags=re.DOTALL).strip()
                if not text_only:
                    text_only = response.strip()
                
                detected = detector.detect_language_of(text_only)
                
                if detected == Language.ENGLISH:
                    # Classify the cause
                    if check_pattern(text_only, REFUSAL_PATTERNS):
                        cause = "Refusal/Apology"
                    elif check_pattern(text_only, SHORT_ANSWER_PATTERNS) or len(text_only.split()) <= 3:
                        cause = "Short Answer / Format Only"
                    elif text_only.isascii():
                        cause = "English Bleed / Hallucination (ASCII)"
                    else:
                        cause = "Other English"
                        
                    all_fallbacks.append({
                        'model': model,
                        'task': task,
                        'target_lang': target_lang,
                        'text': text_only.replace('\n', ' '),
                        'cause': cause
                    })

    # Write full list
    list_path = "D:/multilang-token-efficiency/results/english_fallbacks_list.csv"
    with open(list_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'task', 'target_lang', 'cause', 'text'])
        writer.writeheader()
        writer.writerows(all_fallbacks)
        
    # Aggregate stats
    cause_counts = {}
    for fb in all_fallbacks:
        c = fb['cause']
        cause_counts[c] = cause_counts.get(c, 0) + 1
        
    total_fallbacks = len(all_fallbacks)
    
    pct_path = "D:/multilang-token-efficiency/results/english_fallback_percentages.csv"
    with open(pct_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Cause', 'Count', 'Percentage'])
        for c, count in sorted(cause_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_fallbacks) * 100 if total_fallbacks > 0 else 0
            writer.writerow([c, count, f"{pct:.2f}%"])
            print(f"{c}: {count} ({pct:.2f}%)")
            
    print(f"Total English fallbacks found across all non-English tasks: {total_fallbacks}")

if __name__ == '__main__':
    main()
