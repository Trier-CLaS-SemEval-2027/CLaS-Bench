import os

def get_test_questions(data_dir="data", k=70):
    """
    Load test questions from language-specific data files, or directly from
    the Hugging Face 'DGurgurov/CLaS-Bench' dataset if local files are missing.
    Can be overridden via CLAS_BENCH_K environment variable.
    """
    k = int(os.environ.get("CLAS_BENCH_K", k))
    test_questions = {}

    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                lang_code = filename.replace(".txt", "")
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    questions = [line.strip() for line in lines if line.strip()][:k]
                    test_questions[lang_code] = questions
                    print(f"Loaded {len(questions)} questions for {lang_code}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    if not test_questions:
        print(f"Local '{data_dir}/' not found; loading 'DGurgurov/CLaS-Bench' from Hugging Face...")
        from datasets import load_dataset
        from collections import defaultdict

        ds = load_dataset("DGurgurov/CLaS-Bench", split="train")
        grouped = defaultdict(list)
        for row in ds:
            lang_code = row["language_code"]
            # Map 'cz' in HF dataset to 'cs' used in steering vector filenames
            if lang_code == "cz":
                lang_code = "cs"
            if len(grouped[lang_code]) < k:
                grouped[lang_code].append(row["question"].strip())

        test_questions = dict(grouped)
        for lang_code, questions in sorted(test_questions.items()):
            print(f"Loaded {len(questions)} questions for {lang_code} (from HF)")

    print(f"\nSuccessfully loaded {len(test_questions)} languages")
    return test_questions