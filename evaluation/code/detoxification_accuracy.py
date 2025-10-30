import os
import json
import pandas as pd

INPUT_FILE = "/ltstorage/home/liu/prompt_test/evaluation/分类结果/closed-source_results/deepseek-v3_cls.json"
TOXIC_WORDS_PATH = "/ltstorage/home/liu/emnlp2025/evaluation/Toxic_words.json"
OUTPUT_DIR = "/ltstorage/home/liu/emnlp2025/evaluation/results/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_toxic_words():
    try:
        with open(TOXIC_WORDS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading toxic words file: {str(e)}")
        return []

def calculate_detox_metrics(input_file, toxic_words):
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    model_name = file_name.replace("_cls", "")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {str(e)}")
        return None, None

    total_items = len(data)
    if total_items == 0:
        print("No data found in input file")
        return None, None

    tox_count = 0
    non_tox_count = 0
    total_toxic_words = 0
    total_hit = 0
    per_text_hit = 0

    for i, item in enumerate(data):
        tox_cls = item.get("tox_cls", "").strip().replace("。", "").replace("\n", "")
        if tox_cls == "有毒":
            tox_count += 1
        elif tox_cls == "无毒":
            non_tox_count += 1

        rewritten_text = item.get("rewritten", "")
        if rewritten_text and i < len(toxic_words):
            sample_toxic_words = toxic_words[i].get("toxic_words", [])
            local_hit = False

            for tw in sample_toxic_words:
                total_toxic_words += 1
                if tw in rewritten_text:
                    total_hit += 1
                    local_hit = True

            if local_hit:
                per_text_hit += 1

    non_tox_ratio = round((non_tox_count / total_items * 100) if total_items > 0 else 0.0, 2)
    tox_removal_ratio = round(((1 - (total_hit / total_toxic_words)) * 100) if total_toxic_words > 0 else 100.0, 2)
    text_cleanup_ratio = round(((1 - (per_text_hit / total_items)) * 100) if total_items > 0 else 100.0, 2)

    result = {
        "Model Name": model_name,
        "Total Samples": total_items,
        "S-CLS (%)": non_tox_ratio,
        "W-Clean (%)": tox_removal_ratio,
        "S-Clean (%)": text_cleanup_ratio
    }

    return result, file_name

def main():
    print("Starting detoxification accuracy evaluation...")
    toxic_words = load_toxic_words()
    result, file_name = calculate_detox_metrics(INPUT_FILE, toxic_words)

    if result:
        output_file = os.path.join(OUTPUT_DIR, f"{file_name}_detoxification_accuracy.csv")
        df = pd.DataFrame([result])
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Evaluation completed! Results saved to: {output_file}")
        print("\nMetrics Summary:")
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print("Evaluation failed!")

if __name__ == "__main__":
    main()