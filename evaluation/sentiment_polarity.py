import os
import json
import pandas as pd
from utils.path_utils import get_project_root

PROJECT_ROOT = get_project_root()

INPUT_FILE = PROJECT_ROOT / "classifiers" / "finetuning_llama3-8b_eval_style_results" / "llama3-8b_test_cls.json"  # Replace with path to style-classified file
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results/"  # Replace with path to save style evaluation results

os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_style_metrics(input_file):
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

    toxic_style_count = 0
    neutral_style_count = 0
    polite_style_count = 0

    for item in data:
        style_cls = item.get("style_cls", "").strip().replace("。", "").replace("\n", "")
        if style_cls == "Toxic":
            toxic_style_count += 1
        elif style_cls == "Neutral":
            neutral_style_count += 1
        elif style_cls == "Polite":
            polite_style_count += 1

    toxic_style_ratio = round((toxic_style_count / total_items * 100) if total_items > 0 else 0.0, 2)
    neutral_style_ratio = round((neutral_style_count / total_items * 100) if total_items > 0 else 0.0, 2)
    polite_style_ratio = round((polite_style_count / total_items * 100) if total_items > 0 else 0.0, 2)

    result = {
        "Model Name": model_name,
        "Total Samples": total_items,
        "Toxic (%)": toxic_style_ratio,
        "Neutral (%)": neutral_style_ratio,
        "Polite (%)": polite_style_ratio
    }

    return result, file_name

def main():
    print("Starting sentiment polarity (style) evaluation...")
    result, file_name = calculate_style_metrics(INPUT_FILE)

    if result:
        output_file = os.path.join(OUTPUT_DIR, f"{file_name}_sentiment_polarity.csv")
        df = pd.DataFrame([result])
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Evaluation completed! Results saved to: {output_file}")
        print("\nStyle Metrics Summary:")
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print("Evaluation failed!")

if __name__ == "__main__":
    main()
