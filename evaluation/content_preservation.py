import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils.path_utils import get_project_root

PROJECT_ROOT = get_project_root()

INPUT_FILE = PROJECT_ROOT / "classifiers" / "finetuning_llama3-8b_eval_detoxification_results" / "llama3-8b_test_cls.json"  # Replace with path to the classified file
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results/"  # Replace with path to save evaluation results

# Calculate cosine similarity between two vectors
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Load the Chinese text embedding model
print("Loading Chinese text embedding model...")
model = SentenceTransformer("shibing624/text2vec-base-chinese-paraphrase")
print("Model loaded successfully!")

def process_file():
    model_name = os.path.basename(INPUT_FILE).replace('_cls.json', '')  # 去掉_cls后缀，保持模型名统一

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    toxic_texts = []
    rewritten_texts = []
    similarities = []

    print(f"Processing {model_name} with {len(data)} entries...")
    for entry in tqdm(data):
        toxic = entry.get('toxic', '')
        rewritten = entry.get('rewritten', '')

        if toxic and rewritten:
            embeddings = model.encode([toxic, rewritten])
            similarity = cosine_similarity(embeddings[0], embeddings[1])
            similarity_scaled = round(similarity * 100, 2)
            toxic_texts.append(toxic)
            rewritten_texts.append(rewritten)
            similarities.append(similarity_scaled)

    df = pd.DataFrame({
        'Toxic': toxic_texts,
        'Rewritten': rewritten_texts,
        'Content Preservation (×100)': similarities
    })

    avg_similarity = round(np.mean(similarities), 2) if similarities else 0.0

    avg_row = pd.DataFrame({
        'Toxic': ['Average Content Preservation'],
        'Rewritten': [''],
        'Content Preservation (×100)': [avg_similarity]
    })
    result_df = pd.concat([df, avg_row], ignore_index=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'{model_name}_content_preservation.csv')
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Results saved to {output_path}")

    return model_name, avg_similarity

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found - {INPUT_FILE}")
        return

    try:
        model_name, avg_similarity = process_file()
        print(f"\nProcessing complete! Average Content Preservation: {avg_similarity}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
