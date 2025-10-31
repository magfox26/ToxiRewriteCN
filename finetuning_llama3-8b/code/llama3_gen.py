import os
import json
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from utils.path_utils import get_project_root

PROJECT_ROOT = get_project_root()

DATA_FILE = PROJECT_ROOT / "finetuning_llama3-8b" / "data" / "test_556.json"  
PROMPT_DIR = PROJECT_ROOT / "finetuning_llama3-8b" / "prompt"

USER_TEMPLATE = "输入：{sentence}"

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset_prompts = {}
datasets = set(item["dataset"] for item in data)
for dataset in datasets:
    prompt_path = os.path.join(PROMPT_DIR, f"{dataset}_prompt.txt")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        dataset_prompts[dataset] = f.read().strip()

def generate():
    for item in tqdm(data):
        toxic = item["toxic"]
        dataset = item["dataset"]
        
        system_prompt = dataset_prompts[dataset]
        user_input = USER_TEMPLATE.format(sentence=toxic.strip())
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([formatted_prompt], sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text
        item["rewritten"] = response
    
    json.dump(data, open(OUTPUT_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    model_path = PROJECT_ROOT / "finetuning_llama3-8b" / "output" / "llama3_8b_r1"  # Replace with path to your trained model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(temperature=1, top_p=0.5, max_tokens=32768)
    llm = LLM(model=model_path, gpu_memory_utilization=0.8, tensor_parallel_size=1)
    
    OUTPUT_FILE = PROJECT_root / "finetuning_llama3-8b" / "eval" / "llama3-8b_test.json"  # Replace with path to save generated results
    Path(os.path.dirname(OUTPUT_FILE)).mkdir(parents=True, exist_ok=True)
    generate()
