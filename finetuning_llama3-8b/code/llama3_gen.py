import os
import json
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

DATA_FILE = "/home/ToxiRewriteCN/finetuning_llama3-8b/data/test_556.json"
PROMPT_DIR = "/home/ToxiRewriteCN/finetuning_llama3-8b/prompt"

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
        # idx = item["idx"]
        
        system_prompt = dataset_prompts[dataset]
        user_input = USER_TEMPLATE.format(sentence=toxic.strip())
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([formatted_prompt], sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text
        item["output"] = response
        # print(response)
    
    json.dump(data, open(OUTPUT_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)



if __name__ == "__main__":
    model_path = "/home/ToxiRewriteCN/finetuning_llama3-8b/output/llama3_8b_r1"

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(temperature=1, top_p=0.5, max_tokens=32768)
    

    # Initialize the vLLM engine
    llm = LLM(model=model_path, gpu_memory_utilization=0.8, tensor_parallel_size=1)
    OUTPUT_FILE = "/finetuning_llama3-8b/eval/llama3-8b_test.json"
    Path(os.path.dirname(OUTPUT_FILE)).mkdir(parents=True, exist_ok=True)
    generate()
