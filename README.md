# Chinese Toxic Language Mitigation via Sentiment Polarity Consistent Rewrites

## ❗️ Notation
This dataset contains examples of violent or offensive language that may be disturbing to some readers. Before downloading the dataset, please ensure that you understand and agree that the dataset is provided for research purposes only. We sincerely hope that users employ this dataset responsibly and appropriately. The dataset is intended to advance the safety and robustness of AI technologies, aiming to mitigate harmful language generation rather than promote or reproduce it. Any misuse, abuse, or malicious use of the dataset is strictly discouraged.  

## 📄 Paper
The paper has been accepted in EMNLP 2025 (main conference).   
[Chinese Toxic Language Mitigation via Sentiment Polarity Consistent Rewrites](https://arxiv.org/abs/2505.15297)

## ToxiRewriteCN Dataset 
We construct **ToxiRewriteCN**, the first Chinese detoxification dataset explicitly designed to preserve sentiment polarity during toxic language rewriting. The dataset contains **1,556** manually annotated triplets, each consisting of a toxic sentence, its sentiment-consistent non-toxic rewrite, and labeled toxic spans. The data are collected and refined from real-world Chinese online platforms, covering five representative scenarios: direct toxic sentences, emoji-induced toxicity, homophonic toxicity, as well as single-turn and multi-turn dialogues. The dataset is presented in [dataset/ToxiRewriteCN.json](https://github.com/magfox26/ToxiRewriteCN/blob/main/dataset/ToxiRewriteCN.json).   
Here we simply describe each fine-grain label.  
| Label             | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| toxic             | The original toxic sentence.                                 |
| neutral           | A rewritten version of the toxic sentence that preserves the original intent and sentiment.  |
| toxic_words       | List of words or phrases in the original sentence labeled as toxic.|
| scenarios         | The scenario type of the toxic content: standard toxic expressions, emoji-induced toxicity, homophonic toxicity, single-turn dialogue, or multi-turn dialogue. |

## Quick start 

### Toxicity & Sentiment Classifiers     
#### Environment Setup  
```bash
# Create and activate a new conda environment
conda create -n cls-env python=3.9
conda activate cls-env

# Install required dependencies
pip install -r requirements.txt
```
#### 1.Toxicity Classifier    
```bash
# Step 1: LoRA fine-tuning for toxicity classification (based on Qwen3-32B)
bash lora_qwen3-32b_detox.sh

# Step 2: Merge LoRA adapters with base model weights
bash merge_detox.sh

# Step 3: Generate toxicity classification results (LLaMA3-8B as example)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python eval_detox.py --folder /home/ToxiRewriteCN/finetuning_llama3-8b/eval
```
#### 2.Style Classifier   
```bash
# Step 1: LoRA fine-tuning for style classification (based on Qwen3-32B)
bash lora_qwen3-32b_style.sh

# Step 2: Merge LoRA adapters with base model weights
bash merge_style.sh

# Step 3: Generate style classification results (LLaMA3-8B as example)
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python eval_style.py --folder /home/ToxiRewriteCN/finetuning_llama3-8b/eval
```

### LLaMA3-8B Fine-tuning


### Evaluation 




## Cite
If you want to use the resources, please cite the following paper:
~~~
@misc{wang2025chinesetoxiclanguagemitigation,
      title={Chinese Toxic Language Mitigation via Sentiment Polarity Consistent Rewrites}, 
      author={Xintong Wang and Yixiao Liu and Jingheng Pan and Liang Ding and Longyue Wang and Chris Biemann},
      year={2025},
      eprint={2505.15297},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.15297}, 
}
~~~
