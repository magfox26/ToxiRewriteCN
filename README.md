# Chinese Toxic Language Mitigation via Sentiment Polarity Consistent Rewrites

## ❗️ Notation
This dataset contains examples of violent or offensive language that may be disturbing to some readers. Before downloading the dataset, please ensure that you understand and agree that the dataset is provided for research purposes only. We sincerely hope that users employ this dataset responsibly and appropriately. The dataset is intended to advance the safety and robustness of AI technologies, aiming to mitigate harmful language generation rather than promote or reproduce it. Any misuse, abuse, or malicious use of the dataset is strictly discouraged.  

## 📄 Paper
The paper has been accepted in EMNLP 2025 (main conference).   
[Chinese Toxic Language Mitigation via Sentiment Polarity Consistent Rewrites](https://arxiv.org/abs/2505.15297)

## ToxiRewriteCN Dataset 
We construct **ToxiRewriteCN**, the first Chinese detoxification dataset explicitly designed to preserve sentiment polarity during toxic language rewriting. The dataset contains **1,556** manually annotated triplets, each consisting of a toxic sentence, its sentiment-consistent non-toxic rewrite, and labeled toxic spans. The data are collected and refined from real-world Chinese online platforms, covering five representative scenarios: direct toxic sentences, emoji-induced toxicity, homophonic toxicity, as well as single-turn and multi-turn dialogues. The dataset is presented in [data/ToxiRewriteCN.json](https://github.com/magfox26/ToxiRewriteCN/blob/main/data/ToxiRewriteCN.json).   
Here we simply describe each fine-grain label.  
| Label             | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| toxic             | The original toxic sentence.                                 |
| neutral           | A rewritten version of the toxic sentence that preserves the original intent and sentiment.  |
| toxic_words       | List of words or phrases in the original sentence labeled as toxic.|
| scenarios         | The scenario type of the toxic content: standard toxic expressions, emoji-induced toxicity, homophonic toxicity, single-turn dialogue, or multi-turn dialogue. |

## 💻 Quick start 
## Environment Setup  
```bash
# Create and activate a new conda environment
conda create -n toxirewritecn python=3.9
conda activate toxirewritecn

# Install required dependencies
pip install -r requirements.txt
```
The project leverages MS-Swift framework for the fine-tuning process.
## 1. Toxicity & Sentiment Polairty Classifiers     
### Toxicity Classifier    
```bash
# Step 1: LoRA fine-tuning for toxicity classification (based on Qwen3-32B)
bash train_tox.sh

# Step 2: Merge LoRA adapters with base model weights
bash merge_tox.sh

# Step 3: Generate toxicity classification results 
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python eval_tox.py --folder /home/ToxiRewriteCN/finetuning_llama3-8b/eval  
```
### Sentiment Polairty Classifier   
```bash
# Step 1: LoRA fine-tuning for style classification (based on Qwen3-32B)
bash train_pol.sh

# Step 2: Merge LoRA adapters with base model weights
bash merge_pol.sh

# Step 3: Generate style classification results
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python eval_pol.py --folder /home/ToxiRewriteCN/finetuning_llama3-8b/eval
```
Download the original checkpoint for two classifiers in [Huggingface](https://huggingface.co/maglyx/ToxiRewriteCN/tree/main)

## 2. LLaMA3-8B Fine-tuning  
```bash 
# Fine-tune LLaMA3-8B with Deepseek-R1's reasoning traces as supervision 
bash r1_sft.sh

# Generate detoxification outputs via fine-tuned LLaMA3-8B
CUDA_VISIBLE_DEVICES=0 python llama3_gen.py
```

## 3. Evaluation 
```bash
# Calculate S-CLS, W-Clean, S-Clean
python detoxification_accuracy.py

# Compute content preservation score
python content_preservation.py

# Calculate BLEU, ChrF++, BERTScore-F1 and COMET
python fluency.py

# Assess sentiment polarity score
python sentiment_polarity.py
```

## Cite
If you find our project useful, we hope you can kindly cite:
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
