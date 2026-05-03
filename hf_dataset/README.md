---
language:
- zh
pretty_name: ToxiRewriteCN
license: apache-2.0
task_categories:
- text-classification
- text-generation
tags:
- chinese
- toxicity
- detoxification
- text-rewriting
- sentiment-preservation
- safety
size_categories:
- 1K<n<10K
configs:
- config_name: rewrites
  data_files:
  - split: train
    path: data/train_1000.json
  - split: test
    path: data/test_556.json
- config_name: annotated_triplets
  data_files:
  - split: train
    path: data/ToxiRewriteCN.json
- config_name: r1_rewrite_sft
  data_files:
  - split: train
    path: data/r1_train.json
- config_name: toxicity_classifier_sft
  data_files:
  - split: train
    path: data/train_full_8148.json
- config_name: polarity_classifier_sft
  data_files:
  - split: train
    path: data/train_polarity_ratio121.json
---

# ToxiRewriteCN

ToxiRewriteCN is a Chinese toxic language mitigation dataset introduced in the EMNLP 2025 paper [Chinese Toxic Language Mitigation via Sentiment Polarity Consistent Rewrites](https://aclanthology.org/2025.emnlp-main.1808/). It is designed for detoxification and rewriting research where a toxic input is rewritten into a non-toxic sentence while preserving the original sentiment polarity and intent.

The dataset contains harmful, offensive, and disturbing language. It is released for research on safety, detoxification, robustness, and mitigation. Do not use it to generate, amplify, or target abusive language.

## Dataset Summary

The core annotation file contains 1,556 manually annotated Chinese rewrite triplets. Each example includes a toxic sentence, a sentiment-consistent non-toxic rewrite, labeled toxic spans, and a scenario label. The examples cover direct toxic expressions, emoji-induced toxicity, homophonic toxicity, single-turn dialogues, and multi-turn dialogues.

The repository also includes train/test rewrite splits and chat-format data used for classifier or supervised fine-tuning experiments.

## Dataset Configurations

| Config | Split(s) | Source file(s) | Description |
| --- | --- | --- | --- |
| `rewrites` | `train`, `test` | `data/train_1000.json`, `data/test_556.json` | Train/test rewrite split for detoxification experiments. |
| `annotated_triplets` | `train` | `data/ToxiRewriteCN.json` | Full manually annotated triplets with toxic spans and scenario labels. |
| `r1_rewrite_sft` | `train` | `data/r1_train.json` | Chat-format rewrite data with reasoning-trace supervision. |
| `toxicity_classifier_sft` | `train` | `data/train_full_8148.json` | Chat-format toxicity classifier fine-tuning data. |
| `polarity_classifier_sft` | `train` | `data/train_polarity_ratio121.json` | Chat-format sentiment polarity classifier fine-tuning data. |

## Data Fields

### `annotated_triplets`

- `toxic`: original toxic sentence or dialogue context.
- `neutral`: non-toxic rewrite that preserves the original intent and sentiment polarity.
- `toxic_words`: list of toxic spans annotated in the original sentence.
- `scenarios`: scenario type, one of direct toxic sentences, emoji-induced toxicity, homophonic toxicity, single-turn dialogues, or multi-turn dialogues.

### `rewrites`

- `idx`: example id.
- `dataset`: source subset label.
- `toxic`: original toxic sentence or dialogue context.
- `neutral`: sentiment-consistent non-toxic rewrite.
- `polite`: more polite rewrite variant.

### Chat-Format SFT Configurations

- `idx` or `train_id`: example id.
- `dataset`: source subset label, when available.
- `messages`: conversation-style supervised fine-tuning records with `system`, `user`, and `assistant` turns.

## Scenario Distribution

| Scenario | Count |
| --- | ---: |
| direct toxic sentences | 819 |
| emoji-induced toxicity | 49 |
| homophonic toxicity | 39 |
| single-turn dialogues | 615 |
| multi-turn dialogues | 34 |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{{HF_DATASET_REPO_ID}}", "rewrites")
train = dataset["train"]
test = dataset["test"]

annotated = load_dataset("{{HF_DATASET_REPO_ID}}", "annotated_triplets")["train"]
```

## Responsible Use

This dataset is intended for research on detoxification, toxicity detection, rewriting, sentiment preservation, and Chinese NLP safety. The data may include offensive, discriminatory, or otherwise harmful expressions. Users should avoid exposing examples unnecessarily, should not use the data to target individuals or groups, and should evaluate downstream systems for misuse risks and bias.

## License

The source repository is released under the Apache License 2.0.

## Source Repository

GitHub: [PostMindLab/ToxiRewriteCN](https://github.com/PostMindLab/ToxiRewriteCN)

## Citation

```bibtex
@inproceedings{wang-etal-2025-chinese,
    title = "{C}hinese Toxic Language Mitigation via Sentiment Polarity Consistent Rewrites",
    author = "Wang, Xintong and Liu, Yixiao and Pan, Jingheng and Ding, Liang and Wang, Longyue and Biemann, Chris",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    year = "2025",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1808/",
    pages = "35683--35699"
}
```
