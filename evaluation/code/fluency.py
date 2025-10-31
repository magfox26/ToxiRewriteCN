import json
import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import sacrebleu
from bert_score import score
from comet import download_model, load_from_checkpoint

INPUT_FILE = "/home/ToxiRewriteCN/finetuning_llama3-8b/eval/llama3-8b_test.json"
STANDARD_FILE = "/home/ToxiRewriteCN/data/ToxiRewriteCN.json"
OUTPUT_DIR = "/home/ToxiRewriteCN/evaluation/results/"  

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load COMET model
print("Loading COMET model...")
model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

# Load standard answers
print("Loading standard answers...")
with open(STANDARD_FILE, 'r', encoding='utf-8') as f:
    standard_data = json.load(f)
standard_dict = {item["toxic"]: item["neutral"] for item in standard_data}


def bleu_score(predict, reference, is_sent=True):
    """Calculate BLEU score"""
    if is_sent:
        return sacrebleu.sentence_bleu(predict, [reference], lowercase=True, tokenize="zh").score
    else:
        return sacrebleu.corpus_bleu(predict, [reference], lowercase=True, tokenize="zh").score


def chrfpp_score(predict, reference, is_sent=True):
    """Calculate CHRF++ score"""
    if is_sent:
        return sacrebleu.sentence_chrf(predict, [reference], word_order=2).score
    else:
        return sacrebleu.corpus_chrf(predict, [reference], word_order=2).score


def bertscore_compute_batch(predicts, references):
    """Batch compute BERTScore (only return F1)"""
    _, _, F1 = score(predicts, references, lang="zh", device="cuda" if torch.cuda.is_available() else "cpu")
    return F1.tolist()  # Only return F1 score list


def comet_compute(sources, predictions, references):
    """Calculate COMET scores"""
    data = [{"src": src, "mt": pred, "ref": ref} for src, pred, ref in zip(sources, predictions, references)]
    model_output = comet_model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
    return model_output.scores


def evaluate_single_file():
    file_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    print(f"Evaluating file: {file_name}")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            content = f.read().strip().rstrip(',').rstrip(';')
            if content.startswith('[') and not content.endswith(']'):
                content += ']'
            data = json.loads(content)

    valid_data = []
    for item in data:
        if not all(k in item for k in ['toxic', 'rewritten', 'dataset']):
            print(f"Skipping incomplete sample: {item}")
            continue
        if item['toxic'] in standard_dict:
            valid_data.append({
                'toxic': item['toxic'],
                'rewritten': item['rewritten'],
                'reference': standard_dict[item['toxic']]
            })

    if not valid_data:
        print("No valid samples for evaluation, exiting.")
        return

    toxic_texts = [item['toxic'] for item in valid_data]
    rewritten_texts = [item['rewritten'] for item in valid_data]
    reference_texts = [item['reference'] for item in valid_data]

    # Calculate sentence-level metrics
    print("Calculating sentence-level metrics...")
    sent_bleu = []
    sent_chrfpp = []
    for pred, ref in tqdm(zip(rewritten_texts, reference_texts), total=len(valid_data)):
        sent_bleu.append(bleu_score(pred, ref, is_sent=True))
        sent_chrfpp.append(chrfpp_score(pred, ref, is_sent=True))

    # Calculate BERTScore (only F1)
    print("Calculating BERT-F1 scores...")
    bert_f1 = bertscore_compute_batch(rewritten_texts, reference_texts)

    # Calculate COMET scores
    print("Calculating COMET scores...")
    comet_scores = [round(score * 100, 2) for score in comet_compute(toxic_texts, rewritten_texts, reference_texts)]

    detailed_results = []
    for i, item in enumerate(valid_data):
        detailed_results.append({
            "Original Toxic Text": item['toxic'],
            "Rewritten Text": item['rewritten'],
            "Reference Text": item['reference'],
            "Sentence BLEU": round(sent_bleu[i], 2),
            "Sentence CHRF++": round(sent_chrfpp[i], 2),
            "BERT-F1": round(bert_f1[i] * 100, 2),  # Only retain F1
            "COMET Score": comet_scores[i]  # Already multiplied by 100 and rounded
        })

    print("Calculating overall summary metrics...")
    corpus_bleu = round(bleu_score(rewritten_texts, reference_texts, is_sent=False), 2)
    corpus_chrfpp = round(chrfpp_score(rewritten_texts, reference_texts, is_sent=False), 2)
    avg_bert_f1 = round(np.mean(bert_f1) * 100, 2)
    avg_comet = round(np.mean(comet_scores), 2)
    total_samples = len(valid_data)

    detailed_results.append({k: '' for k in detailed_results[0].keys()})
    detailed_results.append({
        "Original Toxic Text": f"All Samples Summary (Total: {total_samples})",
        "Rewritten Text": "",
        "Reference Text": "",
        "Sentence BLEU": f"Corpus BLEU: {corpus_bleu}",
        "Sentence CHRF++": f"Corpus CHRF++: {corpus_chrfpp}",
        "BERT-F1": f"Avg BERT-F1: {avg_bert_f1}",  # Only retain F1 summary
        "COMET Score": f"Avg COMET Score: {avg_comet}"
    })

    output_file = os.path.join(OUTPUT_DIR, f"{file_name}_fluency.csv")
    pd.DataFrame(detailed_results).to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Evaluation completed. Results saved to: {output_file}")

    print("\n=== Overall Evaluation Scores ===")
    print(f"Total valid samples: {total_samples}")
    print(f"Corpus BLEU: {corpus_bleu}")
    print(f"Corpus CHRF++: {corpus_chrfpp}")
    print(f"Avg BERT-F1: {avg_bert_f1}")
    print(f"Avg COMET Score (×100): {avg_comet}")


if __name__ == "__main__":
    evaluate_single_file()
